import torch
import argparse
import datetime
import logging
import os
import glob
import pickle

import deepspeed
from deepspeed.accelerator import get_accelerator
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import random
import numpy as np
import pandas as pd
import polars as pl
import math
from tqdm import tqdm

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from sklearn.metrics import accuracy_score
from torch.optim import Optimizer

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

from .dataset import mkdir_p
from .iterable_dataset_online_parquet import mask_batch_decoy_1unk_data, mask_spectra_data
from .model import MSGPT

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PKLBatchIterableDataset(torch.utils.data.IterableDataset):
    """Stream PKL files and yield mini-batches directly for large-file training."""

    def __init__(self, pkl_files, batch_size, rank=0, world_size=1, seed=123):
        super().__init__()
        if not pkl_files:
            raise ValueError("未找到任何 pkl 文件，请检查 train_path")
        self.all_files = sorted(pkl_files)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)
        self.seed = int(seed)
        self.epoch = 0
        self._local_files = self.all_files[self.rank::self.world_size]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self._local_files)

    def iter_local_files(self):
        files = list(self._local_files)
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(files)
        return files

    def __iter__(self):
        for fp in self.iter_local_files():
            with open(fp, "rb") as f:
                data = pickle.load(f)

            spectra = torch.as_tensor(data["spectra"])
            spectra_mask = torch.as_tensor(data["spectra_mask"])
            precursors = torch.as_tensor(data["precursors"])
            tokens = torch.as_tensor(data["tokens"])
            labels = torch.as_tensor(data["label"])
            weights = torch.as_tensor(data.get("weight", np.ones_like(data["label"])))

            n = int(labels.shape[0])
            if n <= 0:
                continue

            order = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if idx.numel() == 0:
                    continue
                yield (
                    spectra[idx],
                    spectra_mask[idx],
                    precursors[idx],
                    tokens[idx],
                    labels[idx],
                    weights[idx],
                )


def collect_train_pkl_files(train_path):
    """Support single path or ';' separated multiple paths."""
    paths = [p.strip() for p in str(train_path).split(";") if p.strip()]
    pkl_files = []
    for p in paths:
        pkl_files.extend(glob.glob(os.path.join(p, "*.pkl")))
    return sorted(pkl_files)


def estimate_stream_steps_per_epoch(pkl_files, batch_size, rank=0, world_size=1):
    """Estimate per-rank steps using actual sample counts inside local PKL shards."""
    local_files = sorted(pkl_files)[int(rank)::max(int(world_size), 1)]
    steps = 0
    for fp in local_files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        num_samples = int(len(data["label"]))
        if num_samples <= 0:
            continue
        steps += math.ceil(num_samples / max(int(batch_size), 1))
    return max(steps, 1)

def add_argument():
    parser = argparse.ArgumentParser(description='MSGPT')

    # cuda
    parser.add_argument('--with_cuda',
                        default=True,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--node_num',
                        default=2,
                        type=int,
                        help='the num of mpi node')
    parser.add_argument('--gpu_num',
                        default=2,
                        type=int,
                        help='the total num of gpu')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # train
    parser.add_argument("--config", default="/ajun/dda_bert/yaml/model.yaml")
    parser.add_argument('-f',
                        '--file_size',
                        # default=128*1024,
                        default=256,
                        type=int,
                        help='the size of file')
    

    # deepspeed
    parser.add_argument(
        '--dtype',
        default='bf16',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help='Datatype used for training'
    )
    parser.add_argument(
        '--stage',
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help='Datatype used for training'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


args = add_argument()
print("------------")
print(args.deepspeed)
print(args.deepspeed_config)

def set_seeds(seed):
    logging.info('Unified seeds !!')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # 固定卷积算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError('{} is not a subclass of torch.optim.Optimizer'.format(type(optimizer).__name__))

class WarmupScheduler(object):
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_iter: int,
                 max_iter: int,
                 max_lr: int,
                 min_lr: int,
                 warmup_type: str,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_type = warmup_type
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.last_batch_iteration = last_batch_iteration
        self.org_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def get_exponential_lr_factor(self) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if self.last_batch_iteration <= self.warmup_iter:
            lr_factor *= self.last_batch_iteration / self.warmup_iter
        else:
            lr_factor = (1 - (self.last_batch_iteration - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** 0.9
        print(f'DEBUG by zxf: self.last_batch_iteration is {self.last_batch_iteration}, self.warmup_iter is {self.warmup_iter}, self.max_iter is {self.max_iter}, lr_factor is {lr_factor}')
        return lr_factor

    def get_cosine_lr_factor(self) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if self.last_batch_iteration <= self.warmup_iter:
            lr_factor *= self.last_batch_iteration / self.warmup_iter
        else:
            lr = self.min_lr + \
                 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(self.last_batch_iteration / (self.max_iter - self.warmup_iter) * np.pi))
            lr_factor = lr / self.max_lr
        return lr_factor
        
    def get_lr_ratio(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]

        if self.warmup_type == 'exp':
            ratio = self.get_exponential_lr_factor()
        elif self.warmup_type == 'cos':
            ratio = self.get_cosine_lr_factor()
        else:
            ratio = 1.0
        
        if isinstance(ratio, float):
            ratio = min(1.0, ratio)
        else:
            # when lr_factor is complex, designate lr_factor equal 0 where lr equal min_lr
            ratio = 0.0
        ratio = max(0.0, ratio)
        return ratio

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration

        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        lr_ratio = self.get_lr_ratio()
        print(f'DEBUG by zxf: org_lr is {[org_lr for org_lr in self.org_lrs]}, lr_ratio is {lr_ratio}')
        return [org_lr * lr_ratio if float(org_lr * lr_ratio) > self.min_lr else self.min_lr for org_lr in self.org_lrs]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(len(optimizer.param_groups), param_name,
                                                                           FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


def save_checkpoint(model_engine, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
    save_path = model_engine._get_ckpt_name(save_dir, tag)
    mkdir_p(os.path.join(save_dir, str(tag)))

    zero_optimizer_state = model_engine.zero_optimization() or model_engine.bfloat16_enabled()
    save_frozen_param = model_engine.zero_optimization_partition_gradients() and not exclude_frozen_parameters

    # A hack to save the checkpointing directory. Pipeline parallelism overrides
    # module_state_dict() and uses this path to save the model. module_state_dict()
    # then instead just returns None.  The module_state_dict() implementation in
    # PipelineEngine expects the save path to be set in self._curr_ckpt_path.
    module = model_engine.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters)

    state = dict(module=module,
                 buffer_names=model_engine._get_buffer_names(),
                 optimizer=model_engine.optimizer.state_dict() if model_engine.optimizer and not zero_optimizer_state else None,
                 param_shapes=model_engine._get_zero_param_shapes() if model_engine.optimizer and zero_optimizer_state else None,
                 frozen_param_shapes=model_engine._get_zero_frozen_param_attributes(model_engine._get_param_shape_func)
                 if save_frozen_param else None,
                 shared_params=model_engine._get_shared_params() if model_engine.optimizer and zero_optimizer_state else None,
                 frozen_param_fragments=model_engine._get_zero_frozen_param_attributes(model_engine._get_param_fragment_func)
                 if save_frozen_param else None,
                 lr_scheduler=model_engine.lr_scheduler.state_dict() if model_engine.lr_scheduler is not None else None,
                 data_sampler=model_engine.training_dataloader.data_sampler.state_dict() if
                 (model_engine.training_dataloader is not None and model_engine.curriculum_learning_enabled()) else None,
                 random_ltd=model_engine.random_ltd_scheduler.state_dict() if model_engine.random_ltd_enabled() else None,
                 sparse_tensor_module_names=model_engine.sparse_tensor_module_names,
                 skipped_steps=model_engine.skipped_steps,
                 global_steps=model_engine.global_steps,
                 global_samples=model_engine.global_samples,
                 dp_world_size=model_engine.seq_dp_world_size,
                 mp_world_size=model_engine.mp_world_size,
                 ds_config=model_engine.config)
    state.update(client_state)

    checkpoint_engine = TorchCheckpointEngine()
    checkpoint_engine.save(state, save_path)


def calc_loss(mask, weight, label, pred):
    mask_weight = weight[mask]
    mask_label = label[mask]
    mask_pred = pred[mask]
    
    if len(mask_weight) > 0:
        dda_criterion = nn.BCEWithLogitsLoss(weight=mask_weight)
        dda_loss = dda_criterion(mask_pred, mask_label.flatten())
        return dda_loss.item()
    else:
        return 0

def train(model_engine, trainloader, sw, optim, scheduler, local_device, target_dtype, local_rank, config, epoch, sw_step):
    running_loss, dda_running_loss, mask_running_loss = None, None, None
    target_running_loss, false_target_running_loss, decoy_running_loss = None, None, None

    trainloader.set_epoch(epoch)

    train_bar = tqdm(trainloader, total=len(trainloader))
    for spectra, spectra_mask, precursors, tokens, label, weight in train_bar:
        batch = (spectra, spectra_mask, precursors, tokens, label, weight)
        # mask token （按照unmask取值，进行样本维度的mask操作；sequence层面：40%进行mask（两组））
        spectra, spectra_mask, precursors, tokens, tokens_label, label, weight = mask_batch_decoy_1unk_data(batch, token_mask_ratio=0.4, device=local_device)

        # mask spectra 谱图层面：10%进行mask。
        spectra, spectra_mask = mask_spectra_data(spectra, spectra_mask, device=local_device)

        spectra = spectra.to(local_device)
        spectra_mask = spectra_mask.to(local_device)
        precursors = precursors.to(local_device)
        tokens = tokens.to(local_device)
        tokens_label = tokens_label.to(local_device).to(torch.long)
        label = label.to(local_device)
        weight = weight.to(local_device)

        if target_dtype is not None:
            spectra = spectra.to(target_dtype)
            spectra_mask = spectra_mask.to(target_dtype)
            precursors = precursors.to(target_dtype)
            tokens = tokens.to(target_dtype)
            label = label.to(target_dtype)
            weight = weight.to(target_dtype)

        # forword
        dda_pred, mask_pred = model_engine(spectra, spectra_mask, precursors, tokens)

        # Define dda Loss function
        dda_criterion = nn.BCEWithLogitsLoss(weight=weight)
        dda_loss = dda_criterion(dda_pred, label.flatten())

        # calc loss by type
        target_mask = (label > 0.5)
        target_dda_loss = calc_loss(target_mask, weight, label, dda_pred)

        false_target_mask = (label < 0.5) & (weight < 0.9)
        false_target_dda_loss = calc_loss(false_target_mask, weight, label, dda_pred)

        decoy_mask = (label < 0.5) & (weight > 0.9)
        decoy_dda_loss = calc_loss(decoy_mask, weight, label, dda_pred)

        # Using CrossEntropyLoss function for predicting the masked_token
        mask_criterion = nn.CrossEntropyLoss(ignore_index=0)
        mask_loss = mask_criterion(mask_pred, tokens_label)

        # dda loss，权重系数减少为0.18
        loss = mask_loss + 0.18 * dda_loss

        try:
            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()
        except Exception as e:
            logging.info(f"epoch: {epoch}, rank: {int(os.environ['RANK'])}, error: {e}!!!")
            continue

        # update running_loss
        if running_loss is None:
            running_loss = loss.item()
            dda_running_loss = dda_loss.item()
            mask_running_loss = mask_loss.item()

            target_running_loss = target_dda_loss
            false_target_running_loss = false_target_dda_loss
            decoy_running_loss = decoy_dda_loss
        else:
            running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
            dda_running_loss = 0.99 * dda_running_loss + (1 - 0.99) * dda_loss.item()
            mask_running_loss = 0.99 * mask_running_loss + (1 - 0.99) * mask_loss.item()

            target_running_loss = 0.99 * target_running_loss + (1 - 0.99) * target_dda_loss
            false_target_running_loss = 0.99 * false_target_running_loss + (1 - 0.99) * false_target_dda_loss
            decoy_running_loss = 0.99 * decoy_running_loss + (1 - 0.99) * decoy_dda_loss

        if int(os.environ['RANK']) == 0:
            sw_step += 1

            if sw_step % 1000 == 0:
                scheduler_lr = scheduler.get_last_lr()[0]
                train_bar.set_description(f"[Epoch={epoch}, sw_step={sw_step}, rank={int(os.environ['RANK'])}]")
                train_bar.set_postfix(loss=running_loss, lr=scheduler_lr)

                sw.add_scalar("train/loss", loss.item(), sw_step)
                sw.add_scalar("train/loss_smooth", running_loss, sw_step)

                sw.add_scalar("train/dda_loss", dda_loss.item(), sw_step)
                sw.add_scalar("train/dda_loss_smooth", dda_running_loss, sw_step)

                sw.add_scalar("train/target_loss", target_running_loss, sw_step)
                sw.add_scalar("train/false_target_loss", false_target_running_loss, sw_step)
                sw.add_scalar("train/decoy_loss", decoy_running_loss, sw_step)

                sw.add_scalar("train/mask_loss", mask_loss.item(), sw_step)
                sw.add_scalar("train/mask_loss_smooth", mask_running_loss, sw_step)

                sw.add_scalar("optim/scheduler_lr", scheduler_lr, sw_step)
                sw.add_scalar("optim/epoch", epoch, sw_step)
                sw.flush()

        if (sw_step > 0) and (sw_step % config["ckpt_interval"] == 0):
            logging.info('[%d, %d, %d] save model: %s ' %(epoch, int(os.environ['RANK']), sw_step, config["model_save_path"]))
            save_checkpoint(model_engine, save_dir=config["model_save_path"], tag=epoch, client_state={'epoch': epoch})

    sw.add_scalar("train/train_loss", running_loss, epoch)
    return sw_step


def main():
    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(args.local_rank)
    
    # 加载config
    config_path = args.config
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)
        
    if torch.distributed.get_rank() != 0:
        torch.distributed.barrier()

    vocab = ['<pad>', '<mask>'] + list(config["residues"].keys()) + ['<unk>']
    config["vocab"] = vocab
    config["node_num"] = args.node_num
    config["gpu_num"] = args.gpu_num
    
    # 设置全局seed
    set_seeds(config['seed'])
    
    s2i = {v: i for i, v in enumerate(vocab)}
    logging.info(f"Vocab: {s2i}")
    
    mkdir_p(config["tb_summarywriter"])
    mkdir_p(config["model_save_path"])
    
    config["tb_summarywriter"] = config["tb_summarywriter"] + datetime.datetime.now().strftime("MSGPT_%y_%m_%d_%H_%M")
    sw = SummaryWriter(config["tb_summarywriter"])
    
    if torch.distributed.get_rank() == 0:
        # indicate other ranks can proceed
        torch.distributed.barrier()

    if args.node_num > 1:
        multi_node = True
    else:
        multi_node = False

    train_pkl_files = collect_train_pkl_files(config["train_path"])
    if not train_pkl_files:
        raise RuntimeError(f"未在 train_path 找到 pkl 文件: {config['train_path']}")

    if multi_node:
        rank_for_shard = int(os.environ.get("RANK", 0))
    else:
        rank_for_shard = int(os.environ.get("LOCAL_RANK", 0))

    world_size = max(int(config.get("gpu_num", args.gpu_num)), 1)
    batch_size = int(config["train_batch_size"])
    train_dl = PKLBatchIterableDataset(
        pkl_files=train_pkl_files,
        batch_size=batch_size,
        rank=rank_for_shard,
        world_size=world_size,
        seed=int(config.get("seed", 123)),
    )
    logging.info(f"Streaming pkl files total={len(train_pkl_files)}, local shard files={len(train_dl)}")

    ########################################################################
    # 2. Define a Convolutional Neural Network

    model = MSGPT(
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        vocab_size=len(vocab),
        max_charge=config["max_charge"],
    )
    
    if (config['init_model_path'] is not None) and config['init_model_path'] != '':
        logging.info(f"Loading model checkpoint from '{config['init_model_path']}'")
        model = MSGPT.load_pt(config['init_model_path'], config)
        model = model.to(torch.bfloat16) # 映射模型

    parameters = filter(lambda p: p.requires_grad, model.parameters())
        
    # init optim
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    
    # Update config
    # one_epoch_iters = math.ceil(max(len(train_dl), 1) * int(args.file_size) / int(config['train_batch_size']))
    one_epoch_iters = estimate_stream_steps_per_epoch(
        pkl_files=train_pkl_files,
        batch_size=batch_size,
        rank=rank_for_shard,
        world_size=world_size,
    )
    logging.info(f"streaming steps_per_epoch={one_epoch_iters}, local_rank={rank_for_shard}, world_size={world_size}")

    if args.node_num > 1: # 多机
        max_iters = config["epochs"] * one_epoch_iters * args.node_num
    else: # 单机
        max_iters = config["epochs"] * one_epoch_iters
        
    # 前20 epoch按照exp学习率曲线；后20epoch按照最小学习率曲线
    # max_iters = max_iters // 2
    
    warmup_iters = max(int(config["warmup_ratio"] * max_iters), 1)
    config["train_step_scale"] = max(int(one_epoch_iters * float(config["train_step_ratio"])), 1)
    config["ckpt_interval"] = int(one_epoch_iters * 0.8)
    logging.info(f"Updates max_iters of per epoch is : {max_iters:,},"
                 f" train_step_scale={config['train_step_scale']}, "
                 f" warmup_iters={warmup_iters}, "
                 f" ckpt interval={config['ckpt_interval']}")
    
    # init scheduler
    # scheduler = WarmupScheduler(optim, warmup_iters, max_iters, float(config['learning_rate']), float(config['min_lr']), float(config['second_min_lr']), config['warmup_strategy'])
    scheduler = WarmupScheduler(optim, warmup_iters, max_iters, float(config['learning_rate']), float(config['min_lr']), config['warmup_strategy'])
    
    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) Distributed optimizer
    # 3) DeepSpeed scheduler
    print('args: ', args)
    logging.info(f"rank : {int(os.environ['RANK'])}, local_rank: {int(os.environ['LOCAL_RANK'])}")

    model_engine, optim, _, scheduler = deepspeed.initialize(args=args,
                                                             model=model,
                                                             model_parameters=parameters,
                                                             optimizer=optim,
                                                             lr_scheduler=scheduler)

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype=torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype=torch.half
    logging.info(f"target_dtype: {target_dtype}")

    ########################################################################
    # 4. Train the network
    sw_step = 0
    for epoch in range(config['epochs']):  # loop over the dataset multiple times
        try:
            sw_step = train(model_engine, train_dl, sw, optim, scheduler, local_device, target_dtype, local_rank, config, epoch, sw_step)
        except Exception as e:
            logging.info(f"epoch: {epoch}, rank: {int(os.environ['RANK'])}, error: {e}!!!")
            continue
            
    logging.info('Finished Training')


if __name__ == '__main__':
    main()

### test command:
# nohup deepspeed --bind_cores_to_rank ./train_deepspeed.py --deepspeed --deepspeed_config ./ds_config.json --node_num 1 --gpu_num 1 --config ./yaml/aipc_test_mzml.yaml  > test.log &
