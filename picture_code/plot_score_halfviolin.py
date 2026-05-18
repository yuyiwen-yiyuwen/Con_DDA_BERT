"""随机抽取5个 train score 文件，绘制 target/decoy 分数的半小提琴图"""
import glob, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

score_dir = "/home/yiwen/AIPC/scripts/attantion/train_tims_all_psm_score/train"
output_path = "/home/yiwen/AIPC/score_halfviolin_epoch0.png"
seed = 42

# ── 随机抽取5个文件 ──
all_files = sorted(glob.glob(os.path.join(score_dir, "*_all_psm_score.tsv")))
rng = random.Random(seed)
selected = rng.sample(all_files, 5)
selected_names = [os.path.basename(f) for f in selected]
print(f"选中文件: {selected_names}")

# ── 加载数据 ──
data = {"target": [], "decoy": []}
for f in selected:
    df = pd.read_csv(f, sep="\t", usecols=["label", "model_score"])
    targets = df.loc[df["label"] > 0.5, "model_score"].dropna().values
    decoys = df.loc[df["label"] <= 0.5, "model_score"].dropna().values
    data["target"].append(targets)
    data["decoy"].append(decoys)
    print(f"  {os.path.basename(f)}: target={len(targets)}, decoy={len(decoys)}")

# ── 绘图 ──
fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=True)
colors = {"target": "#e74c3c", "decoy": "#3498db"}
titles = [n.replace("_all_psm_score.tsv", "").replace("train.", "") for n in selected_names]

for i, ax in enumerate(axes):
    for label, side, col in [("target", "low", colors["target"]), ("decoy", "high", colors["decoy"])]:
        values = data[label][i]
        if len(values) == 0:
            continue
        parts = ax.violinplot(
            values, positions=[0], vert=True, showmeans=True,
            showmedians=False, showextrema=False, side=side,
        )
        for body in parts["bodies"]:
            body.set_facecolor(col)
            body.set_alpha(0.7)
            body.set_edgecolor(col)
            body.set_linewidth(0.8)
        # mean marker
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(2)

    ax.set_title(titles[i], fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(-0.05, 1.05)

axes[0].set_ylabel("Model Score", fontsize=13)

# 图例
legend_patches = [
    mpatches.Patch(color=colors["target"], alpha=0.7, label="Target (left)"),
    mpatches.Patch(color=colors["decoy"], alpha=0.7, label="Decoy (right)"),
]
fig.legend(handles=legend_patches, loc="upper center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.97))

fig.suptitle("Target vs Decoy Score Distribution (Half-Violin, 5 Random Train Files)", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n图片已保存: {output_path}")
