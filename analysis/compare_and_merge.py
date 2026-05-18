import os
import pandas as pd
import glob
import shutil

# 定义路径
dir1 = "/home/yiwen/AIPC/scripts/attantion/mzml_results/results_8"
dir2 = "/home/yiwen/Fragpipe_Sage_alphapept_benchmark/tsv_results_0.2_0.5"
output_dir = "/home/yiwen/AIPC/scripts/attantion/merged_best_results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_1fdr_count(df):
    # 过滤掉无效数据
    if df.empty: return 0
    # 根据 scan_number 去重，保留 score 最高的
    df_dedup = df.sort_values("score", ascending=False).drop_duplicates("scan_number")
    # 统计 label=1 且 q_value <= 0.01 的数量
    if "q_value" in df_dedup.columns:
        return len(df_dedup[(df_dedup["label"] == 1) & (df_dedup["q_value"] <= 0.01)])
    else:
        return len(df_dedup[df_dedup["label"] == 1])

# 重新定义 dir1 和 dir2 以防止路径错误
dir1 = "/home/yiwen/AIPC/scripts/attantion/mzml_results/results_8"
dir2 = "/home/yiwen/Fragpipe_Sage_alphapept_benchmark/tsv_results_0.2_0.5"

# 遍历 dir2 中的所有文件 (作为基准，因为它们有 _result.tsv 后缀)
files2 = glob.glob(os.path.join(dir2, "bas_*_benchmark_result.tsv"))

report = []

for f2_path in files2:
    fname = os.path.basename(f2_path)
    # 对应 dir1 中的文件名 (去掉 _result 或者是匹配前缀)
    # dir1 中的文件命名是 bas_a_testdata_0_benchmark.tsv
    # dir2 中的文件命名是 bas_a_testdata_0_benchmark_result.tsv
    f1_name = fname.replace("_result.tsv", ".tsv")
    f1_path = os.path.join(dir1, f1_name)
    
    # 目标输出文件名 (必须符合 aipc_generate_submit.py 的要求)
    target_name = fname 

    if not os.path.exists(f1_path):
        print(f"Warning: {f1_name} not found in dir1, using dir2.")
        shutil.copy(f2_path, os.path.join(output_dir, target_name))
        continue

    # 读取并比较
    df1 = pd.read_csv(f1_path, sep="\t")
    df2 = pd.read_csv(f2_path, sep="\t")

    count1 = get_1fdr_count(df1)
    count2 = get_1fdr_count(df2)

    if count1 > count2:
        # 如果 dir1 多，则使用 dir1 的文件，但重命名以匹配提交脚本要求
        df1.to_csv(os.path.join(output_dir, target_name), sep="\t", index=False)
        report.append(f"{f1_name}: Dir1 ({count1}) > Dir2 ({count2}) -> Selected Dir1")
    else:
        # 如果 dir2 多或相等，则直接复制 dir2
        shutil.copy(f2_path, os.path.join(output_dir, target_name))
        report.append(f"{f1_name}: Dir2 ({count2}) >= Dir1 ({count1}) -> Selected Dir2")

# 打印简要报告
for line in report[:10]:
    print(line)
print(f"...\nTotal files processed: {len(report)}")
print(f"Results saved to: {output_dir}")
