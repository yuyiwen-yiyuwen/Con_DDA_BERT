import pandas as pd
import os
import glob

dir1 = "/home/yiwen/AIPC/scripts/attantion/mzml_cpmpare_results"
dir2 = "/home/yiwen/AIPC/scripts/attantion/mzml_results/results_8"

files1 = set(os.listdir(dir1))
files2 = set(os.listdir(dir2))

# 共有文件 (排除非结果文件)
common_files = [f for f in files1.intersection(files2) if f.endswith(".tsv")]
common_files.sort()

results = []
for f in common_files:
    path1 = os.path.join(dir1, f)
    path2 = os.path.join(dir2, f)
    
    try:
        df1 = pd.read_csv(path1, sep='\t')
        df2 = pd.read_csv(path2, sep='\t')
        
        # 过滤 q_value <= 0.01
        df1 = df1[df1['q_value'] <= 0.01]
        df2 = df2[df2['q_value'] <= 0.01]

        # 在每个 scan 中只保留分数 (score) 最高的 PSM
        if not df1.empty:
            df1 = df1.sort_values('score', ascending=False).drop_duplicates('scan_number')
        if not df2.empty:
            df2 = df2.sort_values('score', ascending=False).drop_duplicates('scan_number')
        
        # 假设通过 scan_number 和 cleaned_sequence 来唯一确定一个 PSM
        intersection = pd.merge(df1, df2, on=['scan_number', 'cleaned_sequence', 'precursor_charge'], how='inner')
        
        results.append({
            "File": f,
            "Count1": len(df1),
            "Count2": len(df2),
            "Intersection": len(intersection)
        })
    except Exception as e:
        print(f"Error processing {f}: {e}")

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
