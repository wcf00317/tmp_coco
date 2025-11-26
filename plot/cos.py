log_files = {
    "Original (Baseline)": "/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-original-probe/training_log.txt",   # 你的 Original Log 路径
    "Residual (Ours)": "/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-residual-probe-update_II/training_log.txt",       # 你的 Residual Log 路径
    "Normalized (Ours)": "/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-normalized-probe-update_II/training_log.txt"    # 你的 Normalized Log 路径
}

import json
import matplotlib.pyplot as plt
import re

def parse_log(file_path):
    steps = []
    cosines = []
    alphas = [] # 顺便把 alpha 也读出来，万一你想画双轴
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if "step" in data and "avg_cosine" in data:
                        steps.append(data["step"])
                        cosines.append(data["avg_cosine"])
                        # Original 模式可能没有 avg_alpha，给个默认值 0
                        alphas.append(data.get("avg_alpha", 0.0)) 
                except json.JSONDecodeError:
                    continue
    return steps, cosines, alphas

# --- 1. 先读取所有数据 ---
data_cache = {}
min_length = float('inf') # 用于记录最短步数

print("Loading data...")
for label, path in log_files.items():
    try:
        steps, cosines, _ = parse_log(path)
        if len(steps) > 0:
            data_cache[label] = (steps, cosines)
            min_length = min(min_length, len(steps)) # 更新最短长度
            print(f"  Loaded {label}: {len(steps)} steps")
        else:
            print(f"  Warning: {label} is empty!")
    except FileNotFoundError:
        print(f"  Error: File not found: {path}")

print(f"Cutting all plots to the first {min_length} steps.")

# --- 2. 绘图设置 ---
plt.figure(figsize=(10, 6), dpi=150)

colors = {
    "Original (Baseline)": "grey", 
    "Residual (Ours)": "#d62728",    # 红色
    "Normalized (Ours)": "#1f77b4"   # 蓝色
}
styles = {
    "Original (Baseline)": "--", 
    "Residual (Ours)": "-", 
    "Normalized (Ours)": "-"
}
alphas_val = {
    "Original (Baseline)": 0.6,
    "Residual (Ours)": 1.0,
    "Normalized (Ours)": 0.9
}

# --- 3. 截断并画图 ---
for label, (steps, cosines) in data_cache.items():
    # 核心修改：只取前 min_length 个点
    cut_steps = steps[:min_length]
    cut_cosines = cosines[:min_length]
    
    # 平滑处理 (可选，让曲线更平滑好看)
    def smooth(data, weight=0.9):
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # 如果觉得噪点太多，可以把下面这就话取消注释
    # cut_cosines = smooth(cut_cosines, 0.9)

    plt.plot(cut_steps, cut_cosines, 
             label=label, 
             color=colors[label], 
             linestyle=styles[label], 
             alpha=alphas_val[label], 
             linewidth=2)

# --- 装饰 ---
plt.title("Evolution of Latent Thought Similarity (Aligned)", fontsize=14, fontweight='bold')
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Avg Cosine Similarity", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='lower left')

# 自动调整 Y 轴范围，聚焦在数据波动区域
all_cut_cosines = []
for _, cosines in data_cache.values():
    all_cut_cosines.extend(cosines[:min_length])
    
if all_cut_cosines:
    y_min = min(all_cut_cosines)
    y_max = max(all_cut_cosines)
    # 稍微留点边距
    plt.ylim(max(0.7, y_min - 0.02), min(1.0, y_max + 0.01))

plt.tight_layout()
plt.savefig("cosine_comparison_aligned.png")
plt.show()
print("Plot saved to cosine_comparison_aligned.png")