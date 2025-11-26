log_files = {
    "Original (Baseline)": "/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-original-probe/training_log.txt",   # 你的 Original Log 路径
    "Residual (Ours)": "/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-residual-probe-update_II/training_log.txt",       # 你的 Residual Log 路径
    "Normalized (Ours)": "/mnt/bn/rai/zout/coconut/checkpoints/gsm-coconut/gsm-normalized-probe-update_II/training_log.txt"    # 你的 Normalized Log 路径
}

import json
import matplotlib.pyplot as plt
import re
import numpy as np

# --- 1. 数据读取与处理工具 ---

def parse_log(file_path):
    steps = []
    cosines = []
    alphas = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if "step" in data and "avg_cosine" in data:
                        steps.append(data["step"])
                        cosines.append(data["avg_cosine"])
                        # Original 模式可能没有 avg_alpha
                        alphas.append(data.get("avg_alpha", 0.0)) 
                except json.JSONDecodeError:
                    continue
    return steps, cosines, alphas

def smooth_curve(points, factor=0.9):
    """使用指数移动平均 (EMA) 进行平滑"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

colors = {
    "Original (Baseline)": "grey", 
    "Residual (Ours)": "#d62728",    # 红色
    "Normalized (Ours)": "#1f77b4"   # 蓝色
}

# 读取数据并对齐长度
data_cache = {}
min_length = float('inf')

print("Loading data...")
for label, path in log_files.items():
    try:
        steps, cosines, alphas = parse_log(path)
        if len(steps) > 0:
            data_cache[label] = {"steps": steps, "cosine": cosines, "alpha": alphas}
            min_length = min(min_length, len(steps))
            print(f"  Loaded {label}: {len(steps)} steps")
    except FileNotFoundError:
        print(f"  Error: File not found: {path}")

print(f"Cutting to first {min_length} steps for alignment.")

# --- 图表 1: 平滑后的 Cosine 对比图 ---
def plot_smooth_cosine():
    plt.figure(figsize=(10, 6), dpi=150)
    
    for label, data in data_cache.items():
        # 截断
        x = data["steps"][:min_length]
        y = data["cosine"][:min_length]
        
        # 平滑 (Factor 越大越平滑)
        y_smooth = smooth_curve(y, factor=0.95)
        
        # 样式设置
        lw = 2.5 if "Ours" in label else 2.0
        alpha = 1.0 if "Ours" in label else 0.5
        ls = "--" if "Original" in label else "-"
        
        plt.plot(x, y_smooth, label=label, color=colors[label], 
                 linestyle=ls, linewidth=lw, alpha=alpha)

    plt.title("Latent Thought Similarity (Smoothed)", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Avg Cosine Similarity", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='lower left')
    
    # 聚焦关键区域 (根据你的数据调整)
    plt.ylim(0.80, 0.98) 
    
    plt.tight_layout()
    plt.savefig("cosine_smoothed.png")
    print("Saved: cosine_smoothed.png")
    plt.show()

# --- 图表 2: 双轴图 (左 Cosine, 右 Alpha) ---
def plot_dual_axis():
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
    ax2 = ax1.twinx()  # 创建共享 X 轴的第二个 Y 轴

    # 绘图数据
    for label, data in data_cache.items():
        x = data["steps"][:min_length]
        
        # 1. 左轴画 Cosine (实线)
        y_cos = smooth_curve(data["cosine"][:min_length], factor=0.95)
        
        # Original 画虚线背景，Ours 画实线
        ls = ":" if "Original" in label else "-"
        alpha_line = 0.4 if "Original" in label else 1.0
        
        ax1.plot(x, y_cos, label=f"{label} (Cosine)", color=colors[label], 
                 linestyle=ls, linewidth=2, alpha=alpha_line)

        # 2. 右轴画 Alpha (点划线/填充)，仅针对 Ours
        if "Ours" in label:
            y_alpha = smooth_curve(data["alpha"][:min_length], factor=0.95)
            
            # 使用填充区域或者虚线来表示 Alpha，避免线条太乱
            # ax2.plot(x, y_alpha, label=f"{label} (Alpha)", color=colors[label], 
            #          linestyle="-.", linewidth=1.5, alpha=0.6)
            
            # 或者使用填充图更直观：
            ax2.fill_between(x, 0, y_alpha, color=colors[label], alpha=0.1, label=f"{label} (Alpha Area)")

    # 设置轴标签
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Cosine Similarity (Lines)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Alpha Intensity (Areas)", fontsize=12, fontweight='bold')
    
    # 设置标题
    plt.title("Correlation: Logic Jumps (Low Cosine) vs. Attention Intensity (High Alpha)", 
              fontsize=14, fontweight='bold')
    
    # 设置范围
    ax1.set_ylim(0.80, 1.0)  # Cosine 范围
    ax2.set_ylim(0, 0.5)     # Alpha 范围 (根据你的日志 0.2~0.3 调整)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, -0.1), ncol=3) # 图例放到底部

    ax1.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("dual_axis_analysis.png")
    print("Saved: dual_axis_analysis.png")
    plt.show()

# --- 执行绘图 ---
plot_smooth_cosine()
plot_dual_axis()