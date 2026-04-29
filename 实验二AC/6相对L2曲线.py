import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 绘图样式配置 (学术顶级期刊标准)
# ==========================================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
    'figure.autolayout': True
})

# ==========================================================
# 2. 路径配置 (所有文件都应是修改后生成的带有历史误差的 PT 文件)
# ==========================================================
SAVE_DIR = "/3241003007/zy/save"
EXP_DIR = os.path.join(SAVE_DIR, "AC_Experiment")

# 注意：字典里的文件名必须是修改后跑出的新文件！
models_info = {
    "Fair Baseline (FF-PINN)": {
        "path": os.path.join(EXP_DIR, "ac_baseline_pinn_results.pt"),
        "color": "gray", "ls": "--", "lw": 2.5
    },
    "AB-PINN (Dynamic Only)": {
        "path": os.path.join(EXP_DIR, "ac_abpinn_results.pt"),
        "color": "dodgerblue", "ls": "-", "lw": 2.0
    },
    "RAD-PINN (Sampling Only)": {
        "path": os.path.join(EXP_DIR, "ac_rads_pinn_results.pt"),
        "color": "forestgreen", "ls": "-", "lw": 2.0
    },
    "RAD-AB-PINN (Ours)": {
        "path": os.path.join(EXP_DIR, "ac_rads_abpinn_results.pt"), 
        "color": "red", "ls": "-", "lw": 3.0
    }
}

fig, ax = plt.subplots(figsize=(12, 7))
lines_drawn = 0

# 统一对齐的最大步数 (AB-PINN跑得最长，对齐到它)
TARGET_MAX_ITER = 32000

# ==========================================================
# 3. 读取数据并绘制误差历史曲线
# ==========================================================
print(">>> 开始读取数据并绘制 Relative L2 Error 历史图...")
for name, info in models_info.items():
    path = info["path"]
    
    if not os.path.exists(path):
        print(f"❌ 警告: 找不到文件! 模型 [{name}] 的新数据尚未跑出: {path}")
        continue

    # 读取新的 .pt 字典
    data = torch.load(path, weights_only=False, map_location='cpu')
    
    # 【核心提取点】：提取 historical error 数据，而不是 loss 
    err_hist = np.array(data.get("err_history", []))
    final_err = data.get("final_error", None)
    iters = np.array(data.get("iters", []))

    if len(err_hist) == 0:
        print(f"⚠️ [{name}] 文件中未找到历史误差数据，跳过。")
        continue

    # 对齐 X 轴与尾部补齐逻辑
    x_axis = iters
    total_steps = len(err_hist)

    # 補齊尾巴逻辑保持不变，确保右边干净利落
    if len(x_axis) > 0 and x_axis[-1] < TARGET_MAX_ITER:
        x_axis = np.append(x_axis, TARGET_MAX_ITER)
        err_hist = np.append(err_hist, err_hist[-1]) 

    # 组装图例
    label = name
    if final_err is not None:
        label += f" (Final Rel L2: {final_err:.1e})"

    # 【灵魂画线】：semilogy 画对数刻度的相对L2误差
    ax.semilogy(x_axis, err_hist, label=label, color=info["color"], 
                linestyle=info["ls"], linewidth=info["lw"], alpha=0.9)
    lines_drawn += 1

# ==========================================================
# 4. 图表美化与保存
# ==========================================================
if lines_drawn > 0:
    ax.set_title("Ablation Study: Relative $L_2$ Error Convergence (Allen-Cahn Equation)", fontweight='bold', pad=15)
    ax.set_xlabel("Iterations", fontweight='bold')
    ax.set_ylabel("Relative $L_2$ Error (Log Scale)", fontweight='bold')
    
    # 锁定视图
    ax.set_xlim([0, TARGET_MAX_ITER])

    # 网格
    ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')

    # 图例
    ax.legend(loc="lower left", framealpha=0.95, edgecolor='black', shadow=True)

    # 保存新的 PDF
    pdf_path = os.path.join(EXP_DIR, "Ablation_Study_Error_History.pdf")
    png_path = os.path.join(EXP_DIR, "Ablation_Study_Error_History.png")
    
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    print(f"\n🎉 完美！相对 L2 误差历史图生成成功！已保存至 {pdf_path}")