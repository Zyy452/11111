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
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
    'figure.autolayout': True
})

# ==========================================================
# 2. 路径配置
# ==========================================================
SAVE_DIR = "/3241003007/zy/save"
EXP_DIR = os.path.join(SAVE_DIR, "AC_Experiment")

# 注意：请确保这些文件名与你前面 5 个脚本中 torch.save 的名字完全一致
models_info = {
    "Fair Baseline (FF-PINN)": {
        "path": os.path.join(EXP_DIR, "ac_fair_baseline_results.pt"),
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
        "path": os.path.join(EXP_DIR, "ac_rads_abpinn_results2.pt"), 
        "color": "red", "ls": "-", "lw": 3.0
    }
}

# ==========================================================
# 3. 读取数据并绘制误差历史曲线
# ==========================================================
fig, ax = plt.subplots(figsize=(11, 6))
lines_drawn = 0
TARGET_MAX_ITER = 20000

print(">>> 开始读取数据并绘制 Relative L2 Error 历史图...")

for name, info in models_info.items():
    path = info["path"]
    
    if not os.path.exists(path):
        print(f"❌ 警告: 找不到文件! 模型 [{name}] 的数据尚未跑出: {path}")
        continue

    # 显式使用 CPU 加载，防止本地绘图时找不到 GPU
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    err_hist = np.array(data.get("err_history", []))
    final_err = data.get("final_error", None)
    iters = np.array(data.get("iters", []))

    if len(err_hist) == 0:
        print(f"⚠️ [{name}] 文件中未找到历史误差数据，跳过。")
        continue

    # 对齐 X 轴逻辑
    x_axis = iters
    
    # ================= 🚀 核心修改区域 =================
    # 补齐尾巴，让曲线一直延伸到 TARGET_MAX_ITER 处
    if len(x_axis) > 0 and x_axis[-1] < TARGET_MAX_ITER:
        x_axis = np.append(x_axis, TARGET_MAX_ITER)
        # 修复 Bug：如果存在 final_err (即经历了 L-BFGS)，则尾端收敛到 final_err
        # 否则才使用 Adam 阶段的最后一次误差
        actual_final = final_err if final_err is not None else err_hist[-1]
        err_hist = np.append(err_hist, actual_final) 
    # ====================================================

    # 组装图例内容
    label_text = name
    if final_err is not None:
        label_text += f" (Err: {final_err:.2e})"

    # 绘制对数纵轴曲线
    ax.semilogy(x_axis, err_hist, label=label_text, color=info["color"], 
                linestyle=info["ls"], linewidth=info["lw"], alpha=0.9)
    lines_drawn += 1

# ==========================================================
# 4. 图表美化与保存
# ==========================================================
if lines_drawn > 0:
    ax.set_title("Convergence of Relative $L_2$ Error (Allen-Cahn Equation)", fontweight='bold', pad=15)
    ax.set_xlabel("Adam (+ L-BFGS) Iterations", fontweight='bold') # 顺手帮你把横坐标标签改严谨了
    ax.set_ylabel("Relative $L_2$ Error (Log Scale)", fontweight='bold')
    
    ax.set_xlim([0, TARGET_MAX_ITER])
    # 自动设置纵轴范围，通常误差在 1e-4 到 1 之间
    ax.set_ylim([1e-4, 2])

    ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')

    # 图例放在左下角，避开曲线集中的地方
    ax.legend(loc="lower left", framealpha=0.95, edgecolor='black', shadow=False)

    pdf_path = os.path.join(EXP_DIR, "Ablation_Study_Error_History2.pdf")
    png_path = os.path.join(EXP_DIR, "Ablation_Study_Error_History2.png")
    
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    print(f"\n🎉 完美！消融对比图已保存至: \nPDF: {pdf_path}\nPNG: {png_path}")
else:
    print("\n😢 没有绘制出任何曲线，请确认各个模型的 .pt 文件是否已经生成。")