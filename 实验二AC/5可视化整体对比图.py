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
# 2. 路径配置 (务必核对绝对路径)
# ==========================================================
SAVE_DIR = "/3241003007/zy/save"
EXP_DIR = os.path.join(SAVE_DIR, "AC_Experiment")

# 注意：字典里的文件名必须和服务器上的完全一致
models_info = {
    "Fair Baseline (FF-PINN)": {
        "path": os.path.join(EXP_DIR, "ac_fair_baseline_results.pt"),
        "color": "gray", "ls": "--", "lw": 2.5
    },
    "AB-PINN (Dynamic Only)": {
        "path": os.path.join(SAVE_DIR, "ac_abpinn_hard_results.npz"),
        "color": "dodgerblue", "ls": "-", "lw": 2.0
    },
    "RAD-PINN (Sampling Only)": {
        "path": os.path.join(SAVE_DIR, "ac_rads_pinn_results.npz"),
        "color": "forestgreen", "ls": "-", "lw": 2.0
    },
    "RAD-AB-PINN (Ours)": {
        "path": os.path.join(EXP_DIR, "ac_rads_abpinn_results.pt"), 
        "color": "red", "ls": "-", "lw": 3.0
    }
}

fig, ax = plt.subplots(figsize=(12, 7))
lines_drawn = 0

# ==========================================
# 💡 核心修改：将视图强行聚焦到最精彩的 15000 步
# ==========================================
FOCUS_MAX_ITER = 8500

# ==========================================================
# 3. 读取数据并绘制
# ==========================================================
print(">>> 开始检查数据文件并绘制曲线...")
for name, info in models_info.items():
    path = info["path"]
    
    if not os.path.exists(path):
        print(f"❌ 警告: 找不到文件! 模型 [{name}] 的数据路径不存在: {path}")
        continue

    print(f"✅ 成功读取 [{name}] 的数据文件...")
    
    if path.endswith(".pt"):
        data = torch.load(path, weights_only=False, map_location='cpu')
        loss_hist = np.array(data.get("loss_history", []))
        final_err = data.get("final_error", None)
        iters = np.array(data.get("iters", []))
    else: # .npz
        data = np.load(path)
        loss_hist = data.get("loss_history", [])
        final_err = data.get("error_u", None)
        iters = np.array([]) 

    if len(loss_hist) == 0:
        print(f"⚠️ [{name}] 文件中 loss_history 为空，跳过绘制。")
        continue

    total_steps = len(loss_hist)
    if len(iters) > 0 and len(iters) == total_steps:
        x_axis = iters
    else:
        if total_steps < 1000: 
            step_size = 25000 // total_steps
            x_axis = np.arange(total_steps) * step_size
        else:
            x_axis = np.arange(total_steps)

    # 组装图例标签
    label = name
    if final_err is not None:
        label += f" (Rel L2: {final_err:.1e})"

    # 画线
    ax.semilogy(x_axis, loss_hist, label=label, color=info["color"], 
                linestyle=info["ls"], linewidth=info["lw"], alpha=0.9)
    lines_drawn += 1

# ==========================================================
# 4. 图表美化与保存
# ==========================================================
if lines_drawn == 0:
    print("❌ 没有任何有效数据被绘制，请检查你的数据文件！")
else:
    ax.set_title("Ablation Study: Loss Convergence (Allen-Cahn Equation)", fontweight='bold', pad=15)
    ax.set_xlabel("Iterations", fontweight='bold')
    ax.set_ylabel("PDE Loss (Log Scale)", fontweight='bold')
    
    # ==========================================
    # 💡 核心修改：锁定 X 轴显示范围
    # ==========================================
    ax.set_xlim([0, FOCUS_MAX_ITER])

    # 精致网格
    ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')

    # 图例设置
    ax.legend(loc="lower left", framealpha=0.95, edgecolor='black', shadow=True)

    pdf_path = os.path.join(EXP_DIR, "Ablation_Study_Convergence_Full.pdf")
    png_path = os.path.join(EXP_DIR, "Ablation_Study_Convergence_Full.png")
    
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    print(f"\n🎉 完美！成功绘制了 {lines_drawn} 条曲线，视图已聚焦至 {FOCUS_MAX_ITER} 步。")
    print(f"图表已保存至: \n - {pdf_path}\n - {png_path}")