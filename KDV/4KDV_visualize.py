import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import os

# ================= 1. 路径配置 =================
SAVE_DIR = "/3241003007/zy/save"
DATA_LOAD_PATH = os.path.join(SAVE_DIR, "KDV_rads_abpinn_results.pt")

if not os.path.exists(DATA_LOAD_PATH):
    raise FileNotFoundError(f"找不到数据文件：{DATA_LOAD_PATH}，请先确认训练脚本已成功运行！")

# ================= 2. 读取数据 =================
print(f"📂 正在加载 KDV 数据...")
data = torch.load(DATA_LOAD_PATH, map_location='cpu', weights_only=False)

x_exact = data["x_exact"].flatten()
t_exact = data["t_exact"].flatten()
u_exact = data["u_exact"]
X_mesh = data["X_mesh"]
T_mesh = data["T_mesh"]
u_pred = data["u_pred"]
final_error = data["final_error"]

centers = data.get("centers", [])
gamma = data.get("gamma", 40.0)
x_vis = data.get("x_vis", np.array([]))
t_vis = data.get("t_vis", np.array([]))

# ================= 3. 绘制高质量 Figure 3 =================
print("🖼️ 正在生成 KDV 统一坐标轴图像...")
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'font.family': 'serif'
})

fig = plt.figure(figsize=(18, 10))
# 使用更高的 hspace 确保标题不重叠
gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.28)

# 获取全局统一范围
t_min, t_max = t_exact.min(), t_exact.max()
x_min, x_max = x_exact.min(), x_exact.max()

# 统一热力图绘图函数，减少重复代码并确保范围一致
def plot_heatmap(ax, X, T, Z, title, cmap='seismic', has_points=False):
    cax = ax.pcolormesh(T, X, Z, cmap=cmap, shading='auto', rasterized=True)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Space (x)")
    # 【关键】统一坐标轴，消除空白
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(x_min, x_max)
    return cax

# ---------- 子图 (a): 真实解 ----------
ax1 = fig.add_subplot(gs[0, 0])
plot_heatmap(ax1, X_mesh, T_mesh, u_exact, "(a) Exact Solution (KdV)")

# ---------- 子图 (b): 预测解 + 采样点 + 专家中心 ----------
ax2 = fig.add_subplot(gs[0, 1])
plot_heatmap(ax2, X_mesh, T_mesh, u_pred, f"(b) RAD-AB-PINN Prediction\n(Rel $L_2$ Error: {final_error:.2e})")

# 绘制采样点
if len(x_vis) > 0:
    plot_points = min(2500, len(t_vis))
    idx_plot = np.random.choice(len(t_vis), plot_points, replace=False)
    # 确保采样点也在统一范围内
    ax2.scatter(t_vis[idx_plot], x_vis[idx_plot], color='black', s=1.0, alpha=0.3, zorder=2, label='RADS Points')

# 绘制专家网络锚点
if len(centers) > 0:
    radius = 1.0 / np.sqrt(gamma) if gamma > 0 else 0.2
    for i, (cx, ct) in enumerate(centers):
        label_c = 'Expert Center' if i == 0 else ""
        ax2.scatter(ct, cx, marker='*', color='gold', s=180, edgecolor='black', linewidth=0.8, zorder=5, label=label_c)
        circle = patches.Circle((ct, cx), radius=radius, fill=False, 
                                edgecolor='gold', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
        ax2.add_patch(circle)
ax2.legend(loc="upper left", framealpha=0.7)

# ---------- 子图 (c): 绝对误差图 ----------
ax3 = fig.add_subplot(gs[0, 2])
err_map = np.abs(u_exact - u_pred)
plot_heatmap(ax3, X_mesh, T_mesh, err_map, "(c) Absolute Error Map", cmap='inferno')

# ---------- 子图 (d)-(f): 时间切片对比 ----------
t_span = t_max - t_min
slice_times = [t_min + 0.15 * t_span, t_min + 0.5 * t_span, t_min + 0.85 * t_span] 

for idx, t0 in enumerate(slice_times):
    ax_slice = fig.add_subplot(gs[1, idx])
    t_idx = (np.abs(t_exact - t0)).argmin()
    
    u_true_plot = u_exact[t_idx, :] 
    u_pred_plot = u_pred[t_idx, :]
    
    ax_slice.plot(x_exact, u_true_plot, 'b-', label="Exact", linewidth=2.0)
    ax_slice.plot(x_exact, u_pred_plot, 'r--', label="Ours", linewidth=2.0)
    
    char = chr(ord('d') + idx) 
    stage = ["Initial/Before", "Collision", "After Collision"][idx]
    ax_slice.set_title(f"({char}) {stage} (t = {t0:.2f})")
    ax_slice.set_xlabel("Space (x)")
    if idx == 0: ax_slice.set_ylabel("u(x,t)")
    ax_slice.set_xlim(x_min, x_max) # 统一切片图横轴
    ax_slice.grid(True, linestyle=':', alpha=0.6)
    ax_slice.legend(loc='upper right')

# ================= 4. 导出结果 =================
# 使用 tight_layout 替代 figure.autolayout 避免比例失真
# plt.tight_layout() 
PDF_SAVE_PATH = os.path.join(SAVE_DIR, "Figure_3_KDV_Aligned.pdf")
plt.savefig(PDF_SAVE_PATH, format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(PDF_SAVE_PATH.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=300)

print(f"✅ 可视化完成！坐标轴已统一。")
print(f"📄 保存路径: {PDF_SAVE_PATH}")