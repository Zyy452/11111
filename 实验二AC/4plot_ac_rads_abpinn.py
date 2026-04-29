import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import os
import datetime

# ================= 1. 路径配置 =================
EXPERIMENT_DIR = "/3241003007/zy/save/AC_Experiment"
DATA_LOAD_PATH = os.path.join(EXPERIMENT_DIR, "ac_rads_abpinn_results.pt")

if not os.path.exists(DATA_LOAD_PATH):
    raise FileNotFoundError(f"找不到数据文件：{DATA_LOAD_PATH}，请先运行 train_ac.py！")

# ================= 2. 读取数据 =================
data = torch.load(DATA_LOAD_PATH, weights_only=False)

x_exact = data["x_exact"].flatten()
t_exact = data["t_exact"].flatten()
u_exact_all = data["u_exact_all"]
X_mesh = data["X_mesh"]
T_mesh = data["T_mesh"]
u_pred = data["u_pred"]
final_error = data["final_error"]

x_vis = data["x_vis"].flatten()
t_vis = data["t_vis"].flatten()

# 尝试读取 centers 和 gammas (如果你已经修改了训练代码的话)
centers = data.get("centers", None)
gammas = data.get("gammas", None)

time_adam = data["time_adam"]
time_lbfgs = data["time_lbfgs"]
time_total = data["time_total"]

# ================= 3. 绘制高质量学术图像 (Figure 2) =================
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'font.family': 'serif'
})

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

t_min, t_max = t_exact.min(), t_exact.max()
x_min, x_max = x_exact.min(), x_exact.max()

# ---------- 子图 (a): 真实解 ----------
ax1 = fig.add_subplot(gs[0, 0])
cax1 = ax1.pcolormesh(T_mesh, X_mesh, u_exact_all, cmap='jet', shading='auto')
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title("(a) Exact Solution")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Space (x)")
ax1.set_xlim([t_min, t_max]); ax1.set_ylim([x_min, x_max])

# ---------- 子图 (b): 预测解 + 采样点 + 动态子域中心 ----------
ax2 = fig.add_subplot(gs[0, 1])
cax2 = ax2.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto', alpha=0.7)
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)

# 1. 画 RADS 对抗采样点
plot_points = min(1500, len(t_vis))
np.random.seed(42)
idx_plot = np.random.choice(len(t_vis), plot_points, replace=False)
ax2.scatter(t_vis[idx_plot], x_vis[idx_plot], color='black', s=2, alpha=0.5, label='RADS Points', zorder=2)

# 2. 画 Dynamic AB-PINN 子域中心与椭圆范围 (如果读取到了数据)
if centers is not None and gammas is not None:
    for i, center in enumerate(centers):
        x_c, t_c = center[0], center[1]
        gamma_val = gammas[i].flatten() # AC 中是 2 维的，比如 [gamma_x, gamma_t]
        
        # 计算椭圆在 t 和 x 方向的半轴长度 (1/sqrt(gamma))
        r_x = 1.0 / np.sqrt(gamma_val[0]) if gamma_val[0] > 0 else 0.2
        r_t = 1.0 / np.sqrt(gamma_val[1]) if gamma_val[1] > 0 else 0.2
        
        if i == 0:  
            color, marker, label_c, label_r = 'white', 's', 'Base Center', 'Base Region'
            s_size = 100
        else:      
            color, marker, label_c, label_r = 'red', '*', 'Dynamic Center', 'Dynamic Region'
            s_size = 200
            
        ax2.scatter(t_c, x_c, marker=marker, color=color, s=s_size, edgecolor='black', linewidth=1.2, zorder=5, label=label_c if i<=1 else "")
        
        # 使用 Ellipse 画出各向异性的影响范围
        ellipse = patches.Ellipse((t_c, x_c), width=2*r_t, height=2*r_x, fill=False, 
                                  edgecolor=color, linestyle='--', linewidth=2, alpha=0.9, zorder=4, label=label_r if i<=1 else "")
        ax2.add_patch(ellipse)

ax2.set_title(f"(b) RAD-AB-PINN, Sampling & Experts\n(Rel $L_2$ Error: {final_error:.2e})")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Space (x)")
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # 去重
ax2.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)
ax2.set_xlim([t_min, t_max]); ax2.set_ylim([x_min, x_max])

# ---------- 子图 (c): 绝对误差 ----------
ax3 = fig.add_subplot(gs[0, 2])
err_map = np.abs(u_exact_all - u_pred)
cax3 = ax3.pcolormesh(T_mesh, X_mesh, err_map, cmap='inferno', shading='auto')
fig.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title("(c) Absolute Error")
ax3.set_xlabel("Time (t)")
ax3.set_ylabel("Space (x)")
ax3.set_xlim([t_min, t_max]); ax3.set_ylim([x_min, x_max])

# ---------- 子图 (d)-(f): 时间切片对比 ----------
slice_times = [0.25, 0.50, 0.75]
for idx, t0 in enumerate(slice_times):
    ax_slice = fig.add_subplot(gs[1, idx])
    t_idx = (np.abs(t_exact - t0)).argmin()
    u_true_plot = u_exact_all[:, t_idx] 
    u_pred_plot = u_pred[:, t_idx]
    ax_slice.plot(x_exact, u_true_plot, 'b-', label="Exact", linewidth=2.5)
    ax_slice.plot(x_exact, u_pred_plot, 'r--', label="Pred", linewidth=2.5)
    char = chr(ord('d') + idx) 
    ax_slice.set_title(f"({char}) Cross-section at t = {t0}")
    ax_slice.set_xlabel("Space (x)")
    if idx == 0: ax_slice.set_ylabel("u(x,t)")
    ax_slice.grid(True, linestyle=':', alpha=0.7)
    ax_slice.legend(loc='upper right')
    ax_slice.set_ylim([-1.2, 1.2]) 

# ================= 4. 导出 PDF =================
PDF_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "Figure 2.pdf")
plt.savefig(PDF_SAVE_PATH, format="pdf", bbox_inches="tight", dpi=300)
print(f"✅ 可视化完成！带有子域中心的学术级组合图已保存为: {PDF_SAVE_PATH}")