import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# ==========================================================
# 1. 设置路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "rad_dynamic_ab_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行 RAD+Dynamic AB-PINN 训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']       
u_exact = data['u_exact']       
x_exact = data['x'].flatten()           
t_exact = data['t'].flatten()           
centers = data['centers']     
gammas = data['gammas']       
x_r = data['x_r']             
t_r = data['t_r']             
train_time = data['train_time']
error_l2 = data['error_l2']

print(f"成功从 {save_dir} 读取数据！")
print(f"模型训练耗时: {train_time:.2f}s | 相对 L2 误差: {error_l2:.3e}")
print(f"动态生成的子域数量: {len(centers)} 个")

# ==========================================================
# 2. 绘制 Figure 1 (论文级别的综合拼图)
# ==========================================================
# 设置全局字体大小和样式（符合学术论文规范）
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif'
})

fig = plt.figure(figsize=(16, 10))
# 创建 2 行 6 列的网格。上半部分2个图各占3列，下半部分3个图各占2列
gs = gridspec.GridSpec(2, 6, height_ratios=[1.2, 1], hspace=0.35, wspace=0.8)

T_mesh, X_mesh = np.meshgrid(t_exact, x_exact)

# 获取物理域的确切边界，用于统一坐标轴
t_min, t_max = t_exact.min(), t_exact.max()
x_min, x_max = x_exact.min(), x_exact.max()

# ---------------------------------------------------------
# 子图 (a): RADS 残差对抗采样点分布热力图
# ---------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0:3])
cax1 = ax1.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto', alpha=0.6)
fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04, label='u(x,t)')

# 随机抽取部分采样点展示，防止画面全黑
plot_points = min(2000, len(t_r))
np.random.seed(42)
idx_plot = np.random.choice(len(t_r), plot_points, replace=False)
ax1.scatter(t_r[idx_plot], x_r[idx_plot], color='black', s=2, alpha=0.6, label=f'RAD Points ({plot_points} sampled)')

ax1.set_title("(a) RADS: Residual-based Adaptive Sampling")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Space (x)")
ax1.legend(loc="upper left", framealpha=0.9)

# 【核心修改】严格锁定坐标轴与物理域一致
ax1.set_xlim([t_min, t_max])
ax1.set_ylim([x_min, x_max])

# ---------------------------------------------------------
# 子图 (b): Dynamic AB-PINN 专家中心追踪与子域范围热力图
# ---------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 3:6])
cax2 = ax2.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto')
fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04, label='u(x,t)')

# 叠加专家子网络中心点及其控制范围 (Influence Region)
for i, center in enumerate(centers):
    x_c, t_c = center[0], center[1]
    
    # 获取该子域的 gamma 值，并计算对应的有效半径 ( 1/sqrt(gamma) )
    # 这代表了高斯权重衰减到一定程度的边界
    gamma_val = np.squeeze(gammas[i])
    radius = 1.0 / np.sqrt(gamma_val) if gamma_val > 0 else 0.2
    
    if i < 2:  
        # 基础子网络 (Base SubNets)
        label_center = 'Base Center' if i == 0 else ""
        label_region = 'Base Region' if i == 0 else ""
        color = 'white'
        ax2.scatter(t_c, x_c, marker='s', color=color, s=100, edgecolor='black', linewidth=1.5, zorder=5, label=label_center)
    else:      
        # 动态新增子网络 (Dynamic SubNets)
        label_center = 'Dynamic Center' if i == 2 else ""
        label_region = 'Dynamic Region' if i == 2 else ""
        color = 'red'
        ax2.scatter(t_c, x_c, marker='*', color=color, s=200, edgecolor='black', linewidth=1.2, zorder=5, label=label_center)
        
    # 【核心修改】绘制表示子域范围的虚线圆
    circle = patches.Circle((t_c, x_c), radius=radius, fill=False, 
                            edgecolor=color, linestyle='--', linewidth=2, alpha=0.9, zorder=4, label=label_region)
    ax2.add_patch(circle)

ax2.set_title("(b) Dynamic AB-PINN: Expert Domains & Centers")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Space (x)")

# 整理 legend，去除重复标签
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc="upper left", framealpha=0.9)

# 【核心修改】严格锁定坐标轴与图(a)完全一致
ax2.set_xlim([t_min, t_max])
ax2.set_ylim([x_min, x_max])

# ---------------------------------------------------------
# 子图 (c)-(e): 激波演化的一维时间切片对比
# ---------------------------------------------------------
slice_times = [0.25, 0.50, 0.75]
col_spans = [(1, 0, 2), (1, 2, 4), (1, 4, 6)]

for idx, (t0, col_span) in enumerate(zip(slice_times, col_spans)):
    ax_slice = fig.add_subplot(gs[col_span[0], col_span[1]:col_span[2]])
    
    t_idx = (np.abs(t_exact - t0)).argmin()
    u_true_plot = u_exact[:, t_idx] 
    u_pred_plot = u_pred[:, t_idx]
    
    ax_slice.plot(x_exact, u_true_plot, 'b-', label="Exact", linewidth=2.5)
    ax_slice.plot(x_exact, u_pred_plot, 'r--', label="Pred (RAD-AB-PINN)", linewidth=2.5)
    
    char = chr(ord('c') + idx) 
    ax_slice.set_title(f"({char}) Cross-section at t = {t0}")
    ax_slice.set_xlabel("Space (x)")
    if idx == 0:
        ax_slice.set_ylabel("u(x,t)")
    ax_slice.grid(True, linestyle=':', alpha=0.7)
    ax_slice.legend(loc='upper right')
    ax_slice.set_ylim([-1.2, 1.2])

# 保存为 PDF
pdf_save_path = os.path.join(save_dir, "Figure 1.pdf")
plt.savefig(pdf_save_path, format="pdf", bbox_inches="tight", dpi=300)
print(f"✅ 修改完成！高质量综合绘图已保存至: {pdf_save_path}")