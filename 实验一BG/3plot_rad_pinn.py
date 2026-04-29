import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 设置路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "rad_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行 Pure RAD-PINN 训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']       
u_exact = data['u_exact']       
x_exact = data['x']           
t_exact = data['t']           
x_r = data['x_r']             
t_r = data['t_r']             
train_time = data['train_time']
error_l2 = data['error_l2']

print(f"成功从 {save_dir} 读取数据！")
print(f"模型训练耗时: {train_time:.2f}s | 相对 L2 误差: {error_l2:.3e}")
print(f"用于可视化的 RAD 配点数量: {len(x_r)} 个")

# ==========================================================
# 2. 绘制图表 1：时间切片对比图
# ==========================================================
times = [0.0, 0.25, 0.50, 0.75, 1.0]
fig1 = plt.figure(figsize=(15, 6))

for i, t0 in enumerate(times):
    idx = (np.abs(t_exact - t0)).argmin()
    u_true_plot = u_exact[:, idx] 
    u_pred_plot = u_pred[:, idx]
    
    plt.subplot(1, 5, i+1)
    plt.plot(x_exact, u_true_plot, 'k-', label="Exact", linewidth=2)
    plt.plot(x_exact, u_pred_plot, 'r--', label="Pure RAD-PINN", linewidth=2)
    plt.title(f"t = {t0}")
    plt.xlabel("x")
    if i == 0:
        plt.ylabel("u(x,t)")
    plt.grid(True, alpha=0.3)
    plt.legend()

fig1.tight_layout()
pdf_save_path_1 = os.path.join(save_dir, "rad_pinn_slices.pdf")
fig1.savefig(pdf_save_path_1, format="pdf", bbox_inches="tight")
print(f"切片可视化已保存至: {pdf_save_path_1}")

# ==========================================================
# 3. 绘制图表 2：预测热力图 vs RAD 采样点散点图
# ==========================================================
T_mesh, X_mesh = np.meshgrid(t_exact, x_exact)

fig2 = plt.figure(figsize=(16, 6))

# ---- 子图 1: 全时空预测热力图 ----
ax1 = plt.subplot(1, 2, 1)
cax1 = ax1.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto')
fig2.colorbar(cax1, ax=ax1, label='u(x,t)')

ax1.set_title("Pure RAD-PINN: Spatiotemporal Prediction")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Space (x)")

# ---- 子图 2: 预测热力图 (低透明度) + RAD 配点分布 ----
ax2 = plt.subplot(1, 2, 2)
cax2 = ax2.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto', alpha=0.4)
fig2.colorbar(cax2, ax=ax2, label='u(x,t)')

# 绘制最终的 L-BFGS 阶段选出的配点
ax2.scatter(t_r, x_r, color='black', s=1, alpha=0.5, label=f"Collocation Points (N={len(x_r)})")

ax2.set_title("RAD Concept: Points cluster at High-Error Regions\n(Notice the density around the shock wave)")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Space (x)")
ax2.legend(loc="upper right")

fig2.tight_layout()
pdf_save_path_2 = os.path.join(save_dir, "rad_pinn_heatmap.pdf")
fig2.savefig(pdf_save_path_2, format="pdf", bbox_inches="tight")
print(f"RAD 采样可视化热力图已保存至: {pdf_save_path_2}")