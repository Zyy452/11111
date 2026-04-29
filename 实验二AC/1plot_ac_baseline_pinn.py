import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 设置路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "ac_baseline_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行 AC Baseline 训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']         # (201, 512)
u_exact = data['u_exact']       # (201, 512)
x_exact = data['x']             # (512,)
t_exact = data['t']             # (201,)
loss_history = data['loss_history']
train_time = data['train_time']
error_l2 = data['error_l2']

print(f"成功从 {save_dir} 读取 AC 数据！")
print(f"模型训练耗时: {train_time:.2f}s | 相对 L2 误差: {error_l2:.3e}")

# ==========================================================
# 2. 绘制图表 1：Loss 曲线与时间切片
# ==========================================================
fig1 = plt.figure(figsize=(18, 5))

# 绘制 Loss 曲线
plt.subplot(1, 4, 1)
plt.semilogy(loss_history, 'k-', linewidth=1.5)
plt.title("Training Loss History")
plt.xlabel("Optimization Steps")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# 绘制时间切片
times = [0.1, 0.5, 0.9]
for i, t0 in enumerate(times):
    # 找到最接近的目标时间索引
    idx = (np.abs(t_exact - t0)).argmin()
    u_true_plot = u_exact[idx, :] 
    u_pred_plot = u_pred[idx, :]
    
    plt.subplot(1, 4, i+2)
    plt.plot(x_exact, u_true_plot, 'k-', label="Exact", linewidth=2)
    plt.plot(x_exact, u_pred_plot, 'r--', label="Baseline PINN", linewidth=2)
    plt.title(f"Allen-Cahn: t = {t_exact[idx]:.2f}")
    plt.xlabel("x")
    if i == 0:
        plt.ylabel("u(x,t)")
    plt.grid(True, alpha=0.3)
    plt.legend()

fig1.tight_layout()
pdf_save_path_1 = os.path.join(save_dir, "ac_baseline_pinn_slices.pdf")
fig1.savefig(pdf_save_path_1, format="pdf", bbox_inches="tight")
print(f"切片及 Loss 曲线可视化已保存至: {pdf_save_path_1}")

# ==========================================================
# 3. 绘制图表 2：全时空预测热力图对比
# ==========================================================
# Meshgrid 格式用于 pcolormesh
T_mesh, X_mesh = np.meshgrid(t_exact, x_exact, indexing='ij')

fig2 = plt.figure(figsize=(18, 5))

# 真实值
ax1 = plt.subplot(1, 3, 1)
cax1 = ax1.pcolormesh(T_mesh, X_mesh, u_exact, cmap='jet', shading='auto')
fig2.colorbar(cax1, ax=ax1, label='u(x,t)')
ax1.set_title("Allen-Cahn: Exact Solution")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Space (x)")

# 预测值
ax2 = plt.subplot(1, 3, 2)
cax2 = ax2.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto')
fig2.colorbar(cax2, ax=ax2, label='u(x,t)')
ax2.set_title(f"Baseline PINN Prediction\n(Rel. L2 Error: {error_l2:.3e})")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Space (x)")

# 绝对误差
error_map = np.abs(u_exact - u_pred)
ax3 = plt.subplot(1, 3, 3)
cax3 = ax3.pcolormesh(T_mesh, X_mesh, error_map, cmap='hot', shading='auto')
fig2.colorbar(cax3, ax=ax3, label='|Error|')
ax3.set_title("Absolute Error")
ax3.set_xlabel("Time (t)")
ax3.set_ylabel("Space (x)")

fig2.tight_layout()
pdf_save_path_2 = os.path.join(save_dir, "ac_baseline_pinn_heatmap.pdf")
fig2.savefig(pdf_save_path_2, format="pdf", bbox_inches="tight")
print(f"全时空热力图对比已保存至: {pdf_save_path_2}")