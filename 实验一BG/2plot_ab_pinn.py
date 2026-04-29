import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 设置路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "ab_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行 AB-PINN 训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']       
Exact = data['u_exact']       
x_exact = data['x']           
t_exact = data['t']           
w_final = data['w_final']     
c_final = data['c']           # 读取子域中心
gamma_final = data['gamma']   # 读取子域宽度
train_time = data['train_time']
error_l2 = data['error_l2']

print(f"成功从 {save_dir} 读取数据！")
print(f"模型训练耗时: {train_time:.2f}s | 相对 L2 误差: {error_l2:.3e}")
print(f"读取到的子域中心 c: {c_final}")

# ==========================================================
# 2. 绘制图表 1：时间切片与权重函数
# ==========================================================
times = [0.0, 0.25, 0.50, 0.75, 1.0]
fig1 = plt.figure(figsize=(15, 8))

for i, t0 in enumerate(times):
    idx = (np.abs(t_exact - t0)).argmin()
    u_true_plot = Exact[:, idx] 
    u_pred_plot = u_pred[:, idx]
    
    plt.subplot(2, 3, i+1)
    plt.plot(x_exact, u_true_plot, 'k-', label="Exact", linewidth=2)
    plt.plot(x_exact, u_pred_plot, 'r--', label="AB-PINN", linewidth=2)
    plt.title(f"Time t = {t0}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.subplot(2, 3, 6)
plt.plot(x_exact, w_final[:, 0], label=f"SubNet 1 (c={c_final[0]:.2f})", color='blue', linewidth=2)
plt.plot(x_exact, w_final[:, 1], label=f"SubNet 2 (c={c_final[1]:.2f})", color='orange', linewidth=2)
# 用虚线在图中标出中心位置
plt.axvline(x=c_final[0], color='blue', linestyle=':', alpha=0.5)
plt.axvline(x=c_final[1], color='orange', linestyle=':', alpha=0.5)
plt.title("Learned Window Functions")
plt.xlabel("x")
plt.ylabel("Weight w(x)")
plt.grid(True, alpha=0.3)
plt.legend()

fig1.tight_layout()
pdf_save_path_1 = os.path.join(save_dir, "ab_pinn_slices.pdf")
fig1.savefig(pdf_save_path_1, format="pdf", bbox_inches="tight")
print(f"切片可视化已保存至: {pdf_save_path_1}")

# ==========================================================
# 3. 绘制图表 2：全时空预测热力图 (带中心线标注)
# ==========================================================
T_mesh, X_mesh = np.meshgrid(t_exact, x_exact)

fig2 = plt.figure(figsize=(10, 6))
# 绘制预测结果的热力图
cax = plt.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='auto')
plt.colorbar(cax, label='u(x,t)')

# 在热力图上绘制 AB-PINN 学习到的空间子域中心
plt.axhline(y=c_final[0], color='white', linestyle='--', linewidth=2, label=f'SubNet 1 Center (x={c_final[0]:.2f})')
plt.axhline(y=c_final[1], color='white', linestyle='-.', linewidth=2, label=f'SubNet 2 Center (x={c_final[1]:.2f})')

plt.title("AB-PINN Spatiotemporal Prediction & Learned Subdomains")
plt.xlabel("Time (t)")
plt.ylabel("Space (x)")
plt.legend(loc="upper right")

fig2.tight_layout()
pdf_save_path_2 = os.path.join(save_dir, "ab_pinn_heatmap.pdf")
fig2.savefig(pdf_save_path_2, format="pdf", bbox_inches="tight")
print(f"热力图可视化已保存至: {pdf_save_path_2}")