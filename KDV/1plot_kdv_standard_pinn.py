import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 路径设置与读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "kdv_standard_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']
u_exact = data['u_exact']
X_mesh = data['X_mesh']
T_mesh = data['T_mesh']
x_exact = data['x_exact']
t_exact = data['t_exact']
final_error = float(data['final_error'])

print(f"✅ 成功读取 KdV Standard PINN 结果数据！")
print(f"👉 相对 L2 误差: {final_error:.4e}")

# ==========================================================
# 2. 绘图 (1x3 图像分布)
# ==========================================================
fig = plt.figure(figsize=(18, 5))

# --- 子图 1: 预测解 ---
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"Standard PINN Predict (Error: {final_error:.4f})")
ax1.set_xlabel("t")
ax1.set_ylabel("x")
plt.colorbar(im1, ax=ax1)

# --- 子图 2: 绝对误差 ---
ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title("Absolute Error")
ax2.set_xlabel("t")
ax2.set_ylabel("x")
plt.colorbar(im2, ax=ax2)

# --- 子图 3: t 截面切片 ---
ax3 = plt.subplot(1, 3, 3)
idx_t = int(0.7 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t, :], 'k-', linewidth=2, label="Exact")
ax3.plot(x_exact, u_pred[idx_t, :], 'r--', linewidth=2, label="Predict")
ax3.set_title(f"Slice at t={t_exact[idx_t]:.2f}")
ax3.set_xlabel("x")
ax3.set_ylabel("u")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
save_img_path = os.path.join(save_dir, "KDV_StandardPINN.pdf")
plt.savefig(save_img_path, dpi=200)
print(f"🎉 图像已成功保存至: {save_img_path}")
plt.show()