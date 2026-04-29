import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 路径设置与读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "kdv_radpinn_results.npz")

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

print(f"✅ 成功读取 KdV RAD-PINN 结果数据！")
print(f"👉 相对 L2 误差: {final_error:.4e}")

# ==========================================================
# 2. 绘图 (1x3 图像分布)
# ==========================================================
fig = plt.figure(figsize=(18, 5))

# --- 子图 1: 预测解 ---
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"RAD-PINN Predict (Error: {final_error:.4f})")
ax1.set_xlabel('t')
ax1.set_ylabel('x')
plt.colorbar(im1, ax=ax1)

# --- 子图 2: 绝对误差 ---
ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title("Absolute Error (RAD-Only)")
ax2.set_xlabel('t')
ax2.set_ylabel('x')
plt.colorbar(im2, ax=ax2)

# --- 子图 3: t 截面切片 ---
ax3 = plt.subplot(1, 3, 3)
idx_t1, idx_t2 = int(0.5 * len(t_exact)), int(0.8 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t1, :], 'k-', linewidth=2, label=f"Exact t={t_exact[idx_t1]:.2f}")
ax3.plot(x_exact, u_pred[idx_t1, :], 'r--', linewidth=2, label="Predict")
ax3.plot(x_exact, u_exact[idx_t2, :], 'b-', linewidth=2, label=f"Exact t={t_exact[idx_t2]:.2f}")
ax3.plot(x_exact, u_pred[idx_t2, :], 'g--', linewidth=2, label="Predict")
ax3.set_title("Wave Profile Slices")
ax3.set_xlabel('x')
ax3.set_ylabel('u(x,t)')
ax3.legend()

plt.tight_layout()

# 保存图像
save_img_path = os.path.join(save_dir, "Aligned_Baseline3_RAD_Only.pdf")
plt.savefig(save_img_path, dpi=200)
print(f"🎉 Baseline 3 图片已成功保存至: {save_img_path}")
plt.show()
