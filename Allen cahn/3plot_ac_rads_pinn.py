import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 设置路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "ac_rads_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']
u_exact = data['u_exact']
X_mesh = data['X_mesh']
T_mesh = data['T_mesh']
x_exact = data['x_exact']
t_exact = data['t_exact']
x_vis = data['x_vis']
t_vis = data['t_vis']
error_u = data['error_u']

print(f"✅ 成功读取 AC RADS PINN 数据！")
print(f"👉 相对 L2 误差: {error_u:.3e}")

# ==========================================================
# 2. 绘图 (1x4 图像分布)
# ==========================================================
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# --- 子图 1: 预测解 ---
im1 = axs[0].contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
axs[0].set_title(f"Predicted u(t,x)\nRel. L2 Error: {error_u:.2e}")
axs[0].set_xlabel("t"); axs[0].set_ylabel("x")
plt.colorbar(im1, ax=axs[0])

# --- 子图 2: 绝对误差 (Log10 Scale) ---
if np.max(np.abs(u_exact)) > 0:
    err_map = np.abs(u_exact - u_pred) + 1e-12
    im2 = axs[1].contourf(T_mesh, X_mesh, np.log10(err_map), 100, cmap='inferno')
    axs[1].set_title("Log10 Absolute Error")
    axs[1].set_xlabel("t"); axs[1].set_ylabel("x")
    plt.colorbar(im2, ax=axs[1])

# --- 子图 3: AAIS / RADS 采样点分布 ---
axs[2].set_title("RADS Adversarial Sampling (Generator)")
axs[2].set_xlabel("t"); axs[2].set_ylabel("x")
axs[2].set_xlim(0, 1); axs[2].set_ylim(-1, 1)
# 绘制散点图展示生成器倾向的采样区域
axs[2].scatter(t_vis, x_vis, s=2, c='blue', alpha=0.5, label='AAIS Points')
axs[2].legend(loc='upper right')
axs[2].grid(True, alpha=0.3)

# --- 子图 4: t=0.5 截面比较 ---
t_slice = 0.5
idx = int(t_slice * (len(t_exact) - 1))  # 根据网格点数动态推断索引
axs[3].plot(x_exact, u_pred[idx, :], 'r--', linewidth=2, label='Prediction')
if np.max(np.abs(u_exact)) > 0:
    axs[3].plot(x_exact, u_exact[idx, :], 'k-', linewidth=1.0, alpha=0.7, label='Exact')
    
axs[3].set_title(f"Slice at t={t_slice}")
axs[3].set_xlabel("x")
axs[3].set_ylabel("u")
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
save_img_path = os.path.join(save_dir, "ac_rads_pinn_result.png")
plt.savefig(save_img_path, dpi=150)
print(f"🎉 图像已成功保存至: {save_img_path}")