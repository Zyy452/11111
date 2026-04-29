import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 设置路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "ac_abpinn_hard_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行训练脚本。")

data = np.load(data_load_path)

u_pred = data['u_pred']
u_exact = data['u_exact']
X_mesh = data['X_mesh']
T_mesh = data['T_mesh']
x_exact = data['x_exact']
t_exact = data['t_exact']
phi_maps = data['phi_maps']
centers_t = data['centers_t']
centers_x = data['centers_x']
error_u = data['error_u']

print(f"✅ 成功读取 AC AB-PINN 数据！")
print(f"👉 相对 L2 误差: {error_u:.3e}")
print(f"👉 生成的子域专家数量: {len(phi_maps)} 个")

# ==========================================================
# 2. 绘图 (重构 1x4 图像分布)
# ==========================================================
fig, axs = plt.subplots(1, 4, figsize=(24, 5))

# --- 子图 1: 预测解 ---
im1 = axs[0].contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
axs[0].set_title("Predicted u(t,x)")
axs[0].set_xlabel("t"); axs[0].set_ylabel("x")
plt.colorbar(im1, ax=axs[0])

# --- 子图 2: 绝对误差 ---
if np.max(np.abs(u_exact)) > 0:
    im2 = axs[1].contourf(T_mesh, X_mesh, np.abs(u_exact - u_pred), 100, cmap='binary')
    axs[1].set_title(f"Absolute Error\n(Rel. L2 = {error_u:.2e})")
    axs[1].set_xlabel("t"); axs[1].set_ylabel("x")
    plt.colorbar(im2, ax=axs[1])

# --- 子图 3: 子域分布 ---
axs[2].set_title("Subdomain Distribution (Dynamic Experts)")
axs[2].set_xlabel("t"); axs[2].set_ylabel("x")
axs[2].set_xlim(0, 1); axs[2].set_ylim(-1, 1)

# 将绝对误差作为灰度背景，验证子域是否生成在误差大的位置
if np.max(np.abs(u_exact)) > 0:
    axs[2].contourf(T_mesh, X_mesh, np.abs(u_exact - u_pred), 20, cmap='Greys', alpha=0.4)

# 叠加绘制每个专家的辐射范围(0.5等高线)与中心点
for i in range(len(phi_maps)):
    phi_map = phi_maps[i]
    c_t = centers_t[i]
    c_x = centers_x[i]
    
    color = plt.cm.tab10(i % 10)
    
    # 绘制影响范围轮廓
    axs[2].contour(T_mesh, X_mesh, phi_map, levels=[0.5], colors=[color], linewidths=2)
    # 绘制中心点
    axs[2].scatter(c_t, c_x, marker='x', s=120, color=color, linewidth=3, label=f'Exp {i+1}')

axs[2].legend(loc='upper right', fontsize='small', framealpha=0.9)

# --- 子图 4: t=0.5 截面比较 ---
t_slice = 0.5
idx = int(t_slice * 200) 
axs[3].plot(x_exact, u_pred[idx, :], 'r--', linewidth=2.5, label='Prediction')
if np.max(np.abs(u_exact)) > 0:
    axs[3].plot(x_exact, u_exact[idx, :], 'k-', linewidth=1.5, alpha=0.7, label='Exact')
    
axs[3].set_title(f"Slice at t={t_slice}")
axs[3].legend()
axs[3].set_ylim([-1.1, 1.1])
axs[3].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
save_img_path = os.path.join(save_dir, "ac_abpinn_hard_result.pdf")
plt.savefig(save_img_path, dpi=150)
print(f"🎉 图像已成功保存至: {save_img_path}")