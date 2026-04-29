import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ================= 1. 路径配置 =================
# 🌟 保持与训练脚本中一致，自动寻找对应点数的文件！
N_TRAIN_POINTS = 15000  

SAVE_DIR = '/3241003007/zy/实验三KDV/KDV_Save'
# 🌟 动态拼接读取路径
DATA_LOAD_PATH = os.path.join(SAVE_DIR, f"KDV_fourier_pts{N_TRAIN_POINTS}_results.pt") 

if not os.path.exists(DATA_LOAD_PATH):
    raise FileNotFoundError(f"❌ 找不到数据文件：{DATA_LOAD_PATH}，请先运行 KDV_train.py 训练 {N_TRAIN_POINTS} 点网络！")

# ================= 2. 读取数据 =================
print(f"📂 正在加载 KDV 傅里叶 {N_TRAIN_POINTS} 点测试数据: {DATA_LOAD_PATH} ...")
data = torch.load(DATA_LOAD_PATH, weights_only=False)

x_exact = data["x_exact"]
t_exact = data["t_exact"]
u_exact = data["u_exact"]
X_mesh = data["X_mesh"]
T_mesh = data["T_mesh"]
u_pred = data["u_pred"]

final_error = data["final_error"]
max_error = data.get("max_error", 0.0) 

time_adam = data["time_adam"]
time_lbfgs = data["time_lbfgs"]
time_total = data["time_total"]

# ================= 3. 成本与误差统计分析 =================
print(f"\n📊 =============== KDV 测试分析统计 ({N_TRAIN_POINTS}点) ===============")
print(f"⏱️ Phase 1 (Adam) 耗时:   {str(datetime.timedelta(seconds=int(time_adam)))}")
print(f"⏱️ Phase 2 (L-BFGS)耗时:  {str(datetime.timedelta(seconds=int(time_lbfgs)))}")
print(f"⏱️ 模型总训练总耗时:      {str(datetime.timedelta(seconds=int(time_total)))}")
print(f"🎯 最终相对 L2 误差(平均): {final_error:.6e}")
print(f"🔥 最终最大绝对误差(最差): {max_error:.6e}")
print("=================================================\n")

# ================= 4. 绘制图像 =================
print("🖼️ 正在生成可视化图像...")
fig = plt.figure(figsize=(18, 5))

# 图1：模型预测热力图
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"KDV Predict (Pts:{N_TRAIN_POINTS} | L2: {final_error:.4f})")
ax1.set_xlabel('t'); ax1.set_ylabel('x')
plt.colorbar(im1, ax=ax1)

# 图2：绝对误差分布
ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title(f"KDV Abs Error ({N_TRAIN_POINTS} pts)")
ax2.set_xlabel('t'); ax2.set_ylabel('x')
plt.colorbar(im2, ax=ax2)

# 图3：特定时间步的切片对比
ax3 = plt.subplot(1, 3, 3)
idx_t1, idx_t2 = int(0.5 * len(t_exact)), int(0.8 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t1, :], 'k-', linewidth=2, label=f"Exact t={t_exact[idx_t1]:.2f}")
ax3.plot(x_exact, u_pred[idx_t1, :], 'r--', linewidth=2, label="Predict")
ax3.plot(x_exact, u_exact[idx_t2, :], 'b-', linewidth=2, label=f"Exact t={t_exact[idx_t2]:.2f}")
ax3.plot(x_exact, u_pred[idx_t2, :], 'g--', linewidth=2, label="Predict")
ax3.set_title("KDV Wave Profile Slices")
ax3.set_xlabel('x'); ax3.set_ylabel('u(x,t)')
ax3.legend()

plt.tight_layout()

# ================= 5. 导出为高分辨率 PDF =================
# 🌟 动态拼接 PDF 保存名
PDF_SAVE_PATH = os.path.join(SAVE_DIR, f"KDV_RAD_AB_Fourier_pts{N_TRAIN_POINTS}_Result.pdf")
plt.savefig(PDF_SAVE_PATH, format="pdf", dpi=300)

print(f"✅ 可视化完成！高清图表已保存为 PDF 格式: {PDF_SAVE_PATH}")
plt.show()