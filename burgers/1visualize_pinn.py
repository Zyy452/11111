import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 设置统一路径并读取数据
# ==========================================================
save_dir = "/3241003007/zy/save"
data_load_path = os.path.join(save_dir, "baseline_pinn_results.npz")

if not os.path.exists(data_load_path):
    raise FileNotFoundError(f"未找到数据文件 {data_load_path}，请先运行训练脚本。")

data = np.load(data_load_path)

u_pred_all = data['u_pred']   # shape [256, 100]
Exact = data['u_exact']       # shape [256, 100]
x_exact = data['x']           # shape [256]
t_exact = data['t']           # shape [100]
train_time = data['train_time']
error_l2 = data['error_l2']

print(f"成功从 {save_dir} 读取数据！")
print(f"该模型的训练耗时: {train_time:.2f}s | 相对 L2 误差: {error_l2:.3e}")

# ==========================================================
# 2. 绘制可视化图片
# ==========================================================
times = [0.0, 0.25, 0.50, 0.75, 1.0]
plt.figure(figsize=(15, 10))

for i, t0 in enumerate(times):
    # 根据目标时间 t0 找到对应的索引 (100个时间步对应 0.0 到 0.99)
    t_idx = int(t0 * 99)
    if t_idx >= len(t_exact):
        t_idx = len(t_exact) - 1 # 防止越界
        
    u_true_plot = Exact[:, t_idx]
    u_pred_plot = u_pred_all[:, t_idx]

    plt.subplot(2, 3, i+1)
    plt.plot(x_exact, u_true_plot, "k-", label="Exact", linewidth=2)
    plt.plot(x_exact, u_pred_plot, "r--", label="PINN", linewidth=2)
    plt.title(f"Time t = {t0}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()

# ==========================================================
# 3. 保存图片为 PDF 至指定目录
# ==========================================================
# 标准化命名：保存可视化结果
pdf_save_path = os.path.join(save_dir, "baseline_pinn_visualization.pdf")
plt.savefig(pdf_save_path, format="pdf", bbox_inches="tight")
print(f"可视化图片已保存为 PDF: {pdf_save_path}")

# 如果需要在屏幕上预览，取消下面这行的注释
plt.show()