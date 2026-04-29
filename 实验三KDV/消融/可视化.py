import torch
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_PATH = '/3241003007/zy/save/KDV'
# 获取所有含有 .pt 文件的子文件夹
subdirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
subdirs.sort()

fig, axes = plt.subplots(len(subdirs), 2, figsize=(12, 3 * len(subdirs)))
if len(subdirs) == 1: axes = np.expand_dims(axes, axis=0)

for i, subdir in enumerate(subdirs):
    # 查找该文件夹下的 .pt 文件
    pt_file = [f for f in os.listdir(os.path.join(BASE_PATH, subdir)) if f.endswith('.pt')]
    if not pt_file: continue
    
    data = torch.load(os.path.join(BASE_PATH, subdir, pt_file[0]))
    u_pred = data["u_pred"]
    u_exact = data["u_exact"]
    
    # 预测图
    im1 = axes[i, 0].imshow(u_pred.T, aspect='auto', cmap='jet')
    axes[i, 0].set_title(f"Config: {subdir}")
    plt.colorbar(im1, ax=axes[i, 0])
    
    # 误差图
    im2 = axes[i, 1].imshow(np.abs(u_pred - u_exact).T, aspect='auto', cmap='inferno')
    axes[i, 1].set_title(f"L2 Error: {data['error']:.5f}")
    plt.colorbar(im2, ax=axes[i, 1])

plt.tight_layout()
summary_path = os.path.join(BASE_PATH, "Ablation_Comparison_All.png")
plt.savefig(summary_path, dpi=300)
print(f"🎉 汇总对比图已生成: {summary_path}")
plt.show()