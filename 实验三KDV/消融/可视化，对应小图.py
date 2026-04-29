import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置路径 =================
BASE_PATH = '/3241003007/zy/save/KDV'

def export_individual_results():
    # 1. 扫描所有子文件夹
    subdirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    
    if not subdirs:
        print(f"❌ 在 {BASE_PATH} 下没找到任何实验文件夹。")
        return

    print(f"🔍 找到 {len(subdirs)} 个实验目录，开始独立生成 PDF...")

    for subdir in subdirs:
        current_dir = os.path.join(BASE_PATH, subdir)
        
        # 2. 寻找该文件夹下的 .pt 数据文件
        pt_files = [f for f in f in os.listdir(current_dir) if f.endswith('.pt')]
        if not pt_files:
            print(f"⚠️ 跳过 {subdir}: 未找到 .pt 数据文件")
            continue
            
        # 3. 加载数据
        data_path = os.path.join(current_dir, pt_files[0])
        try:
            # 兼容处理，防止路径变动导致的加载问题
            data = torch.load(data_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"❌ 无法读取 {data_path}: {e}")
            continue

        u_pred = data["u_pred"]
        u_exact = data["u_exact"]
        x = data["x"]
        t = data["t"]
        config = data.get("config", subdir)
        l2_err = data.get("error", 0.0)

        # 4. 创建独立绘图 (三栏式标准科研布局)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        X, T = np.meshgrid(x, t)

        # --- 子图 1: 预测热力图 ---
        im1 = axes[0].contourf(T, X, u_pred, 100, cmap='jet')
        axes[0].set_title(f"Predict: {config}")
        axes[0].set_xlabel('t'); axes[0].set_ylabel('x')
        plt.colorbar(im1, ax=axes[0])

        # --- 子图 2: 绝对误差分布 ---
        err_map = np.abs(u_pred - u_exact)
        im2 = axes[1].contourf(T, X, err_map, 100, cmap='inferno')
        axes[1].set_title(f"Absolute Error (L2: {l2_err:.2e})")
        axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
        plt.colorbar(im2, ax=axes[1])

        # --- 子图 3: 最终时刻 (t_max) 切片对比 ---
        axes[2].plot(x, u_exact[-1, :], 'k-', linewidth=2, label='Exact')
        axes[2].plot(x, u_pred[-1, :], 'r--', linewidth=2, label='PINN')
        axes[2].set_title("Profile at t_max")
        axes[2].set_xlabel('x'); axes[2].set_ylabel('u(x,t)')
        axes[2].legend()

        plt.tight_layout()

        # 5. 保存 PDF 到对应文件夹
        pdf_filename = f"Result_Report_{subdir}.pdf"
        pdf_path = os.path.join(current_dir, pdf_filename)
        plt.savefig(pdf_path, format='pdf', dpi=300)
        plt.close(fig) # 及时关闭，防止内存溢出
        
        print(f"✅ 已生成: {pdf_path}")

if __name__ == "__main__":
    export_individual_results()
    print("\n🎉 所有实验的 PDF 报告已分别存入各自子文件夹。")