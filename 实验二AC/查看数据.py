import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 设定您的文件路径
file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/KdV.mat"

print(f"📂 正在检查文件: {file_path}")

# 2. 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 错误: 文件不存在！请检查路径或文件名是否正确。")
    # 尝试列出目录下的文件，帮您确认文件名
    dir_name = os.path.dirname(file_path)
    if os.path.exists(dir_name):
        print(f"\n📂 目录 {dir_name} 下的文件有:")
        for f in os.listdir(dir_name):
            if f.endswith(".mat"):
                print(f" - {f}")
else:
    print(f"✅ 文件存在，尝试读取...")
    
    try:
        # 3. 加载数据
        data = scipy.io.loadmat(file_path)
        
        print("\n🔍 --- 文件结构分析 ---")
        print(f"{'Key Name':<15} | {'Type':<15} | {'Shape'}")
        print("-" * 45)
        
        # 4. 打印所有变量名和形状
        useful_keys = []
        for key, value in data.items():
            # 跳过 .mat 文件的元数据
            if key.startswith('__'):
                continue
            
            useful_keys.append(key)
            
            # 获取类型和形状
            val_type = type(value).__name__
            try:
                val_shape = str(value.shape)
            except:
                val_shape = "N/A"
                
            print(f"{key:<15} | {val_type:<15} | {val_shape}")

        print("-" * 45)

        # 5. 尝试绘图 (如果看起来像是 AC 方程的数据)
        # 通常包含 x, t, u (或 usol)
        
        # 自动猜测变量名
        u_key = next((k for k in useful_keys if 'u' in k.lower()), None)
        x_key = next((k for k in useful_keys if 'x' in k.lower()), None)
        t_key = next((k for k in useful_keys if 't' in k.lower()), None)
        
        if u_key and x_key and t_key:
            print(f"\n🎨 尝试绘图: 使用 u='{u_key}', x='{x_key}', t='{t_key}'")
            
            u_data = data[u_key]
            # 确保是 numpy array
            if isinstance(u_data, np.ndarray):
                plt.figure(figsize=(10, 6))
                
                # 如果是标准的 (201, 100) 这种形状，直接画热力图
                if u_data.ndim == 2:
                    plt.imshow(u_data, aspect='auto', cmap='jet', origin='lower')
                    plt.colorbar(label=u_key)
                    plt.title(f"Data Preview: {u_key} (Shape: {u_data.shape})")
                    plt.xlabel("t dimension index")
                    plt.ylabel("x dimension index")
                    plt.savefig("AC_data_preview.png")
                    print("✅ 预览图已保存为 'AC_data_preview.png'")
                    plt.show()
                else:
                    print(f"⚠️ {u_key} 的维度是 {u_data.ndim}，不方便直接画 2D 热力图。")
        else:
            print("\n⚠️ 无法自动识别 x, t, u 变量，请根据上面的列表手动修改代码中的变量名。")

    except Exception as e:
        print(f"\n❌ 读取出错: {e}")