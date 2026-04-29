import os
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import time

# ==========================================================
# 0. 目录与环境初始化
# ==========================================================
# 统一的输出保存文件夹
save_dir = "/3241003007/zy/save"
os.makedirs(save_dir, exist_ok=True)

# 检查并设置设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用设备: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用设备: CUDA")
else:
    device = torch.device("cpu")
    print("使用设备: CPU")

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)
if device.type == "mps":
    torch.mps.manual_seed(42)

# ==========================================================
# 1. 读取 Burgers 方程真解
# ==========================================================
# 原始 mat 数据集路径保持不变
data_path = "/3241003007/zy/实验一BG/burgers_shock.mat"
data = loadmat(data_path)
x_exact = data["x"].flatten()             # shape [256]
t_exact = data["t"].flatten()             # shape [100]
Exact = data["usol"]                      # shape [256, 100]

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)

# ==========================================================
# 2. PINN 网络结构
# ==========================================================
class MLP(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 1]):
        super().__init__()
        net = []
        for i in range(len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

model = MLP().to(device)
nu = 0.01 / np.pi

# ==========================================================
# 3. 准备训练数据
# ==========================================================
N_r = 20000
x_r = (-1 + 2 * torch.rand(N_r, 1, dtype=torch.float32)).to(device)
t_r = torch.rand(N_r, 1, dtype=torch.float32).to(device)

N_ic = 200
x_ic = torch.linspace(-1, 1, N_ic, dtype=torch.float32).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic = -torch.sin(np.pi * x_ic).to(device)

N_bc = 200
t_bc = torch.rand(N_bc, 1, dtype=torch.float32).to(device)
x_bc_L = -1.0 * torch.ones_like(t_bc).to(device)
x_bc_R =  1.0 * torch.ones_like(t_bc).to(device)
u_L_val = torch.zeros_like(t_bc).to(device)
u_R_val = torch.zeros_like(t_bc).to(device)

# ==========================================================
# 4. PDE 残差计算函数
# ==========================================================
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)

    u_grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)
    u_x, u_t = u_grads[0], u_grads[1]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# ==========================================================
# 5. 训练循环
# ==========================================================
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start_time = time.time()
print("开始训练...")

for epoch in range(30001):
    optimizer.zero_grad()

    loss_r = loss_fn(pde_residual(model, x_r, t_r), torch.zeros_like(x_r))
    loss_ic = loss_fn(model(x_ic, t_ic), u_ic)
    loss_bc = loss_fn(model(x_bc_L, t_bc), u_L_val) + loss_fn(model(x_bc_R, t_bc), u_R_val)

    loss = loss_r + loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | PDE: {loss_r.item():.3e} | 累计用时: {elapsed:.1f}s")

# 统计总训练时间
total_train_time = time.time() - start_time

# ==========================================================
# 6. 计算误差并保存数据至指定目录
# ==========================================================
model.eval()

# 构建推理网格以计算全域预测值
x_test = torch.tensor(X_mesh.flatten()[:, None], dtype=torch.float32).to(device)
t_test = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred_all = model(x_test, t_test).cpu().numpy().reshape(len(t_exact), len(x_exact)).T

error_l2 = np.linalg.norm(Exact - u_pred_all, 2) / np.linalg.norm(Exact, 2)

print(f"\n--- 训练完成 ---")
print(f"总训练耗时: {total_train_time:.2f} 秒")
print(f"最终相对 L2 误差: {error_l2:.3e}")

# 标准化命名：保存模型权重
model_save_path = os.path.join(save_dir, "baseline_pinn_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"模型权重已保存至: {model_save_path}")

# 标准化命名：保存预测数据、真实数据及训练时间
data_save_path = os.path.join(save_dir, "baseline_pinn_results.npz")
np.savez(data_save_path, 
         u_pred=u_pred_all, 
         u_exact=Exact, 
         x=x_exact, 
         t=t_exact,
         train_time=total_train_time,
         error_l2=error_l2)
print(f"预测结果及统计数据已保存至: {data_save_path}")