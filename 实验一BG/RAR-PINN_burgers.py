import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os

# ==========================================================
# 0. 设备与随机种子配置
# ==========================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if device.type == "mps": torch.mps.manual_seed(42)

# ==========================================================
# 1. 变量对齐与路径配置
# ==========================================================
data_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat"

N_total_target = 20000
N_initial_r = 18000
RAR_steps = 5
N_add_per_step = 400
Epochs_per_step = 5000 

nu = 0.01 / np.pi

# ==========================================================
# 2. 网络结构与 PDE 残差函数
# ==========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def pde_residual(model, x, t):
    # 核心修复：确保输入是可求导的叶子节点镜像
    x = x.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)
    
    u = model(x, t)
    u_grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_t = u_grads[0], u_grads[1]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# ==========================================================
# 3. 初始数据准备
# ==========================================================
x_ic = torch.linspace(-1, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic = -torch.sin(np.pi * x_ic).to(device)

t_bc = torch.rand(200, 1, dtype=torch.float32).to(device)
x_bc = torch.cat([-torch.ones(100, 1), torch.ones(100, 1)]).to(device)
t_bc = torch.cat([t_bc[:100], t_bc[100:]]).to(device)
u_bc = torch.zeros_like(x_bc).to(device)

x_r = (-1 + 2 * torch.rand(N_initial_r, 1, dtype=torch.float32)).to(device)
t_r = torch.rand(N_initial_r, 1, dtype=torch.float32).to(device)

# ==========================================================
# 4. 训练循环 (RAR 逻辑)
# ==========================================================
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"开始 Baseline+RAR 训练 ({device})...")
start_time = time.time()

for step in range(RAR_steps + 1):
    print(f"\n--- RAR Step {step}/{RAR_steps} | Points: {x_r.shape[0]} ---")
    model.train()
    for epoch in range(Epochs_per_step):
        optimizer.zero_grad()
        res = pde_residual(model, x_r, t_r)
        loss_r = torch.mean(res**2)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
        loss_bc = torch.mean((model(x_bc, t_bc) - u_bc)**2)
        
        loss = loss_r + 2.0 * (loss_ic + loss_bc)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.3e}")

    # RAR 增加采样点
    if step < RAR_steps:
        # 在 100,000 个随机候选点中寻找最大残差
        x_cand = (-1 + 2 * torch.rand(100000, 1, dtype=torch.float32)).to(device)
        t_cand = torch.rand(100000, 1, dtype=torch.float32).to(device)
        
        # 允许计算图以计算残差
        res_cand = pde_residual(model, x_cand, t_cand)
        res_sq = (res_cand**2).detach() # 立即 detach 释放内存
        
        _, idx = torch.topk(res_sq.view(-1), N_add_per_step)
        x_r = torch.cat([x_r, x_cand[idx].detach()], dim=0)
        t_r = torch.cat([t_r, t_cand[idx].detach()], dim=0)
        
        del res_cand, res_sq, x_cand, t_cand

# ==========================================================
# 5. 最终误差评估
# ==========================================================
if not os.path.exists(data_path):
    raise FileNotFoundError(f"未找到数据文件: {data_path}")

data = loadmat(data_path)
Exact = data["usol"]
x_star, t_star = data["x"].flatten(), data["t"].flatten()
X, T = np.meshgrid(x_star, t_star)
x_test = torch.tensor(X.flatten()[:,None], dtype=torch.float32).to(device)
t_test = torch.tensor(T.flatten()[:,None], dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    u_pred = model(x_test, t_test).cpu().numpy().reshape(len(t_star), len(x_star)).T

error_l2 = np.linalg.norm(Exact - u_pred, 2) / np.linalg.norm(Exact, 2)
print(f"\nBaseline+RAR Final Relative L2 Error: {error_l2:.3e} | Time: {time.time()-start_time:.1f}s")

# 简单可视化对比 t=1.0
plt.figure(figsize=(8, 4))
plt.plot(x_star, Exact[:, -1], 'k-', label="True")
plt.plot(x_star, u_pred[:, -1], 'r--', label="PINN+RAR")
plt.title("Solution at t=1.0")
plt.legend()
plt.show()