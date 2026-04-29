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

# 实验预算对齐 (Total: 30000 Epochs, 20000 Points)
N_total_target = 20000
N_initial_r = 18000
RAR_steps = 5
N_add_per_step = 400
Epochs_per_step = 5000 # 6个阶段 * 5000 = 30000

nu = 0.01 / np.pi

# ==========================================================
# 2. AB-PINN 结构定义
# ==========================================================
class SubNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, xt): return self.net(xt)

class AB_PINN(nn.Module):
    def __init__(self, K=2):
        super().__init__()
        self.K = K
        self.subnets = nn.ModuleList([SubNet() for _ in range(K)])
        # 窗口函数参数：中心 c 和 宽度 log_gamma
        self.c = nn.Parameter(torch.tensor([-0.5, 0.5], dtype=torch.float32))
        self.log_gamma = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

    def get_weights(self, x):
        gamma = torch.exp(self.log_gamma)
        diff = x - self.c.view(1, -1)
        logits = - gamma.view(1, -1) * (diff**2)
        return torch.softmax(logits, dim=1)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        w = self.get_weights(x)
        u_out = 0
        for i in range(self.K):
            u_out += w[:, i:i+1] * self.subnets[i](xt)
        return u_out

def pde_residual(model, x, t):
    # 【核心修复】：detach 并重新开启梯度，解决非叶子节点报错
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

# 初始残差点
x_r = (-1 + 2 * torch.rand(N_initial_r, 1, dtype=torch.float32)).to(device)
t_r = torch.rand(N_initial_r, 1, dtype=torch.float32).to(device)

# ==========================================================
# 4. 训练循环 (带 RAR 逻辑)
# ==========================================================
model = AB_PINN(K=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"开始 AB-PINN + RAR 训练 ({device})...")
start_time = time.time()

for step in range(RAR_steps + 1):
    print(f"\n--- RAR Step {step}/{RAR_steps} | Points: {x_r.shape[0]} ---")
    model.train()
    for epoch in range(Epochs_per_step):
        optimizer.zero_grad()
        res = pde_residual(model, x_r, t_r)
        
        loss_r = torch.mean(res**2)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
        
        # 综合 Loss (增加 IC 权重通常有助于 Burgers 方程收敛)
        loss = loss_r + 2.0 * loss_ic
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Step {step} Epoch {epoch} | Loss: {loss.item():.3e}")

    # 执行 RAR 采样
    if step < RAR_steps:
        # 寻找残差最大的 400 个点
        x_cand = (-1 + 2 * torch.rand(100000, 1, dtype=torch.float32)).to(device)
        t_cand = torch.rand(100000, 1, dtype=torch.float32).to(device)
        
        # 计算候选点残差 (不使用 no_grad)
        res_cand = pde_residual(model, x_cand, t_cand)
        res_sq = (res_cand**2).detach()
        
        _, idx = torch.topk(res_sq.view(-1), N_add_per_step)
        
        # 合并新点并分离计算图
        x_r = torch.cat([x_r, x_cand[idx].detach()], dim=0)
        t_r = torch.cat([t_r, t_cand[idx].detach()], dim=0)
        
        del res_cand, res_sq, x_cand, t_cand

# ==========================================================
# 5. 最终误差评估 (Relative L2 Error)
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
print(f"\nAB-PINN + RAR Final Relative L2 Error: {error_l2:.3e}")
print(f"Total Time: {time.time()-start_time:.1f}s")

# 可视化最终窗口权重
plt.figure(figsize=(10, 4))
xx_eval = torch.linspace(-1, 1, 256, dtype=torch.float32).view(-1, 1).to(device)
with torch.no_grad():
    w_final = model.get_weights(xx_eval).cpu().numpy()
plt.plot(xx_eval.cpu().numpy(), w_final[:, 0], label="SubNet 1 Weight")
plt.plot(xx_eval.cpu().numpy(), w_final[:, 1], label="SubNet 2 Weight")
plt.title("Learned Window Functions with RAR")
plt.legend()
plt.show()