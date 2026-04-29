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
# 1. 变量对齐与路径配置 (保持与基准线完全一致)
# ==========================================================
data_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat"

N_total_target = 20000
N_initial_r = 18000
RAR_steps = 5
N_add_per_step = 400
Total_Epochs = 30000
Epochs_per_step = Total_Epochs // (RAR_steps + 1)

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

class AB_Causality_PINN(nn.Module):
    def __init__(self, K=2):
        super().__init__()
        self.K = K
        self.subnets = nn.ModuleList([SubNet() for _ in range(K)])
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
    x = x.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)
    u = model(x, t)
    u_grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_t = u_grads[0], u_grads[1]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# ==========================================================
# 3. 数据准备
# ==========================================================
x_ic = torch.linspace(-1, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
u_ic = -torch.sin(np.pi * x_ic).to(device)
t_ic = torch.zeros_like(x_ic).to(device)

x_r = (-1 + 2 * torch.rand(N_initial_r, 1, dtype=torch.float32)).to(device)
t_r = torch.rand(N_initial_r, 1, dtype=torch.float32).to(device)

# ==========================================================
# 4. 训练循环 (引入因果律动态权重)
# ==========================================================
model = AB_Causality_PINN(K=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"开始 AB-PINN + RAR + 时间因果律加权训练...")
start_time = time.time()

# 因果律参数 epsilon: 从 5.0 逐渐衰减到 0.0
epsilon_init = 5.0 

for step in range(RAR_steps + 1):
    print(f"\n--- RAR Step {step} | Points: {x_r.shape[0]} ---")
    model.train()
    for epoch in range(Epochs_per_step):
        optimizer.zero_grad()
        
        # 计算当前总迭代进度 (0 到 1)
        current_global_epoch = step * Epochs_per_step + epoch
        progress = current_global_epoch / Total_Epochs
        
        # 动态调整因果律强度
        epsilon = epsilon_init * (1 - progress)
        
        # 计算残差
        res = pde_residual(model, x_r, t_r)
        
        # 【关键创新】：应用时间因果律权重
        # t 越小权重越大，随着训练推进，t 较大的点权重逐渐抬升
        causality_weights = torch.exp(-epsilon * t_r)
        loss_r = torch.mean(causality_weights * (res**2))
        
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
        
        loss = loss_r + 5.0 * loss_ic # 略微提高 IC 权重以配合因果律
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.3e} | Epsilon: {epsilon:.2f}")

    # RAR 采样
    if step < RAR_steps:
        x_cand = (-1 + 2 * torch.rand(100000, 1, dtype=torch.float32)).to(device)
        t_cand = torch.rand(100000, 1, dtype=torch.float32).to(device)
        res_cand = pde_residual(model, x_cand, t_cand)
        res_sq = (res_cand**2).detach()
        _, idx = torch.topk(res_sq.view(-1), N_add_per_step)
        x_r = torch.cat([x_r, x_cand[idx].detach()], dim=0)
        t_r = torch.cat([t_r, t_cand[idx].detach()], dim=0)
        del res_cand, res_sq, x_cand, t_cand

# ==========================================================
# 5. 最终评估与对比
# ==========================================================
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
print(f"\nAB-PINN + RAR + Causality Final Relative L2 Error: {error_l2:.3e}")
print(f"Total Time: {time.time()-start_time:.1f}s")

# 绘图检查激波捕捉
plt.figure(figsize=(6, 4))
plt.plot(x_star, Exact[:, -1], 'k-', label="True")
plt.plot(x_star, u_pred[:, -1], 'b--', label="AB-RAR-Causality")
plt.title("Shock Capturing at t=1.0")
plt.legend()
plt.show()