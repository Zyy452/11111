import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time
import datetime

# ==========================================================
# 0. 基础配置与目录设置
# ==========================================================
BASE_SAVE_DIR = "/3241003007/zy/save"
EXPERIMENT_DIR = os.path.join(BASE_SAVE_DIR, "AC_Experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print(f"🔥 任务: Fair Baseline (FF-PINN + Hard IC) | 设备: {device}")

# === 1. 数据读取 ===
data_path = "/3241003007/zy/实验二AC/AC.mat"
if not os.path.exists(data_path):
    # 兼容备用路径
    data_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"

data = scipy.io.loadmat(data_path)
x_exact, t_exact = data["x"].flatten(), data["tt"].flatten()
u_exact = data["uu"].T
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

# === 2. 网络定义 (仅傅立叶 + 硬约束) ===
class PeriodicEmbedding(nn.Module):
    def forward(self, x): return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class FairBaselinePINN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)
        
    def forward(self, x, t):
        u_net = self.net(torch.cat([self.embed(x), t], dim=1))
        return torch.tanh(t) * u_net + (x**2 * torch.cos(np.pi * x))

model = FairBaselinePINN().to(device)

# === 3. 训练函数 ===
EPSILON, GAMMA = 0.0001, 5.0
def pde_residual(x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return torch.mean((u_t - EPSILON * u_xx + GAMMA * (u**3 - u))**2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 4. 训练循环 (Phase 1: Adam) ===
start_time = time.time()
loss_history = []
err_history = []
iters = []

print(">>> 开始 Adam 训练...")
for it in range(20001):
    optimizer.zero_grad()
    x_train = torch.rand(8000, 1, device=device, dtype=dtype) * 2 - 1
    t_train = torch.rand(8000, 1, device=device, dtype=dtype)
    x_train.requires_grad_(True); t_train.requires_grad_(True)
    
    loss = pde_residual(x_train, t_train)
    loss.backward()
    optimizer.step()
    
    if it in [10000, 15000]:
        for param_group in optimizer.param_groups: param_group['lr'] *= 0.5
        
    if it % 1000 == 0:
        model.eval()
        with torch.no_grad():
            u_pred_check = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
            curr_err = np.linalg.norm(u_exact - u_pred_check) / np.linalg.norm(u_exact)
        model.train()
        
        loss_history.append(loss.item())
        err_history.append(curr_err)
        iters.append(it)
        print(f"Iter {it:5d} | Loss: {loss.item():.5e} | Rel L2 Err: {curr_err:.4f}")

phase1_time = time.time() - start_time

# === 5. Phase 2: L-BFGS 微调 ===
print("\n>>> 开始 L-BFGS 训练...")
phase2_start = time.time()

x_lbfgs = (torch.rand(10000, 1, device=device, dtype=dtype) * 2 - 1).requires_grad_(True)
t_lbfgs = torch.rand(10000, 1, device=device, dtype=dtype).requires_grad_(True)
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, line_search_fn="strong_wolfe")

def closure():
    lbfgs.zero_grad()
    loss = pde_residual(x_lbfgs, t_lbfgs)
    loss.backward()
    return loss

lbfgs.step(closure)
phase2_time = time.time() - phase2_start

# === 6. 数据保存 ===
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)

total_time = phase1_time + phase2_time

# 将所有指标打包保存
save_data = {
    "u_pred": u_pred,
    "loss_history": np.array(loss_history),
    "err_history": np.array(err_history),
    "iters": np.array(iters),
    "final_error": final_error,
    "time_adam": phase1_time,
    "time_lbfgs": phase2_time,
    "time_total": total_time
}

DATA_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "ac_fair_baseline_results.pt")
torch.save(save_data, DATA_SAVE_PATH)

print(f"\n✨ Fair Baseline 最终相对 L2 误差: {final_error:.4e} ✨")
print(f"🎉 训练数据已保存至: {DATA_SAVE_PATH}")