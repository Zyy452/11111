import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import os

# ==========================================================
# 0. 基础配置与目录设置
# ==========================================================
BASE_SAVE_DIR = "/3241003007/zy/save"
EXP_DIR = os.path.join(BASE_SAVE_DIR, "AC_Experiment")
os.makedirs(EXP_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64 
torch.set_default_dtype(dtype)
torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备
# ==========================================================
data = scipy.io.loadmat("/3241003007/zy/实验二AC/AC.mat")
x_exact = data["x"].flatten()   
t_exact = data["tt"].flatten()  
u_exact_all = data["uu"].T      # (201, 512)
X, T = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 模型定义 (FF-PINN / Periodic Embedding)
# ==========================================================
class PeriodicEmbedding(nn.Module):
    def forward(self, x):
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class FFPINN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t):
        features = torch.cat([self.embed(x), t], dim=1)
        u_net = self.net(features)
        # 满足初始条件 u(x,0) = x^2 * cos(pi*x)
        return torch.tanh(t) * u_net + (x**2 * torch.cos(np.pi * x))

model = FFPINN(hidden=128).to(device)

# ==========================================================
# 3. 物理约束
# ==========================================================
EPSILON, GAMMA = 0.0001, 5.0
def pde_residual(x, t):
    u = model(x, t)
    u_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    return u_dt - EPSILON * u_dxx + GAMMA * (u**3 - u)

# ==========================================================
# 4. 训练循环 (Phase 1: Adam)
# ==========================================================
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
err_history = [] 
iters = []       

print(">>> [Data Prep] Fair Baseline (FF-PINN) Training (tracking L2 history)...")
for epoch in range(20001): 
    # 【修复】：加上 .detach() 斩断隐式计算图
    x_f = (-1 + 2 * torch.rand(8000, 1, device=device, dtype=dtype)).detach().requires_grad_(True)
    t_f = torch.rand(8000, 1, device=device, dtype=dtype).detach().requires_grad_(True)

    optimizer_adam.zero_grad()
    res = pde_residual(x_f, t_f)
    loss = torch.mean(res**2)
    loss.backward()
    optimizer_adam.step()
    
    if epoch % 1000 == 0:
        model.eval()
        with torch.no_grad():
            u_p = model(X_star, T_star).cpu().numpy().reshape(u_exact_all.shape)
            c_e = np.linalg.norm(u_exact_all - u_p) / np.linalg.norm(u_exact_all)
        model.train()
        
        loss_history.append(loss.item())
        err_history.append(c_e)
        iters.append(epoch)
        print(f"Ep {epoch:5d} | Loss: {loss.item():.4e} | Rel L2: {c_e:.4f}")

# ==========================================================
# 5. Phase 2: L-BFGS
# ==========================================================
print("\n>>> 开始微调 (L-BFGS)...")
# 【修复】：同样加上 .detach()
x_f_full = (-1 + 2 * torch.rand(10000, 1, device=device, dtype=dtype)).detach().requires_grad_(True)
t_f_full = torch.rand(10000, 1, device=device, dtype=dtype).detach().requires_grad_(True)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    loss = torch.mean(pde_residual(x_f_full, t_f_full)**2)
    loss.backward()
    return loss
lbfgs.step(closure)

# ==========================================================
# 6. 最终验证与保存
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact_all.shape)
final_error = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)

save_data = {
    "u_pred": u_pred, 
    "u_exact_all": u_exact_all, 
    "loss_history": np.array(loss_history),
    "err_history": np.array(err_history), 
    "iters": np.array(iters),
    "final_error": final_error,
}
DATA_SAVE_PATH = os.path.join(EXP_DIR, "ac_fair_baseline_results.pt")
torch.save(save_data, DATA_SAVE_PATH)
print(f"\n✨✨ FF-PINN Final Rel L2 Error: {final_error:.4e} | 已保存至 {DATA_SAVE_PATH}")