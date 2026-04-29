import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import os

# ==========================================================
# 0. 基础配置与目录设置
# ==========================================================
save_dir = "/3241003007/zy/save"
EXP_DIR = os.path.join(save_dir, "AC_Experiment")
os.makedirs(EXP_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dtype = torch.float64 
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备
# ==========================================================
data = scipy.io.loadmat("/3241003007/zy/实验二AC/AC.mat")
x_exact = data["x"].flatten()   
t_exact = data["tt"].flatten()  
u_exact_raw = data["uu"]        
X, T = np.meshgrid(x_exact, t_exact)
u_exact_all = u_exact_raw.T     # (201, 512)
X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 模型定义
# ==========================================================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f'tanh_{i}', nn.Tanh())
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, t): return self.net(torch.cat([x, t], dim=1))

layers = [2, 128, 128, 128, 128, 1]
model = PINN(layers).to(device)

# ==========================================================
# 3. 物理约束
# ==========================================================
EPSILON, GAMMA = 0.0001, 5.0
def pde_residual(model, x, t):
    u = model(x, t)
    u_grads = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dt, u_dx = u_grads[0], u_grads[1]
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    return u_dt - EPSILON * u_dxx + GAMMA * (u**3 - u)

# ==========================================================
# 4. 训练数据采样 (已彻底修复计算图问题)
# ==========================================================
N_f, N_b = 10000, 200   

# 加入 .detach() 斩断隐式计算图，使其成为纯粹的叶子节点
x_f = (-1 + 2 * torch.rand(N_f, 1, device=device, dtype=dtype)).detach().requires_grad_(True)
t_f = torch.rand(N_f, 1, device=device, dtype=dtype).detach().requires_grad_(True)

x_ic = (-1 + 2 * torch.rand(N_b, 1, device=device, dtype=dtype)).detach()
t_ic = torch.zeros_like(x_ic).detach()
u_ic_exact = x_ic**2 * torch.cos(np.pi * x_ic)

t_bc = torch.rand(N_b, 1, device=device, dtype=dtype).detach().requires_grad_(True)

# 使用 full_like 直接生成常数张量，避免乘法产生的计算图
x_bc_left = torch.full_like(t_bc, -1.0, requires_grad=True)
x_bc_right = torch.full_like(t_bc, 1.0, requires_grad=True)

# ==========================================================
# 5. 训练循环 (Phase 1: Adam)
# ==========================================================
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
start_time = time.time()
loss_history = []
err_history = []
iters = []

print(">>> [Data Prep] Baseline Adam Training (tracking L2 history)...")
for epoch in range(5001): 
    optimizer_adam.zero_grad()
    res = pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(res**2)
    u_ic_err = model(x_ic, t_ic) - u_ic_exact
    loss_ic = torch.mean(u_ic_err**2)
    u_l, u_r = model(x_bc_left, t_bc), model(x_bc_right, t_bc)
    loss_bc_u = torch.mean((u_l - u_r)**2)
    u_x_l = torch.autograd.grad(u_l, x_bc_left, torch.ones_like(u_l), create_graph=True)[0]
    u_x_r = torch.autograd.grad(u_r, x_bc_right, torch.ones_like(u_r), create_graph=True)[0]
    loss_bc_ux = torch.mean((u_x_l - u_x_r)**2)
    loss = loss_pde + 100.0 * loss_ic + 10.0 * (loss_bc_u + loss_bc_ux)
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
        print(f"Ep {epoch:5d} | Loss: {loss.item():.2e} | Rel L2: {c_e:.4f}")

# ==========================================================
# Phase 2: L-BFGS
# ==========================================================
print("\n>>> 开始微调 (L-BFGS)...")
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=2000, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    loss_pde = torch.mean(pde_residual(model, x_f, t_f)**2)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic_exact)**2)
    u_l, u_r = model(x_bc_left, t_bc), model(x_bc_right, t_bc)
    u_x_l = torch.autograd.grad(u_l, x_bc_left, torch.ones_like(u_l), create_graph=True)[0]
    u_x_r = torch.autograd.grad(u_r, x_bc_right, torch.ones_like(u_r), create_graph=True)[0]
    loss = loss_pde + 100.0 * loss_ic + 10.0 * (torch.mean((u_l - u_r)**2) + torch.mean((u_x_l - u_x_r)**2))
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
DATA_SAVE_PATH = os.path.join(EXP_DIR, "ac_baseline_pinn_results.pt")
torch.save(save_data, DATA_SAVE_PATH)
print(f"\n✨✨ Baseline Final Relative L2 Error: {final_error:.4e} | 已保存至 {DATA_SAVE_PATH}")