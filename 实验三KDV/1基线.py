import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os

# ==========================================================
# 0. 基础配置 
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print("🔥 任务: Standard PINN (Baseline 1) - 严格对齐版 (N=8000, Width=64)")

torch.manual_seed(1234)
np.random.seed(1234)

# 注意修改路径
DATA_PATH = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/KdV.mat"
SAVE_DIR = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================================
# 1. 数据读取
# ==========================================================
data = scipy.io.loadmat(DATA_PATH)
x_exact = data["x"].flatten()   
t_exact = data["tt"].flatten()  
u_exact = data["uu"].real.T     

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 标准网络 (对齐宽度 64)
# ==========================================================
class StandardPINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 1]):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.net.add_module('output', nn.Linear(layers[-2], layers[-1]))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

model = StandardPINN().to(device)

# ==========================================================
# 3. 采样与物理损失 (对齐 N=8000, 完整边界与权重)
# ==========================================================
LAMBDA_1 = 1.0
LAMBDA_2 = 0.0025

def pde_residual(x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

# 全局对齐：仅使用 8000 个采样点
N_f = 8000  
N_bc = 400   
x_min, x_max = x_exact.min(), x_exact.max()
t_min, t_max = t_exact.min(), t_exact.max()

x_f = (torch.rand(N_f, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min).requires_grad_(True)
t_f = (torch.rand(N_f, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min).requires_grad_(True)

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)

t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
x_bc_lb = (x_min * torch.ones_like(t_bc)).requires_grad_(True)
x_bc_ub = (x_max * torch.ones_like(t_bc)).requires_grad_(True)

def compute_loss():
    res = pde_residual(x_f, t_f)
    loss_f = torch.mean(res**2)
    
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    # 完整的一二阶周期边界
    u_lb, u_ub = model(x_bc_lb, t_bc), model(x_bc_ub, t_bc)
    u_x_lb = torch.autograd.grad(u_lb, x_bc_lb, torch.ones_like(u_lb), create_graph=True)[0]
    u_x_ub = torch.autograd.grad(u_ub, x_bc_ub, torch.ones_like(u_ub), create_graph=True)[0]
    u_xx_lb = torch.autograd.grad(u_x_lb, x_bc_lb, torch.ones_like(u_x_lb), create_graph=True)[0]
    u_xx_ub = torch.autograd.grad(u_x_ub, x_bc_ub, torch.ones_like(u_x_ub), create_graph=True)[0]
    
    loss_bc_u = torch.mean((u_lb - u_ub)**2)
    loss_bc_ux = torch.mean((u_x_lb - u_x_ub)**2)
    loss_bc_uxx = torch.mean((u_xx_lb - u_xx_ub)**2)
              
    # 严格对齐完全体的权重
    total_loss = loss_f + 30.0 * loss_ic + 20.0 * loss_bc_u + 1.0 * loss_bc_ux + 0.1 * loss_bc_uxx
    return total_loss

# ==========================================================
# 4. 训练流程 (Adam 20000 步)
# ==========================================================
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
ADAM_ITERS = 20000
start_time = time.time()

print(f"\n>>> Phase 1: Adam Optimization ({ADAM_ITERS} Iters)")
for it in range(ADAM_ITERS + 1):
    optimizer_adam.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer_adam.step()
    
    if it % 1000 == 0:
        with torch.no_grad():
            u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
            err = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
        print(f"Iter {it:5d} | Loss: {loss.item():.5e} | Rel L2 Error: {err:.4f}")

print(f"\n>>> Phase 2: L-BFGS Optimization")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=5000, max_eval=5000,
    history_size=50, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe"
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)

# ==========================================================
# 5. 结果可视化
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)

print(f"\n✨ Standard PINN 最终相对 L2 误差: {final_error:.4e} ✨")

fig = plt.figure(figsize=(18, 5))
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"Standard PINN Predict (Error: {final_error:.4f})")
plt.colorbar(im1, ax=ax1)

ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title("Absolute Error")
plt.colorbar(im2, ax=ax2)

ax3 = plt.subplot(1, 3, 3)
idx_t = int(0.7 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t, :], 'k-', linewidth=2, label="Exact")
ax3.plot(x_exact, u_pred[idx_t, :], 'r--', linewidth=2, label="Predict")
ax3.set_title(f"Slice at t={t_exact[idx_t]:.2f}")
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Baseline1_StandardPINN.png"), dpi=200)
plt.show()