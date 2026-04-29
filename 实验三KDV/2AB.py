import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import time

# ==========================================================
# 0. 基础配置
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print("🔥 任务: Baseline 2 (纯 AB-PINN) - 严格对齐版 (N=8000, 完整边界)")
torch.manual_seed(1234)
np.random.seed(1234)

# 注意修改路径
DATA_PATH = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/KdV.mat"
SAVE_DIR = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==========================================================
# 1. 数据读取与预处理
# ==========================================================
data = scipy.io.loadmat(DATA_PATH)
x_exact = data["x"].flatten()
t_exact = data["tt"].flatten()
u_exact = data["uu"].real.T

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
x_min, x_max = x_exact.min(), x_exact.max()
t_min, t_max = t_exact.min(), t_exact.max()

# ==========================================================
# 2. 动态自适应基网络 (对齐主网64，专家40)
# ==========================================================
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.out = nn.Linear(layers[-2], layers[-1])
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t): 
        return self.out(self.net(torch.cat([x, t], dim=1)))

class DynamicABPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = MLP(layers=[2, 64, 64, 64, 64, 1]) 
        self.experts = nn.ModuleList()                     
        self.centers = []                                  
        self.gamma = 40.0                                 

    def forward(self, x, t):
        u_pred = self.base_net(x, t)
        for i, expert in enumerate(self.experts):
            cx, ct = self.centers[i]
            weight = torch.exp(-self.gamma * ((x - cx)**2 + (t - ct)**2))
            u_pred = u_pred + weight * expert(x, t)
        return u_pred
        
    def add_expert(self, cx, ct):
        new_expert = MLP(layers=[2, 40, 40, 40, 1]).to(device)
        nn.init.zeros_(new_expert.out.weight)
        nn.init.zeros_(new_expert.out.bias)
        self.experts.append(new_expert)
        self.centers.append((cx, ct))
        print(f"🌟 [空投专家] 第 {len(self.experts)} 个子网，锚点: x={cx:.3f}, t={ct:.3f}")
        return new_expert

model = DynamicABPINN().to(device)

# ==========================================================
# 3. 物理残差与边界配置 (对齐8000点与完整边界)
# ==========================================================
LAMBDA_1 = 1.0
LAMBDA_2 = 0.0025

def pde_residual(model, x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

N_f = 8000 
N_bc = 400

def generate_uniform_points():
    x_f = (torch.rand(N_f, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min).requires_grad_(True)
    t_f = (torch.rand(N_f, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min).requires_grad_(True)
    return x_f, t_f

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)

def compute_loss(x_f, t_f, t_bc=None):
    res = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(res**2)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    
    if t_bc is None: 
        t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        
    x_lb = (x_min * torch.ones_like(t_bc)).requires_grad_(True)
    x_ub = (x_max * torch.ones_like(t_bc)).requires_grad_(True)
    
    u_lb, u_ub = model(x_lb, t_bc), model(x_ub, t_bc)
    u_x_lb = torch.autograd.grad(u_lb, x_lb, torch.ones_like(u_lb), create_graph=True)[0]
    u_x_ub = torch.autograd.grad(u_ub, x_ub, torch.ones_like(u_ub), create_graph=True)[0]
    u_xx_lb = torch.autograd.grad(u_x_lb, x_lb, torch.ones_like(u_x_lb), create_graph=True)[0]
    u_xx_ub = torch.autograd.grad(u_x_ub, x_ub, torch.ones_like(u_x_ub), create_graph=True)[0]
    
    loss_bc_u = torch.mean((u_lb - u_ub)**2)
    loss_bc_ux = torch.mean((u_x_lb - u_x_ub)**2)
    loss_bc_uxx = torch.mean((u_xx_lb - u_xx_ub)**2)
    
    total_loss = loss_f + 30.0 * loss_ic + 20.0 * loss_bc_u + 1.0 * loss_bc_ux + 0.1 * loss_bc_uxx
    return total_loss, res

# ==========================================================
# 4. 训练流程与时间统计
# ==========================================================
global_start_time = time.time()

print("\n>>> 开始 Phase 1: Adam 20000 步")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_f, t_f = generate_uniform_points() 

for it in range(20001):
    if it > 0 and it % 2000 == 0:
        x_f, t_f = generate_uniform_points() 
        current_res = torch.abs(pde_residual(model, x_f, t_f)).detach()
        max_res_idx = torch.argmax(current_res)
        
        if current_res[max_res_idx].item() > 0.05 and len(model.experts) < 5: 
            new_exp = model.add_expert(x_f[max_res_idx].item(), t_f[max_res_idx].item())
            optimizer.add_param_group({'params': new_exp.parameters(), 'lr': 1e-3})

    optimizer.zero_grad()
    loss, _ = compute_loss(x_f, t_f)
    loss.backward()
    optimizer.step()
    
    if it % 1000 == 0:
        with torch.no_grad():
            err = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
        print(f"Adam Iter {it:5d} | Loss: {loss.item():.5e} | Error: {err:.4f} | Experts: {len(model.experts)}")

phase1_time = time.time() - global_start_time
print(f"⏱️ Phase 1 (Adam) 耗时: {phase1_time:.2f} 秒")

print("\n>>> 开始 Phase 2: L-BFGS 高精度微调...")
x_f_lbfgs, t_f_lbfgs = generate_uniform_points()
fixed_t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, max_eval=5000, history_size=50, tolerance_grad=1e-5, line_search_fn="strong_wolfe")

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    optimizer_lbfgs.zero_grad()
    loss, _ = compute_loss(x_f_lbfgs, t_f_lbfgs, t_bc=fixed_t_bc)
    loss.backward()
    lbfgs_iter += 1
    if lbfgs_iter % 500 == 0:
        with torch.no_grad():
            e = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
        print(f"L-BFGS Iter {lbfgs_iter:4d} | Loss: {loss.item():.5e} | Rel L2 Error: {e:.5f}")
    return loss

optimizer_lbfgs.step(closure)

total_time = time.time() - global_start_time
print(f"\n✨✨=========================================✨✨")
print(f"⏱️ 纯 AB-PINN 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

# ==========================================================
# 5. 结果评估与保存
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)

print(f"✨ Baseline 2 最终相对 L2 误差: {final_error:.4e} ✨")

fig = plt.figure(figsize=(18, 5))

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"AB-PINN Predict (Error: {final_error:.4f})")
ax1.set_xlabel('t'); ax1.set_ylabel('x')
plt.colorbar(im1, ax=ax1)

ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title("Absolute Error (AB-Only)")
ax2.set_xlabel('t'); ax2.set_ylabel('x')
plt.colorbar(im2, ax=ax2)

ax3 = plt.subplot(1, 3, 3)
idx_t1, idx_t2 = int(0.5 * len(t_exact)), int(0.8 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t1, :], 'k-', linewidth=2, label=f"Exact t={t_exact[idx_t1]:.2f}")
ax3.plot(x_exact, u_pred[idx_t1, :], 'r--', linewidth=2, label="Predict")
ax3.plot(x_exact, u_exact[idx_t2, :], 'b-', linewidth=2, label=f"Exact t={t_exact[idx_t2]:.2f}")
ax3.plot(x_exact, u_pred[idx_t2, :], 'g--', linewidth=2, label="Predict")
ax3.set_title("Wave Profile Slices")
ax3.set_xlabel('x'); ax3.set_ylabel('u(x,t)')
ax3.legend()

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, "Baseline2_AB_Only.png")
plt.savefig(save_path, dpi=200)
print(f"✅ Baseline 2 图片已成功保存至: {save_path}")
plt.show()