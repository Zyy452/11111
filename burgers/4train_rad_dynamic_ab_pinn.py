import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import time
import os

# ==========================================================
# 0. 基础配置 (CPU + Float64) 与目录设置
# ==========================================================
save_dir = "/3241003007/zy/save"
os.makedirs(save_dir, exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda") 
dtype = torch.float64 
torch.set_default_dtype(dtype)

print(f" 设备: {device} | 精度: Float64 (High Accuracy Fix)")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据读取
# ==========================================================
try:
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat"
    if not os.path.exists(file_path): 
        file_path = "/3241003007/zy/实验一BG/burgers_shock.mat"
    data = loadmat(file_path)
    x_exact = data["x"].flatten()
    t_exact = data["t"].flatten()
    u_exact_all = data["usol"]
    print(f" 成功读取真值数据自: {file_path}")
except:
    print("错误：找不到 .mat 文件，请检查路径！")
    exit()

X_star = torch.tensor(np.meshgrid(x_exact, t_exact)[0].flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(np.meshgrid(x_exact, t_exact)[1].flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 模型定义 (Dynamic AB-PINN)
# ==========================================================
class DeepSubNet(nn.Module):
    def __init__(self, hidden=64, is_new_addition=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        if is_new_addition:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

class DynamicABPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.subnets = nn.ModuleList([DeepSubNet(hidden), DeepSubNet(hidden)])
        self.centers = nn.ParameterList([
            nn.Parameter(torch.tensor([-0.5, 0.25], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([0.5, 0.75], device=device, dtype=dtype))
        ])
        self.log_gammas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype))
        ])

    def add_subdomain(self, mu_init):
        new_net = DeepSubNet(self.hidden, is_new_addition=True).to(device)
        mu = nn.Parameter(torch.tensor(mu_init, device=device, dtype=dtype).view(1, 2))
        lg = nn.Parameter(torch.tensor([2.5], device=device, dtype=dtype)) 
        
        self.subnets.append(new_net)
        self.centers.append(mu)
        self.log_gammas.append(lg)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        logits = []
        for i in range(len(self.subnets)):
            dist_sq = torch.sum((xt - self.centers[i])**2, dim=1, keepdim=True)
            logits.append(-torch.exp(self.log_gammas[i]) * dist_sq)
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1)
        
        u_out = 0
        for i in range(len(self.subnets)):
            u_out += weights[:, i] * self.subnets[i](x, t)
        return u_out

# ==========================================================
# 3. 采样与Loss (RAD 变体采样)
# ==========================================================
def pde_residual(model, x, t, current_nu):
    u = model(x, t)
    u_dt, u_dx = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    return u_dt + u * u_dx - current_nu * u_dxx

def redistribute_points_hybrid(model, n_total, current_nu):
    model.eval()
    
    n_uniform = int(0.4 * n_total) 
    x_uni = (-1 + 2 * torch.rand(n_uniform, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_uni = torch.rand(n_uniform, 1, device=device, dtype=dtype).requires_grad_(True)
    
    n_adaptive = n_total - n_uniform
    n_cand = n_adaptive * 5 
    x_cand = (-1 + 2 * torch.rand(n_cand, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_cand = torch.rand(n_cand, 1, device=device, dtype=dtype).requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        res = pde_residual(model, x_cand, t_cand, current_nu)
        score = torch.abs(res).detach().cpu().numpy().flatten()
    
    score = score ** 2 
    pdf = score / (np.sum(score) + 1e-10)
    idx = np.random.choice(n_cand, size=n_adaptive, p=pdf, replace=False)
    
    x_adapt = x_cand[idx].detach().requires_grad_(True)
    t_adapt = t_cand[idx].detach().requires_grad_(True)
    
    x_final = torch.cat([x_uni, x_adapt], dim=0)
    t_final = torch.cat([t_uni, t_adapt], dim=0)
    
    model.train()
    return x_final, t_final

# ==========================================================
# 4. 训练设置
# ==========================================================
target_nu = 0.01 / np.pi
start_nu = 10.0 * target_nu 
ADAM_EPOCHS = 20000 
N_total = 3000  

model = DynamicABPINN(hidden=64).to(device)

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

t_bc = torch.rand(400, 1, device=device, dtype=dtype) 
x_bc_l = -1.0 * torch.ones_like(t_bc)
x_bc_r = 1.0 * torch.ones_like(t_bc)
u_bc_val = torch.zeros_like(t_bc)

def compute_loss_val(model, x_r, t_r, nu_val):
    res = pde_residual(model, x_r, t_r, nu_val)
    loss_r = torch.mean(res**2)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    loss_bc = torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + \
              torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2)
    return loss_r + 100.0 * (loss_ic + loss_bc), res

def validate(model):
    model.eval()
    with torch.no_grad():
        u_pred = model(X_star, T_star).cpu().numpy().reshape(len(t_exact), len(x_exact)).T
        error = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)
    model.train()
    return error

curr_nu = start_nu
x_r, t_r = redistribute_points_hybrid(model, N_total, curr_nu)

# ==========================================================
# Phase 1: Adam
# ==========================================================
print(">>> Phase 1: Adam (With Validation)...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
start_time = time.time()

for epoch in range(ADAM_EPOCHS + 1):
    if epoch < 10000:
        curr_nu = start_nu - (start_nu - target_nu) * (epoch / 10000)
    else:
        curr_nu = target_nu

    if epoch > 0 and epoch % 2500 == 0:
        x_r, t_r = redistribute_points_hybrid(model, N_total, curr_nu)

    optimizer.zero_grad()
    loss, res_tensor = compute_loss_val(model, x_r, t_r, curr_nu)
    loss.backward()
    optimizer.step()

    if epoch > 2000 and epoch % 3000 == 0:
        max_res = res_tensor.abs().max().item()
        if len(model.subnets) < 6 and max_res > 0.02: 
            idx = torch.argmax(res_tensor.abs())
            new_mu = [x_r[idx].item(), t_r[idx].item()]
            model.add_subdomain(new_mu)
            print(f"⚡️ [Add Subnet] -> Total: {len(model.subnets)} | Center: {new_mu}")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    if epoch % 1000 == 0:
        val_err = validate(model)
        print(f"Ep {epoch:5d} | Loss: {loss.item():.5e} | Val L2: {val_err:.4f}")

print(f"Adam Time: {time.time()-start_time:.1f}s")

# ==========================================================
# Phase 2: L-BFGS (High Density)
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning (High Density Grid)...")

N_fine = 8000 
x_r, t_r = redistribute_points_hybrid(model, N_fine, target_nu)

lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=10000, max_eval=10000, 
    history_size=100, 
    tolerance_grad=1e-15, tolerance_change=1e-15,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    loss, _ = compute_loss_val(model, x_r, t_r, target_nu)
    loss.backward()
    return loss

lbfgs.step(closure)
print(f"L-BFGS Final Loss: {closure().item():.5e}")

total_train_time = time.time() - start_time

# ==========================================================
# 最终验证与数据保存
# ==========================================================
final_error = validate(model)
print(f"\n✨✨✨ Corrected Final Relative L2 Error: {final_error:.4e} ✨✨✨")
print(f"总训练耗时: {total_train_time:.1f}s")

model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(len(t_exact), len(x_exact)).T
    centers_final = torch.stack([c.view(-1).detach() for c in model.centers]).cpu().numpy()
    gammas_final = torch.stack([torch.exp(lg).view(-1).detach() for lg in model.log_gammas]).cpu().numpy()

# 提取用于 RAD 可视化的最终采样点
x_r_plot = x_r.detach().cpu().numpy().flatten()
t_r_plot = t_r.detach().cpu().numpy().flatten()

model_save_path = os.path.join(save_dir, "rad_dynamic_ab_pinn_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"模型权重已保存至: {model_save_path}")

data_save_path = os.path.join(save_dir, "rad_dynamic_ab_pinn_results.npz")
np.savez(data_save_path, 
         u_pred=u_pred, 
         u_exact=u_exact_all, 
         x=x_exact, 
         t=t_exact,
         centers=centers_final, 
         gammas=gammas_final,
         x_r=x_r_plot,       # 【新增】保存最终的 RAD 采样空间坐标
         t_r=t_r_plot,       # 【新增】保存最终的 RAD 采样时间坐标
         train_time=total_train_time,
         error_l2=final_error)
print(f"预测结果及参数数据已保存至: {data_save_path}")