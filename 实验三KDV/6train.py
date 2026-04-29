import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time
import datetime

# ================= 1. 基础配置 =================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
dtype = torch.float64
torch.set_default_dtype(dtype)

# 换个幸运随机数种子
torch.manual_seed(8888)
np.random.seed(8888)

# ======== 目录设置 (自动识别并加上 KDV 前缀) ========
# 👇 注意修改为你的实际路径
DATA_PATH = '/3241003007/zy/实验三KDV/KdV.mat'
SAVE_DIR = os.path.join(os.path.dirname(DATA_PATH), "KDV_Save")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 KDV 实验数据将保存在: {SAVE_DIR}")

try:
    data = scipy.io.loadmat(DATA_PATH)
    x_exact, t_exact = data["x"].flatten(), data["tt"].flatten()
    u_exact = data["uu"].real.T
    print(f"✅ KDV 数据加载成功: x={x_exact.shape}, t={t_exact.shape}, u={u_exact.shape}")
except Exception as e:
    raise FileNotFoundError(f"❌ 无法读取 {DATA_PATH}，请检查路径！错误信息: {e}")

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
x_min, x_max, t_min, t_max = x_exact.min(), x_exact.max(), t_exact.min(), t_exact.max()

# ================= 2. 傅里叶特征动态自适应基网络 =================
class FourierMLP(nn.Module):
    def __init__(self, layers, fourier_dim=32, sigma=2.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(2, fourier_dim, dtype=dtype, device=device) * sigma, requires_grad=False)
        self.net = nn.Sequential()
        current_dim = fourier_dim * 2 
        
        for i in range(1, len(layers) - 1): 
            self.net.add_module(f'linear_{i}', nn.Linear(current_dim, layers[i]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
            current_dim = layers[i]
            
        self.out = nn.Linear(current_dim, layers[-1])
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        projected = 2.0 * np.pi * (inputs @ self.B)
        fourier_features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)
        return self.out(self.net(fourier_features))

class DynamicABPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = FourierMLP(layers=[2, 80, 80, 80, 80, 1], fourier_dim=32, sigma=2.0) 
        self.experts = nn.ModuleList()                     
        self.centers = []                                  
        self.gamma = 60.0 

    def forward(self, x, t):
        u_pred = self.base_net(x, t)
        for i, expert in enumerate(self.experts):
            cx, ct = self.centers[i]
            weight = torch.exp(-self.gamma * ((x - cx)**2 + (t - ct)**2))
            u_pred = u_pred + weight * expert(x, t)
        return u_pred
        
    def add_expert(self, cx, ct):
        new_expert = FourierMLP(layers=[2, 40, 40, 40, 1], fourier_dim=16, sigma=2.0).to(device)
        nn.init.zeros_(new_expert.out.weight); nn.init.zeros_(new_expert.out.bias)
        self.experts.append(new_expert)
        self.centers.append((cx, ct))
        print(f"🎯 [精确制导空投] 增加第 {len(self.experts)} 个子网，锚点: x={cx:.3f}, t={ct:.3f}")
        return new_expert 

model = DynamicABPINN().to(device)

# ================= 3. 物理残差与极端 RAD 采样 =================
LAMBDA_1, LAMBDA_2 = 1.0, 0.0025
def pde_residual(model, x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

def generate_aais_points(model, n_points=6000): 
    N_uni, N_res = int(0.1 * n_points), int(0.9 * n_points)
    x_uni = torch.rand(N_uni, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
    t_uni = torch.rand(N_uni, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    
    N_pool = 60000 
    x_pool = torch.rand(N_pool, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
    t_pool = torch.rand(N_pool, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    
    x_pool.requires_grad_(True); t_pool.requires_grad_(True)
    prob = (torch.abs(pde_residual(model, x_pool, t_pool)) ** 2).flatten().detach()
    prob = prob / (torch.sum(prob) + 1e-10)
    idx = torch.multinomial(prob, N_res, replacement=True)
    
    x_f = torch.cat([x_uni, x_pool[idx].detach()], dim=0); t_f = torch.cat([t_uni, t_pool[idx].detach()], dim=0)
    x_f.requires_grad_(True); t_f.requires_grad_(True)
    return x_f, t_f

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)
N_bc = 400

def compute_loss(x_f, t_f, t_bc=None):
    res = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(res**2)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    
    if t_bc is None:
        t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        
    x_lb = (torch.ones_like(t_bc) * x_min).requires_grad_(True)
    x_ub = (torch.ones_like(t_bc) * x_max).requires_grad_(True)
    
    u_lb = model(x_lb, t_bc)
    u_ub = model(x_ub, t_bc)
    
    u_x_lb = torch.autograd.grad(u_lb, x_lb, torch.ones_like(u_lb), create_graph=True)[0]
    u_x_ub = torch.autograd.grad(u_ub, x_ub, torch.ones_like(u_ub), create_graph=True)[0]
    
    u_xx_lb = torch.autograd.grad(u_x_lb, x_lb, torch.ones_like(u_x_lb), create_graph=True)[0]
    u_xx_ub = torch.autograd.grad(u_x_ub, x_ub, torch.ones_like(u_x_ub), create_graph=True)[0]
    
    loss_bc_u = torch.mean((u_lb - u_ub)**2)
    loss_bc_ux = torch.mean((u_x_lb - u_x_ub)**2)
    loss_bc_uxx = torch.mean((u_xx_lb - u_xx_ub)**2)
              
    total_loss = loss_f + 100.0 * loss_ic + 50.0 * loss_bc_u + 10.0 * loss_bc_ux + 1.0 * loss_bc_uxx
    return total_loss, res

# ================= 4. Phase 1: Adam 预训练 =================
print("\n🔥 开始 Phase 1: 压力测试完全体 (6000点，有专家网络)")
time_start_global = time.time() 
time_start_phase1 = time.time() 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_f, t_f = generate_aais_points(model, n_points=6000) 

for it in range(15001):
    if it > 2000 and it % 2000 == 0 and it < 14000:
        x_f, t_f = generate_aais_points(model, n_points=6000) 
        current_res = torch.abs(pde_residual(model, x_f, t_f)).detach()
        max_res_idx = torch.argmax(current_res)
        
        if current_res[max_res_idx].item() > 0.05 and len(model.experts) < 5: 
            new_expert = model.add_expert(x_f[max_res_idx].item(), t_f[max_res_idx].item())
            optimizer.add_param_group({'params': new_expert.parameters(), 'lr': 1e-3})

    optimizer.zero_grad()
    loss, _ = compute_loss(x_f, t_f)
    loss.backward()
    optimizer.step()
    
    if it % 1000 == 0:
        with torch.no_grad():
            err = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
        print(f"Adam Iter {it:5d} | Loss: {loss.item():.5e} | Rel L2 Error: {err:.4f} | Experts: {len(model.experts)}")

time_end_phase1 = time.time()
phase1_time = time_end_phase1 - time_start_phase1
print(f"⏱️ Phase 1 耗时: {str(datetime.timedelta(seconds=int(phase1_time)))}")

# ================= 5. Phase 2: 分段 L-BFGS 微调 =================
print("\n🚀 开始 Phase 2: L-BFGS...")
time_start_phase2 = time.time() 

x_anchor = torch.rand(6000, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
t_anchor = torch.rand(6000, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
x_anchor.requires_grad_(True)
t_anchor.requires_grad_(True)
fixed_t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min

lbfgs_iter = 0

for stage in range(3):
    print(f"\n🌊 --- 进入 L-BFGS 第 {stage + 1} 阶段 ---")
    x_rad, t_rad = generate_aais_points(model, n_points=6000) 
    x_f_lbfgs = torch.cat([x_anchor, x_rad], dim=0)
    t_f_lbfgs = torch.cat([t_anchor, t_rad], dim=0)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=8000, max_eval=8000, 
        history_size=50, tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn="strong_wolfe"
    )

    def closure():
        global lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss, _ = compute_loss(x_f_lbfgs, t_f_lbfgs, t_bc=fixed_t_bc)
        loss.backward()
        lbfgs_iter += 1
        if lbfgs_iter % 500 == 0:
            with torch.no_grad():
                e = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
            print(f"L-BFGS Iter {lbfgs_iter:5d} | Stage {stage+1} Loss: {loss.item():.5e} | Rel L2 Error: {e:.5f}")
        return loss

    optimizer_lbfgs.step(closure)
    
time_end_phase2 = time.time()
phase2_time = time_end_phase2 - time_start_phase2
print(f"⏱️ Phase 2 耗时: {str(datetime.timedelta(seconds=int(phase2_time)))}")

# ================= 6. 保存所有结果 =================
time_end_global = time.time()
total_time = time_end_global - time_start_global

model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    max_error = np.max(np.abs(u_exact - u_pred))

save_data = {
    "x_exact": x_exact,
    "t_exact": t_exact,
    "u_exact": u_exact,
    "X_mesh": X_mesh,
    "T_mesh": T_mesh,
    "u_pred": u_pred,
    "final_error": final_error,
    "max_error": max_error,
    "time_adam": phase1_time,
    "time_lbfgs": phase2_time,
    "time_total": total_time,
}

DATA_SAVE_PATH = os.path.join(SAVE_DIR, "KDV_sparse_fourier_results.pt")
torch.save(save_data, DATA_SAVE_PATH)

print(f"\n✅ =============================================")
print(f"✅ [稀疏测试] 完全体 最终相对 L2 误差 (平均): {final_error:.6f}")
print(f"🔥 [稀疏测试] 完全体 最终最大绝对误差 (最差): {max_error:.6f}")
print(f"⏱️  总耗时: {str(datetime.timedelta(seconds=int(total_time)))}")
print(f"🎉 数据及结果已打包保存至: {DATA_SAVE_PATH}")
print(f"✅ =============================================")