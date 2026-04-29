import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time
import datetime

# ================= 1. 基础配置 =================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda") # 如果有GPU可改为 cuda
dtype = torch.float64
torch.set_default_dtype(dtype)

torch.manual_seed(1234)
np.random.seed(1234)

# ======== 目录设置 (加上 KDV 前缀区分) ========
DATA_PATH = "/3241003007/zy/实验三KDV/KdV.mat"
SAVE_DIR = "/3241003007/zy/save"

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

# ================= 2. 动态自适应基网络 (瘦身版) =================
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
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, t): 
        return self.out(self.net(torch.cat([x, t], dim=1)))

class DynamicABPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = MLP(layers=[2, 80, 80, 80, 80, 1]) 
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
        nn.init.zeros_(new_expert.out.weight); nn.init.zeros_(new_expert.out.bias)
        self.experts.append(new_expert)
        self.centers.append((cx, ct))
        print(f"🎯 [精确制导空投] 增加第 {len(self.experts)} 个子网，锚点: x={cx:.3f}, t={ct:.3f}")
        return new_expert 

model = DynamicABPINN().to(device)

# ================= 3. 物理残差与 RAD 采样 =================
LAMBDA_1, LAMBDA_2 = 1.0, 0.0025
def pde_residual(model, x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

def generate_aais_points(model, n_points=10000):
    N_uni, N_res = int(0.3 * n_points), int(0.7 * n_points)
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
              
    total_loss = loss_f + 30.0 * loss_ic + 20.0 * loss_bc_u + 1.0 * loss_bc_ux + 0.1 * loss_bc_uxx
    return total_loss, res

# ================= 4. Phase 1: Adam 预训练 =================
print("\n🔥 开始 Phase 1: Combined (RAD采样 + 架构生长) Adam")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_f, t_f = generate_aais_points(model, n_points=15000) 

phase1_start = time.time()

for it in range(15001):
    if it > 2000 and it % 2000 == 0 and it < 14000:
        x_f, t_f = generate_aais_points(model, n_points=15000) 
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

phase1_time = time.time() - phase1_start
print(f"✅ Phase 1 完成，耗时: {str(datetime.timedelta(seconds=int(phase1_time)))}")

# ================= 5. Phase 2: 分段 L-BFGS 微调 =================
print("\n🚀 开始 Phase 2: 多阶段 L-BFGS 高精度微调 (混合锚点策略)...")
phase2_start = time.time()

x_anchor = torch.rand(5000, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
t_anchor = torch.rand(5000, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
x_anchor.requires_grad_(True)
t_anchor.requires_grad_(True)
fixed_t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min

lbfgs_iter = 0

for stage in range(3):
    print(f"\n🌊 --- 进入 L-BFGS 第 {stage + 1} 阶段 (重新撒点 + 固定锚点) ---")
    x_rad, t_rad = generate_aais_points(model, n_points=10000)
    
    x_f_lbfgs = torch.cat([x_anchor, x_rad], dim=0)
    t_f_lbfgs = torch.cat([t_anchor, t_rad], dim=0)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=5000, max_eval=5000, 
        history_size=50, tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn="strong_wolfe"
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

phase2_time = time.time() - phase2_start
print(f"✅ Phase 2 完成，耗时: {str(datetime.timedelta(seconds=int(phase2_time)))}")

# ================= 6. 保存所有结果 (用于可视化) =================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    
    # 💥 【新增】提取子网络中心和超参数
    centers_np = np.array(model.centers)  # 提取 [(cx, ct), ...] 列表为 numpy 数组
    gamma_val = model.gamma               # 提取高斯衰减系数
    
    # 💥 【新增】提取最后一轮的采样点 (用于展示 RADS 采样效果)
    x_vis = x_f_lbfgs.detach().cpu().numpy().flatten()
    t_vis = t_f_lbfgs.detach().cpu().numpy().flatten()

total_time = phase1_time + phase2_time

save_data = {
    "x_exact": x_exact,
    "t_exact": t_exact,
    "u_exact": u_exact,
    "X_mesh": X_mesh,
    "T_mesh": T_mesh,
    "u_pred": u_pred,
    "centers": centers_np,   # <--- 【新增】
    "gamma": gamma_val,      # <--- 【新增】
    "x_vis": x_vis,          # <--- 【新增】
    "t_vis": t_vis,          # <--- 【新增】
    "final_error": final_error,
    "time_adam": phase1_time,
    "time_lbfgs": phase2_time,
    "time_total": total_time,
}

DATA_SAVE_PATH = os.path.join(SAVE_DIR, "KDV_rads_abpinn_results.pt")
torch.save(save_data, DATA_SAVE_PATH)

print(f"\n✅ =============================================")
print(f"✅ 最终相对 L2 误差: {final_error:.6f}")
print(f"⏱️ 训练总耗时: {str(datetime.timedelta(seconds=int(total_time)))}")
print(f"🎉 数据及结果已打包保存至: {DATA_SAVE_PATH}")
print(f"✅ =============================================")