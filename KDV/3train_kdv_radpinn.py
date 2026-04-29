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
os.makedirs(save_dir, exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 自动检测 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}，正在使用 GPU 训练！")
else:
    device = torch.device("cpu")
    print(f"⚠️ 未检测到 GPU，已回退至 CPU 训练。")

dtype = torch.float64
torch.set_default_dtype(dtype)

print("🔥 任务: Baseline 3 (纯 RAD-PINN) - 严格对齐版 (N=8000, 完整边界)")
torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据读取与预处理 (兼容多环境)
# ==========================================================
def get_kdv_data():
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/KdV.mat"
    if not os.path.exists(file_path):
        file_path = "/3241003007/zy/实验三KDV/KdV.mat"
        
    try:
        data = scipy.io.loadmat(file_path)
        x_exact = data["x"].flatten()
        t_exact = data["tt"].flatten()
        u_exact = data["uu"].real.T
        print(f"✅ 成功加载 KdV.mat 真值数据: x={x_exact.shape}, t={t_exact.shape}, u={u_exact.shape}")
        return x_exact, t_exact, u_exact
    except Exception as e:
        print(f"❌ 读取数据失败: {e}，使用伪数据占位")
        x_exact = np.linspace(-1, 1, 512)
        t_exact = np.linspace(0, 1, 201)
        u_exact = np.zeros((201, 512))
        return x_exact, t_exact, u_exact

x_exact, t_exact, u_exact = get_kdv_data()

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
x_min, x_max = x_exact.min(), x_exact.max()
t_min, t_max = t_exact.min(), t_exact.max()

# ==========================================================
# 2. 网络架构 (对齐宽度 64，无动态节点)
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
# 3. 物理残差与 RDES 演化采样 (对齐8000点与完整边界)
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

def generate_aais_points(model):
    N_uni, N_res = int(0.3 * N_f), int(0.7 * N_f) 
    x_uni = torch.rand(N_uni, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
    t_uni = torch.rand(N_uni, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    
    # 建立巨大候选池用于计算残差概率
    N_pool = 100000
    x_pool = (torch.rand(N_pool, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min).requires_grad_(True)
    t_pool = (torch.rand(N_pool, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min).requires_grad_(True)
    
    res_pool = torch.abs(pde_residual(model, x_pool, t_pool))
    prob = (res_pool ** 2).flatten().detach() 
    prob = prob / (torch.sum(prob) + 1e-10) 
    
    idx = torch.multinomial(prob, N_res, replacement=True)
    x_res = x_pool[idx].detach()
    t_res = t_pool[idx].detach()
    
    x_f = torch.cat([x_uni, x_res], dim=0).requires_grad_(True)
    t_f = torch.cat([t_uni, t_res], dim=0).requires_grad_(True)
    return x_f, t_f

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)

def compute_loss(x_f, t_f, t_bc=None):
    loss_f = torch.mean(pde_residual(model, x_f, t_f)**2)
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
    return total_loss

# ==========================================================
# 4. 训练流程与时间统计
# ==========================================================
global_start_time = time.time() 

print("\n>>> 开始训练 Phase 1: Adam 20000 步")
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
x_f, t_f = generate_aais_points(model) 

for it in range(20001):
    if it > 0 and it % 2000 == 0:
        x_f, t_f = generate_aais_points(model) 
        
    optimizer_adam.zero_grad()
    loss = compute_loss(x_f, t_f) 
    loss.backward()
    optimizer_adam.step()
    
    if it % 1000 == 0:
        with torch.no_grad():
            u_pred_check = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
            err = np.linalg.norm(u_exact - u_pred_check) / np.linalg.norm(u_exact)
        print(f"Iter {it:5d} | Loss: {loss.item():.5e} | Rel L2 Error: {err:.4f}")

phase1_time = time.time() - global_start_time
print(f"⏱️ Phase 1 (Adam) 耗时: {phase1_time:.2f} 秒")

print("\n>>> 开始训练 Phase 2: L-BFGS 优化")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=5000, max_eval=5000, 
    history_size=50, tolerance_grad=1e-7, line_search_fn="strong_wolfe"
)
x_f, t_f = generate_aais_points(model) 
fixed_t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    optimizer_lbfgs.zero_grad()
    loss = compute_loss(x_f, t_f, t_bc=fixed_t_bc) 
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
print(f"⏱️ 纯 RAD-PINN 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

# ==========================================================
# 5. 数据提取与保存
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)

print(f"✨ Baseline 3 最终相对 L2 误差: {final_error:.4e} ✨")

# 保存模型权重
model_save_path = os.path.join(save_dir, "kdv_radpinn_model.pth")
torch.save(model.state_dict(), model_save_path)

# 保存可视化所需数据
data_save_path = os.path.join(save_dir, "kdv_radpinn_results.npz")
np.savez(data_save_path, 
         u_pred=u_pred, 
         u_exact=u_exact, 
         X_mesh=X_mesh, 
         T_mesh=T_mesh,
         x_exact=x_exact,
         t_exact=t_exact,
         final_error=final_error,
         total_time=total_time)

print(f"✅ 数据及模型已统一保存至 {save_dir} 目录下")