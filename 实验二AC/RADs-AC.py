import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os

# ==========================================================
# 0. 基础配置 (CPU + Float64)
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cpu")

print(f"🔥 任务: Fourier-Hybrid-PINN for Allen-Cahn | 设备: {device}")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备
# ==========================================================
file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"
have_ground_truth = False

if os.path.exists(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        x_exact = data["x"].flatten()
        t_exact = data["tt"].flatten()
        u_exact_raw = data["uu"]
        
        X, T = np.meshgrid(x_exact, t_exact)
        u_exact_all = u_exact_raw.T
        
        X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
        T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)
        have_ground_truth = True
        print(f"✅ 成功加载数据 (Grid: {u_exact_all.shape})")
    except Exception as e:
        print(f"⚠️ 数据读取出错: {e}")
else:
    print("❌ 错误：找不到 .mat 文件！")
    exit()

# ==========================================================
# 2. 模型定义 (🔥 核心改动：加入傅里叶特征映射)
# ==========================================================
class FourierPINN(nn.Module):
    def __init__(self, hidden=128, num_layers=4, scale=2.0):
        super().__init__()
        # 随机傅里叶特征矩阵 B (冻结，不参与训练)
        self.B = nn.Parameter(torch.randn(2, hidden // 2, dtype=dtype) * scale, requires_grad=False)
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        # 投影到高频空间
        proj = 2.0 * np.pi * torch.matmul(inputs, self.B)
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return self.net(features)

model = FourierPINN(hidden=128, num_layers=5, scale=1.5).to(device)

# ==========================================================
# 3. PDE 残差计算
# ==========================================================
def compute_pde_residual_matrix(model, x, t):
    u = model(x, t)
    u_grads = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dt, u_dx = u_grads[0], u_grads[1]
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    
    # Allen-Cahn: u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0
    f = u_dt - 0.0001 * u_dxx + 5.0 * (u**3 - u)
    return f

# ==========================================================
# 4. 🔥 混合自适应采样 (修复点阵坍缩)
# ==========================================================
def resample_hybrid_points(model, n_total, device):
    model.eval()
    
    # 1. 均匀采样比例提升至 50% (稳住大盘)
    n_uni = int(0.5 * n_total)
    x_uni = (-1 + 2 * torch.rand(n_uni, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_uni = torch.rand(n_uni, 1, device=device, dtype=dtype).requires_grad_(True)
    
    # 2. 自适应采样 50%
    n_adapt = n_total - n_uni
    n_cand = n_adapt * 5 
    x_cand = (-1 + 2 * torch.rand(n_cand, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_cand = torch.rand(n_cand, 1, device=device, dtype=dtype).requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        f_cand = compute_pde_residual_matrix(model, x_cand, t_cand)
        err_cand = torch.abs(f_cand).detach().cpu().numpy().flatten()
    
    # 🔥 降低极化程度，使用 1.2 次方代替平方
    score = err_cand ** 1.2 
    prob = score / (np.sum(score) + 1e-10) 
    
    idx = np.random.choice(n_cand, size=n_adapt, p=prob, replace=False)
    x_adapt = x_cand[idx].detach().requires_grad_(True)
    t_adapt = t_cand[idx].detach().requires_grad_(True)
    
    x_new = torch.cat([x_uni, x_adapt], dim=0)
    t_new = torch.cat([t_uni, t_adapt], dim=0)
    
    model.train()
    return x_new, t_new

# ==========================================================
# 5. 训练准备与循环
# ==========================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 增加采样点密度以适应复杂的高频界面
N_f = 10000 
N_b = 500   
EPOCHS = 6000
RESAMPLE_FREQ = 1000 

x_ic = (-1 + 2 * torch.rand(N_b, 1, device=device, dtype=dtype))
t_ic = torch.zeros_like(x_ic)
u_ic_exact = x_ic**2 * torch.cos(np.pi * x_ic)

t_bc = torch.rand(N_b, 1, device=device, dtype=dtype).requires_grad_(True)
x_bc_left = -1.0 * torch.ones_like(t_bc, requires_grad=True)
x_bc_right = 1.0 * torch.ones_like(t_bc, requires_grad=True)

x_f, t_f = resample_hybrid_points(model, N_f, device)

start_time = time.time()
print(">>> 开始 Fourier-Hybrid-PINN 训练...")

for epoch in range(EPOCHS + 1):
    if epoch > 0 and epoch % RESAMPLE_FREQ == 0:
        x_f, t_f = resample_hybrid_points(model, N_f, device)
        
    optimizer.zero_grad()
    
    res = compute_pde_residual_matrix(model, x_f, t_f)
    loss_pde = torch.mean(res**2)
    
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic_exact)**2)
    
    u_left = model(x_bc_left, t_bc)
    u_right = model(x_bc_right, t_bc)
    loss_bc_u = torch.mean((u_left - u_right)**2)
    
    u_x_left = torch.autograd.grad(u_left, x_bc_left, torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_bc_right, torch.ones_like(u_right), create_graph=True)[0]
    loss_bc_ux = torch.mean((u_x_left - u_x_right)**2)
    
    # 稍微拉高边界的约束力度
    loss = loss_pde + 100.0 * loss_ic + 50.0 * (loss_bc_u + loss_bc_ux)
    
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Ep {epoch:5d} | Loss: {loss.item():.4e} | PDE: {loss_pde.item():.4e}")

# ==========================================================
# 6. L-BFGS 微调
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning...")
x_f, t_f = resample_hybrid_points(model, N_f, device)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=2500, history_size=50, line_search_fn="strong_wolfe")

def closure():
    lbfgs.zero_grad()
    res = compute_pde_residual_matrix(model, x_f, t_f)
    loss_pde = torch.mean(res**2)
    
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic_exact)**2)
    
    u_left = model(x_bc_left, t_bc)
    u_right = model(x_bc_right, t_bc)
    loss_bc_u = torch.mean((u_left - u_right)**2)
    
    u_x_left = torch.autograd.grad(u_left, x_bc_left, torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_bc_right, torch.ones_like(u_right), create_graph=True)[0]
    loss_bc_ux = torch.mean((u_x_left - u_x_right)**2)
    
    loss = loss_pde + 100.0 * loss_ic + 50.0 * (loss_bc_u + loss_bc_ux)
    loss.backward(retain_graph=True)
    return loss

lbfgs.step(closure)
print(f"Final L-BFGS Loss: {closure().item():.5e}")
print(f"Total Time: {time.time()-start_time:.1f}s")

# ==========================================================
# 7. 评估与绘图
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star)
    u_pred_np = u_pred.cpu().numpy().reshape(201, 512)

error_l2 = np.linalg.norm(u_exact_all - u_pred_np) / np.linalg.norm(u_exact_all)
print(f"\n✅✅✅ Fourier-Hybrid-PINN Error: {error_l2:.4e} ✅✅✅")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(u_pred_np.T, interpolation='nearest', cmap='jet', extent=[0, 1, -1, 1], origin='lower', aspect='auto')
plt.colorbar(label='u_pred')
plt.title(f"Fourier-PINN (Error: {error_l2:.2e})")
plt.xlabel("t"); plt.ylabel("x")

plt.subplot(2, 2, 2)
plt.imshow(u_exact_all.T, interpolation='nearest', cmap='jet', extent=[0, 1, -1, 1], origin='lower', aspect='auto')
plt.colorbar(label='u_exact')
plt.title("Ground Truth")
plt.xlabel("t"); plt.ylabel("x")

plt.subplot(2, 2, 3)
err_map = np.abs(u_exact_all - u_pred_np)
plt.imshow(err_map.T, interpolation='nearest', cmap='binary', extent=[0, 1, -1, 1], origin='lower', aspect='auto')
plt.colorbar(label='Abs Error')
plt.title("Error Map")
plt.xlabel("t"); plt.ylabel("x")

plt.subplot(2, 2, 4)
n_uni = int(0.5 * N_f)
plt.scatter(t_f[n_uni:].detach().cpu(), x_f[n_uni:].detach().cpu(), s=1, alpha=0.3, c='red', label='Adaptive')
plt.scatter(t_f[:n_uni].detach().cpu(), x_f[:n_uni].detach().cpu(), s=1, alpha=0.3, c='blue', label='Uniform')
plt.title("Balanced Sampling Distribution")
plt.xlabel("t"); plt.ylabel("x")
plt.xlim(0, 1); plt.ylim(-1, 1)
plt.legend()

plt.tight_layout()
plt.savefig('RAD_AC_Result.png', dpi=300)
print("✅ 最终结果已保存为 RAD_AC_Result.png")