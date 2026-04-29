import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# ==========================================================
# 0. 基础配置 (强制 CPU 以使用 float64)
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 强制使用 CPU
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print(f" 任务: AB-PINN Hard-Constraint AC (Fixing the Turn)")
print(f" 注意: 已强制使用 CPU (Float64)")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 物理参数与真值
# ==========================================================
# Allen-Cahn Eq: u_t - 0.0001*u_xx + 5*(u^3 - u) = 0
EPSILON = 0.0001
GAMMA = 5.0 

def get_exact_data():
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"
    try:
        data = scipy.io.loadmat(file_path)
        x = data["x"].flatten()
        t = data["tt"].flatten()
        usol = data["uu"]
        X, T = np.meshgrid(x, t)
        return x, t, X, T, usol.T 
    except:
        print(" 未找到数据文件，使用伪数据占位")
        x = np.linspace(-1, 1, 512)
        t = np.linspace(0, 1, 201)
        X, T = np.meshgrid(x, t)
        return x, t, X, T, np.zeros((201, 512))

x_exact, t_exact, X_mesh, T_mesh, u_exact = get_exact_data()

# ==========================================================
# 2. 网络定义
# ==========================================================

class PeriodicEmbedding(nn.Module):
    """  x -> [cos(pi*x), sin(pi*x)] """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class WindowFunction(nn.Module):
    def __init__(self, input_dim=3, center_init=None, radius_init=0.2):
        super().__init__()
        if center_init is None: center_init = torch.zeros(input_dim)
        
        self.mu = nn.Parameter(center_init.clone().detach().reshape(1, -1))
        
        scale_val = 1.0 / radius_init
        self.L_diag = nn.Parameter(torch.ones(input_dim) * scale_val)
        self.L_tril = nn.Parameter(torch.zeros(input_dim * (input_dim - 1) // 2))
        
    def get_L_matrix(self):
        dim = self.mu.shape[1]
        L = torch.zeros(dim, dim, device=self.mu.device, dtype=self.mu.dtype)
        idx = torch.arange(dim)
        L[idx, idx] = torch.abs(self.L_diag) + 1e-5
        if dim > 1:
            indices = torch.tril_indices(dim, dim, offset=-1)
            L[indices[0], indices[1]] = self.L_tril
        return L

    def forward(self, x):
        L = self.get_L_matrix()
        diff = x - self.mu 
        x_trans = torch.matmul(diff, L) 
        phi = torch.exp(-0.5 * torch.sum(x_trans**2, dim=1, keepdim=True))
        return phi, x_trans

class LocalExpert(nn.Module):
    def __init__(self, input_dim=3, hidden=20, center=None, radius=0.2):
        super().__init__()
        self.window = WindowFunction(input_dim, center, radius)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x_embedded):
        phi, x_trans = self.window(x_embedded)
        u_local = self.net(x_trans)
        return phi * u_local, phi

class ABPINN_Hard(nn.Module):
    def __init__(self, hidden_global=40):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.input_dim = 3 
        
        self.global_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, 1)
        )
        self.experts = nn.ModuleList([])
        
    def add_expert(self, center_vec):
        center_vec = center_vec.to(dtype=dtype, device=device)
        # 半径设为 0.4，兼顾局部覆盖和修补
        expert = LocalExpert(self.input_dim, hidden=20, center=center_vec, radius=0.4).to(device)
        self.experts.append(expert)
        
    def forward(self, x, t):
        x_emb = self.embed(x) 
        features = torch.cat([x_emb, t], dim=1)
        u_net = self.global_net(features)
        
        for expert in self.experts:
            val, phi = expert(features)
            u_net = u_net + val
        
        # Hard IC
        u_0 = x**2 * torch.cos(np.pi * x)
        u_final = torch.tanh(t) * u_net + u_0
        return u_final

# ==========================================================
# 3. 训练核心
# ==========================================================
model = ABPINN_Hard().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def get_residual_points(N=2000):
    x = torch.rand(N, 1, device=device, dtype=dtype) * 2 - 1
    t = torch.rand(N, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t

def calculate_pde_loss(x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    res = u_t - EPSILON * u_xx + GAMMA * (u**3 - u)
    return torch.mean(res**2), res

# ==========================================================
# 4. 训练循环 (Adam + L-BFGS)
# ==========================================================
MAX_EXPERTS = 8          
ITER_PER_ADD = 4000      
ADAM_ITERS = 32000      
FREEZE_ITER = 28000      

loss_history = []
start_time = time.time()

print(f">>> Phase 1: Adam Training...")

for it in range(ADAM_ITERS + 1):
    x_f, t_f = get_residual_points(N=2000)
    
    optimizer.zero_grad()
    loss_pde, _ = calculate_pde_loss(x_f, t_f)
    loss = loss_pde 
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_history.append(loss.item())

    # 动态添加 Expert
    if it > 0 and it % ITER_PER_ADD == 0 and len(model.experts) < MAX_EXPERTS:
        # 全局扫描
        x_scan = torch.linspace(-1, 1, 100, device=device, dtype=dtype).view(-1,1)
        t_scan = torch.linspace(0, 1, 50, device=device, dtype=dtype).view(-1,1)
        X_s, T_s = torch.meshgrid(x_scan.squeeze(), t_scan.squeeze(), indexing='ij')
        
        x_flat = X_s.reshape(-1, 1).clone().detach().requires_grad_(True)
        t_flat = T_s.reshape(-1, 1).clone().detach().requires_grad_(True)
        
        _, res_val = calculate_pde_loss(x_flat, t_flat)
        res_abs = torch.abs(res_val).detach()
        
        mask = (t_flat > 0.05).squeeze()
        
        if mask.sum() > 0:
            
            masked_res = res_abs.squeeze()[mask]
            max_val_masked = torch.max(masked_res)
           
            res_abs_search = res_abs.clone()
            res_abs_search[~mask.reshape(res_abs.shape)] = -1.0
            
            max_val, idx = torch.max(res_abs_search, 0) # 展平索引
            best_x = x_flat[idx]
            best_t = t_flat[idx]
            
            print(f" Iter {it}: Max Residual {max_val.item():.4f} found at (x={best_x.item():.2f}, t={best_t.item():.2f})")
            
            with torch.no_grad():
                x_emb_c = model.embed(best_x.unsqueeze(0))
                center_new = torch.cat([x_emb_c, best_t.unsqueeze(0)], dim=1)
            
            model.add_expert(center_new[0])
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        else:
            print("    No valid region found for adding expert.")

    # 冻结窗口
    if it == FREEZE_ITER:
        print("Freezing Window Parameters...")
        for expert in model.experts:
            expert.window.mu.requires_grad = False
            expert.window.L_diag.requires_grad = False
            expert.window.L_tril.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    if it % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Iter {it:5d} | Loss: {loss.item():.6e} | Experts: {len(model.experts)} | Time: {elapsed:.1f}s")

# ==========================================================
# 5. L-BFGS 微调
# ==========================================================
print("\n>>>  Phase 2: L-BFGS Fine-tuning...")
x_full, t_full = get_residual_points(N=8000) 

lbfgs = torch.optim.LBFGS(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1.0, 
    max_iter=2500, # 增加迭代次数
    max_eval=3000,
    history_size=50,
    tolerance_grad=1e-9,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    loss_pde, _ = calculate_pde_loss(x_full, t_full)
    loss = loss_pde
    loss.backward()
    return loss

lbfgs.step(closure)
final_loss = closure().item()
print(f" L-BFGS Final Loss: {final_loss:.6e}")

# ==========================================================
# 6. 绘图与可视化
# ==========================================================
model.eval()
print(">>> 绘图验证中...")

X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

with torch.no_grad():
    u_pred_flat = model(X_star, T_star)
    u_pred = u_pred_flat.cpu().numpy().reshape(X_mesh.shape)

if np.max(np.abs(u_exact)) > 0:
    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    print(f"✨ Relative L2 Error: {error_u:.2%}")

fig, axs = plt.subplots(1, 4, figsize=(24, 5))

# 1. 预测解
im1 = axs[0].contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
axs[0].set_title("Predicted u(t,x)")
axs[0].set_xlabel("t"); axs[0].set_ylabel("x")
plt.colorbar(im1, ax=axs[0])

# 2. 绝对误差
if np.max(np.abs(u_exact)) > 0:
    im2 = axs[1].contourf(T_mesh, X_mesh, np.abs(u_exact - u_pred), 100, cmap='binary')
    axs[1].set_title("Absolute Error")
    axs[1].set_xlabel("t"); axs[1].set_ylabel("x")
    plt.colorbar(im2, ax=axs[1])

# 3. 子域分布 (重点关注这里)
axs[2].set_title("Subdomain Distribution (New Strategy)")
axs[2].set_xlabel("t"); axs[2].set_ylabel("x")
axs[2].set_xlim(0, 1); axs[2].set_ylim(-1, 1)

# 背景放误差图，这样能看到子域是不是生成在误差大的地方
if np.max(np.abs(u_exact)) > 0:
    axs[2].contourf(T_mesh, X_mesh, np.abs(u_exact - u_pred), 20, cmap='Greys', alpha=0.4)

with torch.no_grad():
    for i, expert in enumerate(model.experts):
        phi_flat, _ = expert.window(torch.cat([model.embed(X_star), T_star], dim=1))
        phi_map = phi_flat.cpu().numpy().reshape(X_mesh.shape)
        
        color = plt.cm.tab10(i % 10)
        # 画轮廓
        axs[2].contour(T_mesh, X_mesh, phi_map, levels=[0.5], colors=[color], linewidths=2)
        
        # 标记中心
        max_idx = np.argmax(phi_map)
        center_t = T_mesh.flatten()[max_idx]
        center_x = X_mesh.flatten()[max_idx]
        axs[2].scatter(center_t, center_x, marker='x', s=120, color=color, linewidth=3, label=f'Exp {i+1}')

axs[2].legend(loc='upper right', fontsize='small', framealpha=0.9)

# 4. 截面比较 (t=0.5)
t_slice = 0.5
idx = int(t_slice * 200) 
axs[3].plot(x_exact, u_pred[idx, :], 'r--', linewidth=2.5, label='Prediction')
if np.max(np.abs(u_exact)) > 0:
    axs[3].plot(x_exact, u_exact[idx, :], 'k-', linewidth=1.5, alpha=0.7, label='Exact')
axs[3].set_title(f"Slice at t={t_slice}")
axs[3].legend()
axs[3].set_ylim([-1.1, 1.1])
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ABPINN_Corrected_Expert.png", dpi=150)
print(" Done.")
plt.show()