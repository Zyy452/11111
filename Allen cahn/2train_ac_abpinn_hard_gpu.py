import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time

# ==========================================================
# 0. 基础配置与目录设置
# ==========================================================
save_dir = "/3241003007/zy/save"
os.makedirs(save_dir, exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 自动检测 GPU (CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}，正在使用 GPU 训练！")
else:
    device = torch.device("cpu")
    print(f"⚠️ 未检测到 GPU，已回退至 CPU 训练。")

dtype = torch.float64
torch.set_default_dtype(dtype)

print(f"🔥 任务: AB-PINN Hard-Constraint AC | 设备: {device} | 精度: Float64")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 物理参数与真值读取
# ==========================================================
EPSILON = 0.0001
GAMMA = 5.0 

def get_exact_data():
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"
    if not os.path.exists(file_path): 
        file_path = "/3241003007/zy/实验二AC/AC.mat"
        
    try:
        data = scipy.io.loadmat(file_path)
        x = data["x"].flatten()
        t = data["tt"].flatten()
        usol = data["uu"]
        X, T = np.meshgrid(x, t)
        print("✅ 成功加载 AC.mat 真值数据")
        return x, t, X, T, usol.T 
    except:
        print("❌ 未找到数据文件，使用伪数据占位")
        x = np.linspace(-1, 1, 512)
        t = np.linspace(0, 1, 201)
        X, T = np.meshgrid(x, t)
        return x, t, X, T, np.zeros((201, 512))

x_exact, t_exact, X_mesh, T_mesh, u_exact = get_exact_data()

# ==========================================================
# 2. 网络定义 (AB-PINN 架构 - 已修复 GPU 设备对齐)
# ==========================================================
class PeriodicEmbedding(nn.Module):
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
        
        # [GPU 修复] 强制索引张量生成在与 mu 相同的设备上
        idx = torch.arange(dim, device=self.mu.device)
        L[idx, idx] = torch.abs(self.L_diag) + 1e-5
        
        if dim > 1:
            # [GPU 修复] 强制下三角索引生成在与 mu 相同的设备上
            indices = torch.tril_indices(dim, dim, offset=-1, device=self.mu.device)
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
        # 确保中心向量与网络挂载在同一设备
        center_vec = center_vec.to(dtype=dtype, device=device)
        expert = LocalExpert(self.input_dim, hidden=20, center=center_vec, radius=0.4).to(device)
        self.experts.append(expert)
        
    def forward(self, x, t):
        x_emb = self.embed(x) 
        features = torch.cat([x_emb, t], dim=1)
        u_net = self.global_net(features)
        
        for expert in self.experts:
            val, phi = expert(features)
            u_net = u_net + val
        
        u_0 = x**2 * torch.cos(np.pi * x)
        u_final = torch.tanh(t) * u_net + u_0
        return u_final

# ==========================================================
# 3. 训练函数定义
# ==========================================================
# 将整个模型挂载至设备 (GPU)
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
# 4. 训练循环 (Phase 1: Adam)
# ==========================================================
MAX_EXPERTS = 8          
ITER_PER_ADD = 4000      
ADAM_ITERS = 32000      
FREEZE_ITER = 28000      

loss_history = []
start_time = time.time()

print(f">>> Phase 1: Adam Training with Dynamic Experts...")

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
        x_scan = torch.linspace(-1, 1, 100, device=device, dtype=dtype).view(-1,1)
        t_scan = torch.linspace(0, 1, 50, device=device, dtype=dtype).view(-1,1)
        X_s, T_s = torch.meshgrid(x_scan.squeeze(), t_scan.squeeze(), indexing='ij')
        
        x_flat = X_s.reshape(-1, 1).clone().detach().requires_grad_(True)
        t_flat = T_s.reshape(-1, 1).clone().detach().requires_grad_(True)
        
        _, res_val = calculate_pde_loss(x_flat, t_flat)
        res_abs = torch.abs(res_val).detach()
        mask = (t_flat > 0.05).squeeze()
        
        if mask.sum() > 0:
            res_abs_search = res_abs.clone()
            res_abs_search[~mask.reshape(res_abs.shape)] = -1.0
            
            max_val, idx = torch.max(res_abs_search, 0) 
            best_x, best_t = x_flat[idx], t_flat[idx]
            
            print(f"⚡️ Iter {it}: Max Residual {max_val.item():.4f} found at (x={best_x.item():.2f}, t={best_t.item():.2f})")
            
            with torch.no_grad():
                x_emb_c = model.embed(best_x.unsqueeze(0))
                center_new = torch.cat([x_emb_c, best_t.unsqueeze(0)], dim=1)
            
            model.add_expert(center_new[0])
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        else:
            print("    No valid region found for adding expert.")

    # 冻结窗口参数
    if it == FREEZE_ITER:
        print("❄️ Freezing Window Parameters...")
        for expert in model.experts:
            expert.window.mu.requires_grad = False
            expert.window.L_diag.requires_grad = False
            expert.window.L_tril.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    if it % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Iter {it:5d} | Loss: {loss.item():.6e} | Experts: {len(model.experts)} | Time: {elapsed:.1f}s")

# ==========================================================
# 5. Phase 2: L-BFGS 微调
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning...")
x_full, t_full = get_residual_points(N=8000) 

lbfgs = torch.optim.LBFGS(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1.0, max_iter=2500, max_eval=3000,
    history_size=50, tolerance_grad=1e-9,
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
loss_history.append(final_loss)
total_train_time = time.time() - start_time
print(f"L-BFGS Final Loss: {final_loss:.6e} | Total Time: {total_train_time:.1f}s")

# ==========================================================
# 6. 数据提取与保存
# ==========================================================
model.eval()

# 将验证坐标放入当前设备 (GPU)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

with torch.no_grad():
    u_pred_flat = model(X_star, T_star)
    u_pred = u_pred_flat.cpu().numpy().reshape(X_mesh.shape)
    
    expert_phi_maps = []
    expert_centers_t = []
    expert_centers_x = []
    
    for i, expert in enumerate(model.experts):
        phi_flat, _ = expert.window(torch.cat([model.embed(X_star), T_star], dim=1))
        phi_map = phi_flat.cpu().numpy().reshape(X_mesh.shape)
        expert_phi_maps.append(phi_map)
        
        max_idx = np.argmax(phi_map)
        center_t = T_mesh.flatten()[max_idx]
        center_x = X_mesh.flatten()[max_idx]
        expert_centers_t.append(center_t)
        expert_centers_x.append(center_x)

error_u = 0.0
if np.max(np.abs(u_exact)) > 0:
    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    print(f"\n✨✨✨ AB-PINN Final Relative L2 Error: {error_u:.4e} ✨✨✨")

model_save_path = os.path.join(save_dir, "ac_abpinn_hard_model.pth")
torch.save(model.state_dict(), model_save_path)

data_save_path = os.path.join(save_dir, "ac_abpinn_hard_results.npz")
np.savez(data_save_path, 
         u_pred=u_pred, 
         u_exact=u_exact, 
         X_mesh=X_mesh, 
         T_mesh=T_mesh,
         x_exact=x_exact,
         t_exact=t_exact,
         phi_maps=np.array(expert_phi_maps),
         centers_t=np.array(expert_centers_t),
         centers_x=np.array(expert_centers_x),
         loss_history=np.array(loss_history),
         train_time=total_train_time,
         error_u=error_u)

print(f"数据及模型已统一保存至 {save_dir} 目录下")