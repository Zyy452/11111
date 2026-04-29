import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time

# ==========================================================
# 0. 配置与目录
# ==========================================================
BASE_SAVE_DIR = "/3241003007/zy/save"
EXP_DIR = os.path.join(BASE_SAVE_DIR, "AC_Experiment")
os.makedirs(EXP_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 参数与真值
# ==========================================================
EPSILON, GAMMA = 0.0001, 5.0 
data = scipy.io.loadmat("/3241003007/zy/实验二AC/AC.mat")
u_exact_all = data["uu"].T # (201, 512)
x_exact, t_exact = data["x"].flatten(), data["tt"].flatten()
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 网络定义 (AB-PINN 相关)
# ==========================================================
class PeriodicEmbedding(nn.Module):
    def forward(self, x): return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class WindowFunction(nn.Module):
    def __init__(self, input_dim=3, center_init=None, radius_init=0.2):
        super().__init__()
        if center_init is None: center_init = torch.zeros(input_dim)
        self.mu = nn.Parameter(center_init.clone().detach().reshape(1, -1))
        self.L_diag = nn.Parameter(torch.ones(input_dim) * (1.0 / radius_init))
        self.L_tril = nn.Parameter(torch.zeros(input_dim * (input_dim - 1) // 2))
    def get_L_matrix(self):
        L = torch.zeros(self.mu.shape[1], self.mu.shape[1], device=self.mu.device, dtype=self.mu.dtype)
        idx = torch.arange(self.mu.shape[1], device=self.mu.device)
        L[idx, idx] = torch.abs(self.L_diag) + 1e-5
        indices = torch.tril_indices(self.mu.shape[1], self.mu.shape[1], offset=-1, device=self.mu.device)
        L[indices[0], indices[1]] = self.L_tril
        return L
    def forward(self, x):
        phi = torch.exp(-0.5 * torch.sum(torch.matmul(x - self.mu, self.get_L_matrix())**2, dim=1, keepdim=True))
        return phi, x

class LocalExpert(nn.Module):
    def __init__(self, input_dim=3, hidden=20, center=None, radius=0.2):
        super().__init__()
        self.window = WindowFunction(input_dim, center, radius)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for m in self.net.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)
    def forward(self, x_emb):
        phi, x_t = self.window(x_emb)
        return phi * self.net(x_t)

class ABPINN_Hard(nn.Module):
    def __init__(self, hidden_global=40):
        super().__init__()
        self.embed, self.input_dim = PeriodicEmbedding(), 3
        self.global_net = nn.Sequential(nn.Linear(3, hidden_global), nn.Tanh(), nn.Linear(hidden_global, hidden_global), nn.Tanh(), nn.Linear(hidden_global, 1))
        self.experts = nn.ModuleList([])
    def add_expert(self, c): self.experts.append(LocalExpert(3, 20, center=c.to(device), radius=0.4).to(device))
    def forward(self, x, t):
        x_emb = self.embed(x) 
        features = torch.cat([x_emb, t], dim=1)
        u_net = self.global_net(features)
        for expert in self.experts: u_net = u_net + expert(features)
        return torch.tanh(t) * u_net + (x**2 * torch.cos(np.pi * x))

# ==========================================================
# 3. 损失计算
# ==========================================================
model = ABPINN_Hard().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def calculate_pde_loss(x, t):
    u = model(x, t)
    u_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    res = u_dt - EPSILON * u_dxx + GAMMA * (u**3 - u)
    return torch.mean(res**2)

# ==========================================================
# 4. 训练循环 (Phase 1: Adam)
# ==========================================================
MAX_EXP, ITER_ADD, ADAM_ITERS = 8, 4000, 32000      
loss_history = []
err_history = []
iters = []
start_time = time.time()

print(f"\n>>> [Data Prep] AB-PINN Training (tracking L2 history)...")
for it in range(ADAM_ITERS + 1):
    # 【修复】：斩断计算图并开启求导
    x_f = (torch.rand(2000, 1, device=device, dtype=dtype) * 2 - 1).detach().requires_grad_(True)
    t_f = torch.rand(2000, 1, device=device, dtype=dtype).detach().requires_grad_(True)
    
    optimizer.zero_grad()
    loss = calculate_pde_loss(x_f, t_f)
    loss.backward()
    optimizer.step()
    
    # 动态添加 Expert 逻辑保持不变
    if it > 0 and it % ITER_ADD == 0 and len(model.experts) < MAX_EXP:
        x_s = torch.linspace(-1, 1, 100, device=device, dtype=dtype).view(-1,1)
        t_s = torch.linspace(0, 1, 50, device=device, dtype=dtype).view(-1,1)
        X_s, T_s = torch.meshgrid(x_s.squeeze(), t_s.squeeze(), indexing='ij')
        x_flat, t_flat = X_s.reshape(-1,1).clone().detach().requires_grad_(True), T_s.reshape(-1,1).clone().detach().requires_grad_(True)
        u_temp = model(x_flat, t_flat)
        u_t_temp = torch.autograd.grad(u_temp, t_flat, torch.ones_like(u_temp), create_graph=True)[0]
        u_x_temp = torch.autograd.grad(u_temp, x_flat, torch.ones_like(u_temp), create_graph=True)[0]
        u_xx_temp = torch.autograd.grad(u_x_temp, x_flat, torch.ones_like(u_x_temp), create_graph=True)[0]
        res_v = u_t_temp - EPSILON * u_xx_temp + GAMMA * (u_temp**3 - u_temp)
        res_abs = torch.abs(res_v).detach(); mask = (t_flat > 0.05)
        if mask.sum() > 0:
            res_abs_search = res_abs.clone(); res_abs_search[~mask] = -1.0
            idx = torch.argmax(res_abs_search, 0); best_x, best_t = x_flat[idx], t_flat[idx]
            model.add_expert(torch.cat([model.embed(best_x.unsqueeze(0)), best_t.unsqueeze(0)], dim=1)[0])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if it % 1000 == 0:
        model.eval()
        with torch.no_grad():
            u_p = model(X_star, T_star).cpu().numpy().reshape(u_exact_all.shape)
            c_e = np.linalg.norm(u_exact_all - u_p) / np.linalg.norm(u_exact_all)
        model.train()
        loss_history.append(loss.item())
        err_history.append(c_e)
        iters.append(it)
        print(f"Iter {it:5d} | Loss: {loss.item():.6e} | Rel L2: {c_e:.4f} | Experts: {len(model.experts)}")

# ==========================================================
# Phase 2: L-BFGS
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning...")
# 【修复】：必须使用 detach 斩断隐式计算图，并且必须加上 requires_grad_(True) ！！！
x_full = (torch.rand(8000, 1, device=device, dtype=dtype) * 2 - 1).detach().requires_grad_(True)
t_full = torch.rand(8000, 1, device=device, dtype=dtype).detach().requires_grad_(True)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=2500, line_search_fn="strong_wolfe")
def closure(): 
    lbfgs.zero_grad()
    loss = calculate_pde_loss(x_full, t_full)
    loss.backward()
    return loss
lbfgs.step(closure)

# ==========================================================
# 6. 数据提取与保存 (保存为 PT 字典)
# ==========================================================
model.eval()
with torch.no_grad():
    # 【修复】：统一格式为 u_exact_all.shape
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact_all.shape)
    error_u = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)

# 保存 pt 字典
save_data = {
    "u_pred": u_pred, 
    "u_exact_all": u_exact_all, 
    "loss_history": np.array(loss_history),
    "err_history": np.array(err_history), 
    "iters": np.array(iters),
    "final_error": error_u,
}
DATA_SAVE_PATH = os.path.join(EXP_DIR, "ac_abpinn_results.pt")
torch.save(save_data, DATA_SAVE_PATH)
print(f"\n✨✨ AB-PINN Final Rel L2 Error: {error_u:.4e} | 保存至 {DATA_SAVE_PATH}")