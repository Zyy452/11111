import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time

# ==========================================================
# 0. 配置与目录 (统一保存在新建文件夹)
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
x_exact, t_exact, usol = data["x"].flatten(), data["tt"].flatten(), data["uu"]
X, T = np.meshgrid(x_exact, t_exact)
u_exact_all = usol.T 
X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 网络定义 (Generator, Embedding, Hard-PINN)
# ==========================================================
class RADS_Generator(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)
        )
    def forward(self, z):
        raw = self.net(z)
        return torch.tanh(raw[:, 0:1]), torch.sigmoid(raw[:, 1:2])

class PeriodicEmbedding(nn.Module):
    def forward(self, x):
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class PurePINN_Hard(nn.Module):
    def __init__(self, hidden_global=128):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.global_net = nn.Sequential(
            nn.Linear(3, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, hidden_global), nn.Tanh(),
            nn.Linear(hidden_global, hidden_global), nn.Tanh(), nn.Linear(hidden_global, 1)
        )
        for m in self.global_net.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)
    def forward(self, x, t):
        u_net = self.global_net(torch.cat([self.embed(x), t], dim=1))
        return torch.tanh(t) * u_net + (x**2 * torch.cos(np.pi * x))

# ==========================================================
# 3. 损失计算
# ==========================================================
model = PurePINN_Hard(hidden_global=128).to(device)
generator = RADS_Generator().to(device)
def calculate_pde_loss(x, t):
    u = model(x, t)
    u_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    res = u_dt - EPSILON * u_dxx + GAMMA * (u**3 - u)
    return torch.mean(res**2)

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3)
optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)

# ==========================================================
# 4. RADS 训练 (Phase 1: Adam)
# ==========================================================
TOTAL_ITERS, BATCH_SIZE, RADS_RATIO = 20000, 8000, 0.6 
loss_history = [] 
err_history = [] 
iters = [] 

print(f"\n>>> [Data Prep] RAD-PINN Training (tracking L2 history)...")
for it in range(TOTAL_ITERS + 1):
    # A. 训练 Generator
    for _ in range(2): 
        optim_gen.zero_grad()
        x_gen, t_gen = generator(torch.randn(BATCH_SIZE // 2, 2, device=device, dtype=dtype))
        loss_gen = -calculate_pde_loss(x_gen, t_gen)
        loss_gen.backward(); optim_gen.step()
        
    # B. 训练 PINN
    optim_pinn.zero_grad()
    N_aais = int(BATCH_SIZE * RADS_RATIO)
    with torch.no_grad(): 
        x_adv, t_adv = generator(torch.randn(N_aais, 2, device=device, dtype=dtype))
    
    # 【修复】：安全起见，从 generator 出来的特征也 detach 切断一切潜在联系
    x_adv = x_adv.detach().requires_grad_(True)
    t_adv = t_adv.detach().requires_grad_(True)
    
    # 【核心修复】：加上 detach() 彻底斩断乘减法运算的隐式图！
    x_uni = (torch.rand(BATCH_SIZE - N_aais, 1, device=device, dtype=dtype) * 2 - 1).detach().requires_grad_(True)
    t_uni = torch.rand(BATCH_SIZE - N_aais, 1, device=device, dtype=dtype).detach().requires_grad_(True)
    
    x_in, t_in = torch.cat([x_adv, x_uni], dim=0), torch.cat([t_adv, t_uni], dim=0)
    u_temp = model(x_in, t_in)
    u_t_temp = torch.autograd.grad(u_temp, t_in, torch.ones_like(u_temp), create_graph=True)[0]
    u_x_temp = torch.autograd.grad(u_temp, x_in, torch.ones_like(u_temp), create_graph=True)[0]
    u_xx_temp = torch.autograd.grad(u_x_temp, x_in, torch.ones_like(u_x_temp), create_graph=True)[0]
    
    res_sq = (u_t_temp - EPSILON * u_xx_temp + GAMMA * (u_temp**3 - u_temp))**2
    w = torch.ones_like(res_sq)
    w[:N_aais] = 5.0 
    loss = torch.mean(res_sq * w)
    loss.backward()
    optim_pinn.step()
    
    # 在 Adam 循环中每隔 1000 步计算误差
    if it % 1000 == 0:
        model.eval()
        with torch.no_grad():
            u_p = model(X_star, T_star).cpu().numpy().reshape(u_exact_all.shape)
            c_e = np.linalg.norm(u_exact_all - u_p) / np.linalg.norm(u_exact_all)
        model.train()
        loss_history.append(loss.item())
        err_history.append(c_e)
        iters.append(it)
        print(f"Iter {it:5d} | Loss: {loss.item():.2e} | Rel L2: {c_e:.4f}")

# ==========================================================
# Phase 2: L-BFGS
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning...")
# 【核心修复】：同样加上 detach() 
x_f_full = (torch.rand(10000, 1, device=device, dtype=dtype) * 2 - 1).detach().requires_grad_(True)
t_f_full = torch.rand(10000, 1, device=device, dtype=dtype).detach().requires_grad_(True)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    loss = calculate_pde_loss(x_f_full, t_f_full)
    loss.backward()
    return loss
lbfgs.step(closure)

# ==========================================================
# 6. 数据提取与保存 
# ==========================================================
model.eval()
generator.eval()
with torch.no_grad():
    # 【修复】：把原来的 X_mesh.shape 改成了正确的 u_exact_all.shape
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact_all.shape)
    error_u = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)

save_data = {
    "u_pred": u_pred, 
    "u_exact_all": u_exact_all, 
    "loss_history": np.array(loss_history),
    "err_history": np.array(err_history),
    "iters": np.array(iters),
    "final_error": error_u,
}
DATA_SAVE_PATH = os.path.join(EXP_DIR, "ac_rads_pinn_results.pt")
torch.save(save_data, DATA_SAVE_PATH)
print(f"\n✨✨ RAD-PINN Final Rel L2 Error: {error_u:.4e} | 保存至 {DATA_SAVE_PATH}")