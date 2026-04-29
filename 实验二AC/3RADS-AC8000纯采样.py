import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# ==========================================================
# 0. 基础配置 (CPU Float64 以保证精度)
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print(f"任务: Pure RADS PINN (无子网络版) | Device: CPU")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 物理参数与真值
# ==========================================================
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
        print("✅ 成功加载真实数据")
        return x, t, X, T, usol.T 
    except:
        print("未找到数据文件，使用伪数据占位")
        x = np.linspace(-1, 1, 512)
        t = np.linspace(0, 1, 201)
        X, T = np.meshgrid(x, t)
        return x, t, X, T, np.zeros((201, 512))

x_exact, t_exact, X_mesh, T_mesh, u_exact = get_exact_data()

# ==========================================================
# 2. 网络定义 (Generator 和 主网络)
# ==========================================================

# --- RADS生成器 (对抗采样) ---
class RADS_Generator(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    
    def forward(self, z):
        raw = self.net(z)
        x = torch.tanh(raw[:, 0:1])      # 映射到 [-1, 1]
        t = torch.sigmoid(raw[:, 1:2])   # 映射到 [0, 1]
        return x, t

# --- 周期性嵌入 ---
class PeriodicEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

# --- 主模型 (全局网络 + 硬约束) ---
class PurePINN_Hard(nn.Module):
    def __init__(self, hidden_global=128):                             # 稍微加宽网络补偿没有专家的损失
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
        
        for m in self.global_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        
    def forward(self, x, t):
        x_emb = self.embed(x) 
        features = torch.cat([x_emb, t], dim=1)
        u_net = self.global_net(features)
        
        # Hard IC: 彻底解决初始时刻的误差爆炸
        u_0 = x**2 * torch.cos(np.pi * x)
        u_final = torch.tanh(t) * u_net + u_0
        return u_final

# ==========================================================
# 3. 损失计算与工具函数
# ==========================================================
model = PurePINN_Hard(hidden_global=128).to(device)
generator = RADS_Generator().to(device)

def calculate_pde_loss(x, t):
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    res = u_t - EPSILON * u_xx + GAMMA * (u**3 - u)
    return torch.mean(res**2), res

def get_uniform_points(N):
    x = torch.rand(N, 1, device=device, dtype=dtype) * 2 - 1
    t = torch.rand(N, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3)
optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)

# ==========================================================
# 4. RADS 对抗训练循环
# ==========================================================
TOTAL_ITERS = 20000              
BATCH_SIZE = 8000 
RADS_RATIO = 0.6 

loss_history = [] 
start_time = time.time()

print(f">>> Phase 1: Pure RADS Training (Batch={BATCH_SIZE})...")

for it in range(TOTAL_ITERS + 1):
    
    # --- A. 训练 Generator (最大化残差) ---
    for _ in range(2): 
        optim_gen.zero_grad()
        z = torch.randn(BATCH_SIZE // 2, 2, device=device, dtype=dtype)
        x_gen, t_gen = generator(z)
        loss_val, _ = calculate_pde_loss(x_gen, t_gen)
        loss_gen = -loss_val 
        loss_gen.backward()
        optim_gen.step()

    # --- B. 训练 PINN (最小化残差) ---
    optim_pinn.zero_grad()
    
    # 1. RADS 采样 (困难样本)
    N_aais = int(BATCH_SIZE * RADS_RATIO)
    with torch.no_grad():
        z = torch.randn(N_aais, 2, device=device, dtype=dtype)
        x_adv, t_adv = generator(z)
    x_adv.requires_grad_(True)
    t_adv.requires_grad_(True)
    
    # 2. 均匀采样 (全局保底)
    N_uni = BATCH_SIZE - N_aais
    x_uni, t_uni = get_uniform_points(N_uni)
    
    # 3. 组合
    x_train = torch.cat([x_adv, x_uni], dim=0)
    t_train = torch.cat([t_adv, t_uni], dim=0)
    
    # 4. 计算残差
    _, res = calculate_pde_loss(x_train, t_train)
    res_sq = res**2
    
    # 5. 加权 Loss (让网络更关注 RADS 找出的高误差区)
    weights = torch.ones_like(res_sq)
    weights[:N_aais] = 5.0 
    loss = torch.mean(res_sq * weights)
    
    loss.backward()
    optim_pinn.step()
    
    loss_history.append(loss.item())

    # 学习率衰减
    if it in [10000, 15000]:
        for param_group in optim_pinn.param_groups:
            param_group['lr'] *= 0.5

    if it % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Iter {it:5d} | PINN Loss: {loss.item():.6e} | Time: {elapsed:.1f}s")

# ==========================================================
# 5. L-BFGS 终极微调
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning...")
x_full, t_full = get_uniform_points(N=10000) 

lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0, 
    max_iter=5000, 
    max_eval=6000,
    history_size=100, 
    tolerance_grad=1e-11, 
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    loss_pde, _ = calculate_pde_loss(x_full, t_full)
    loss = loss_pde
    loss.backward()
    return loss

lbfgs.step(closure)
print(f"✅ L-BFGS Final Loss: {closure().item():.6e}")

# ==========================================================
# 6. 绘图与验证
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
    print(f"\n✨✨✨ Relative L2 Error: {error_u:.4e} ✨✨✨")

fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# 1. 预测解
im1 = axs[0].contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
axs[0].set_title(f"Predicted u(t,x) | Err: {error_u:.2e}")
plt.colorbar(im1, ax=axs[0])

# 2. 绝对误差 (Log Scale)
if np.max(np.abs(u_exact)) > 0:
    err_map = np.abs(u_exact - u_pred) + 1e-12
    im2 = axs[1].contourf(T_mesh, X_mesh, np.log10(err_map), 100, cmap='inferno')
    axs[1].set_title("Log10 Absolute Error")
    plt.colorbar(im2, ax=axs[1])

# 3. AAIS 点分布
axs[2].set_title("RADS Adversarial Sampling")
axs[2].set_xlim(0, 1); axs[2].set_ylim(-1, 1)
with torch.no_grad():
    z_vis = torch.randn(3000, 2, device=device, dtype=dtype)
    x_vis, t_vis = generator(z_vis)
    axs[2].scatter(t_vis.cpu(), x_vis.cpu(), s=2, c='blue', alpha=0.5, label='AAIS Points')
axs[2].legend()

# 4. 截面
t_slice = 0.5
idx = int(t_slice * 200) 
axs[3].plot(x_exact, u_pred[idx, :], 'r--', linewidth=2, label='Prediction')
if np.max(np.abs(u_exact)) > 0:
    axs[3].plot(x_exact, u_exact[idx, :], 'k-', linewidth=1.0, alpha=0.7, label='Exact')
axs[3].set_title(f"Slice at t={t_slice}")
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("RADS_PINN.png", dpi=150)
print("✅ Done. 图像已保存为 RADS_PINN.png")