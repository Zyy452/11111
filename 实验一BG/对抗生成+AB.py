import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os

# ==========================================================
# 0. 基础配置 (CPU + Float64 保障极限精度)
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print(f"🚀 冲刺 e-04: 位移场生成器 (Displacement Field) + RFF AB-PINN | Device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ==========================================================
# 1. 数据读取 (Burgers)
# ==========================================================
try:
    file_path = "/3241003007/zy/实验一BG/burgers_shock.mat"
    if not os.path.exists(file_path): file_path = "burgers_shock.mat"
    data = loadmat(file_path)
    x_exact = data["x"].flatten()
    t_exact = data["t"].flatten()
    u_exact_all = data["usol"]
except:
    print("❌ 错误：找不到 burgers_shock.mat 文件！请检查路径。")
    exit()

X_star = torch.tensor(np.meshgrid(x_exact, t_exact)[0].flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(np.meshgrid(x_exact, t_exact)[1].flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 网络定义 (RFF AB-PINN + Displacement Generator)
# ==========================================================

# --- A. 位移场生成器 (Displacement Generator) ---
class DisplacementGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    
    def forward(self, z):
        # 🌟 核心修改：z 是均匀分布的输入点，网络输出一个有限的位移 (delta)
        delta = self.net(z) * 0.3  # 限制最大位移距离 (0.3)，防止点乱飞
        # 在原坐标基础上加上偏移量，并限制在定义域内
        x_new = torch.clamp(z[:, 0:1] + delta[:, 0:1], -1.0, 1.0)
        t_new = torch.clamp(z[:, 1:2] + delta[:, 1:2], 0.0, 1.0)
        return x_new, t_new

# --- B. 支持 RFF 的 AB-PINN 子网络 ---
class DeepSubNet(nn.Module):
    def __init__(self, hidden=64, is_new_addition=False, use_rff=False):
        super().__init__()
        self.use_rff = use_rff
        
        # 引入随机傅里叶特征 (RFF) 处理高频激波
        if use_rff:
            self.B = nn.Parameter(torch.randn(2, hidden//2, dtype=dtype) * 3.0, requires_grad=False)
            in_dim = hidden
        else:
            in_dim = 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Zero Init 策略：确保在添加新网络时，对全局解的初始扰动为 0
        if is_new_addition:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        if self.use_rff:
            proj = 2.0 * np.pi * torch.matmul(xt, self.B)
            xt = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return self.net(xt)

class DynamicABPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        # 初始的全局网不使用 RFF，保持低频平滑拟合
        self.subnets = nn.ModuleList([DeepSubNet(hidden, use_rff=False), DeepSubNet(hidden, use_rff=False)])
        self.centers = nn.ParameterList([
            nn.Parameter(torch.tensor([-0.5, 0.25], dtype=dtype)),
            nn.Parameter(torch.tensor([0.5, 0.75], dtype=dtype))
        ])
        self.log_gammas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0], dtype=dtype)),
            nn.Parameter(torch.tensor([1.0], dtype=dtype))
        ])

    def add_subdomain(self, mu_init):
        # 动态添加的新网络强制开启 RFF，作为啃硬骨头的“激波专家”
        new_net = DeepSubNet(self.hidden, is_new_addition=True, use_rff=True).to(device)
        mu = nn.Parameter(torch.tensor(mu_init, dtype=dtype).view(1, 2))
        lg = nn.Parameter(torch.tensor([3.0], dtype=dtype)) # 更尖锐的控制域权重
        
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
# 3. 损失函数与采样工具
# ==========================================================
def pde_residual(model, x, t, current_nu):
    u = model(x, t)
    u_dt, u_dx = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    return u_dt + u * u_dx - current_nu * u_dxx

def get_uniform_points(N):
    x = (-1 + 2 * torch.rand(N, 1, device=device, dtype=dtype))
    t = torch.rand(N, 1, device=device, dtype=dtype)
    return x, t

# ==========================================================
# 4. 训练设置
# ==========================================================
target_nu = 0.01 / np.pi
ADAM_EPOCHS = 20000 
BATCH_SIZE = 4000
RADS_RATIO = 0.6  

model = DynamicABPINN(hidden=64).to(device)
generator = DisplacementGenerator().to(device)

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

t_bc = torch.rand(400, 1, device=device, dtype=dtype)
x_bc_l = -1.0 * torch.ones_like(t_bc)
x_bc_r = 1.0 * torch.ones_like(t_bc)
u_bc_val = torch.zeros_like(t_bc)

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)
scheduler_pinn = torch.optim.lr_scheduler.ExponentialLR(optim_pinn, gamma=0.999)

# ==========================================================
# 5. 主训练循环 (对抗生成 + 架构演化)
# ==========================================================
print("\n>>> Phase 1: Adversarial Adam (Displacement Gen vs RFF AB-PINN)...")
start_time = time.time()

for epoch in range(ADAM_EPOCHS + 1):
    curr_nu = target_nu 

    # ----------------------------------------------------
    # A. 训练 Generator 
    # ----------------------------------------------------
    if epoch % 2 == 0:
        optim_gen.zero_grad()
        # 🌟 喂给生成器的是均匀网格点，而不是纯噪声！
        z_x, z_t = get_uniform_points(BATCH_SIZE // 2)
        z = torch.cat([z_x, z_t], dim=1).requires_grad_(True)
        
        x_gen, t_gen = generator(z)
        res = pde_residual(model, x_gen, t_gen, curr_nu)
        
        # 🌟 彻底移除了 Variance Loss！因为均匀底座已经保证了多样性
        loss_gen = -torch.mean(res**2) 
        loss_gen.backward()
        optim_gen.step()

    # ----------------------------------------------------
    # B. 训练 AB-PINN 
    # ----------------------------------------------------
    optim_pinn.zero_grad()
    
    N_adv = int(BATCH_SIZE * RADS_RATIO)
    with torch.no_grad():
        z_x, z_t = get_uniform_points(N_adv)
        z = torch.cat([z_x, z_t], dim=1)
        x_adv, t_adv = generator(z)
        
    x_adv = x_adv.detach().requires_grad_(True)
    t_adv = t_adv.detach().requires_grad_(True)
    
    x_uni, t_uni = get_uniform_points(BATCH_SIZE - N_adv)
    x_uni.requires_grad_(True); t_uni.requires_grad_(True)
    
    x_train = torch.cat([x_adv, x_uni], dim=0)
    t_train = torch.cat([t_adv, t_uni], dim=0)
    
    res_pinn = pde_residual(model, x_train, t_train, curr_nu)
    loss_r = torch.mean(res_pinn**2)
    
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    loss_bc = torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2)
              
    loss_pinn = loss_r + 200.0 * (loss_ic + loss_bc)
    loss_pinn.backward()
    optim_pinn.step()

    # ----------------------------------------------------
    # C. 协同进化：添加 RFF 专家网络 
    # ----------------------------------------------------
    if epoch > 3000 and epoch % 3000 == 0:
        res_eval = res_pinn[:N_adv].abs().detach() 
        max_res = res_eval.max().item()
            
        if len(model.subnets) < 6 and max_res > 0.05:
            idx = torch.argmax(res_eval)
            new_mu = [x_adv[idx].item(), t_adv[idx].item()]
            
            print(f"⚡️ [侦测到激波]: Max Res={max_res:.4f} @ {new_mu} -> 添加 RFF 子网 {len(model.subnets)+1}")
            model.add_subdomain(new_mu)
            
            # 添加子网后重置优化器状态
            optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
            scheduler_pinn = torch.optim.lr_scheduler.ExponentialLR(optim_pinn, gamma=0.999)

    if epoch % 1000 == 0:
        scheduler_pinn.step()
        print(f"Ep {epoch:5d} | PINN Loss: {loss_pinn.item():.4e} | Subnets: {len(model.subnets)}")

print(f"Phase 1 Time: {time.time()-start_time:.1f}s")

# ==========================================================
# 6. L-BFGS 终极微调 (结合困难样本挖掘 Hard-Mining)
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning with Hard-Mining...")

# 🌟 Hard-Mining: 强行筛选当前残差最高的 3000 个点喂给 L-BFGS
x_dense = torch.linspace(-1, 1, 100, device=device, dtype=dtype)
t_dense = torch.linspace(0, 1, 100, device=device, dtype=dtype)
X_d, T_d = torch.meshgrid(x_dense, t_dense, indexing='ij')
X_d_flat = X_d.flatten().unsqueeze(1).requires_grad_(True)
T_d_flat = T_d.flatten().unsqueeze(1).requires_grad_(True)

with torch.no_grad():
    res_dense = pde_residual(model, X_d_flat, T_d_flat, target_nu).abs()
    topk_idx = torch.topk(res_dense.squeeze(), 3000).indices

x_hard = X_d_flat[topk_idx].detach().requires_grad_(True)
t_hard = T_d_flat[topk_idx].detach().requires_grad_(True)

with torch.no_grad():
    z_x, z_t = get_uniform_points(3000)
    x_adv, t_adv = generator(torch.cat([z_x, z_t], dim=1))
x_uni, t_uni = get_uniform_points(2000)

x_lbfgs = torch.cat([x_hard, x_adv.detach().requires_grad_(True), x_uni.requires_grad_(True)], dim=0)
t_lbfgs = torch.cat([t_hard, t_adv.detach().requires_grad_(True), t_uni.requires_grad_(True)], dim=0)

lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, max_iter=10000, max_eval=10000, 
    history_size=100, tolerance_grad=1e-15, tolerance_change=1e-15,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    res = pde_residual(model, x_lbfgs, t_lbfgs, target_nu)
    loss = torch.mean(res**2) + 300.0 * (
        torch.mean((model(x_ic, t_ic) - u_ic)**2) + 
        torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2)
    )
    loss.backward()
    return loss

lbfgs.step(closure)

# ==========================================================
# 7. 验证与出图
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(len(t_exact), len(x_exact)).T

error_l2 = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)
print(f"\n✨✨✨ Final Relative L2 Error (Displacement Gen + RFF AB-PINN): {error_l2:.4e} ✨✨✨")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
idx = int(0.5 * len(t_exact))
plt.plot(x_exact, u_exact_all[:, idx], 'k-', linewidth=3, label="Exact")
plt.plot(x_exact, u_pred[:, idx], 'r--', linewidth=2, label="Prediction")
plt.title(f"Burgers at t=0.5 | L2 Error: {error_l2:.2e}")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
with torch.no_grad():
    z_x, z_t = get_uniform_points(4000)
    x_vis, t_vis = generator(torch.cat([z_x, z_t], dim=1))
plt.scatter(t_vis.cpu(), x_vis.cpu(), s=2, c='blue', alpha=0.3, label='Displacement Gen Points')

centers_np = np.array([[c.detach().cpu().numpy().flatten()[1], c.detach().cpu().numpy().flatten()[0]] for c in model.centers])
plt.scatter(centers_np[:, 0], centers_np[:, 1], c='red', edgecolors='black', s=150, marker='*', label='RFF Subnets', zorder=10)

plt.xlabel("t"); plt.ylabel("x")
plt.title("Displacement Generator & RFF Subnet Locations")
plt.legend()
plt.tight_layout()
plt.savefig('Adv_Disp_RFF_ABPINN_Burgers.png', dpi=300)
print("✅ 结果已保存为 Adv_Disp_RFF_ABPINN_Burgers.png")