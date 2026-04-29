import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os

# ==========================================================
# 0. 基础配置 (CPU + Float64)
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cpu") # 4090跑这种小规模点数并不一定比CPU快，且CPU Float64最稳
dtype = torch.float64 
torch.set_default_dtype(dtype)

print(f"🔥 设备: {device} | 精度: Float64 (Stable Hybrid Version)")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据读取
# ==========================================================
try:
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat"
    data = loadmat(file_path)
    x_exact = data["x"].flatten()
    t_exact = data["t"].flatten()
    u_exact_all = data["usol"]
    print(f"✅ 成功读取真值数据")
except FileNotFoundError:
    print("❌ 错误：找不到 .mat 文件！")
    exit()

X_star = torch.tensor(np.meshgrid(x_exact, t_exact)[0].flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(np.meshgrid(x_exact, t_exact)[1].flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 模型定义 (Deep Network + Zero Init)
# ==========================================================
class DeepSubNet(nn.Module):
    def __init__(self, hidden=64, is_new_addition=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # 初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        # 🔥 关键修改：如果是中途加入的新网络，将输出层初始化为0
        # 这样它刚加入时输出为0，不会破坏当前的 Loss，随后慢慢学习
        if is_new_addition:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

class DynamicABPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        # 初始 2 个子网，覆盖左右
        self.subnets = nn.ModuleList([DeepSubNet(hidden), DeepSubNet(hidden)])
        self.centers = nn.ParameterList([
            nn.Parameter(torch.tensor([-0.5, 0.2], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([0.5, 0.8], device=device, dtype=dtype))
        ])
        self.log_gammas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype))
        ])

    def add_subdomain(self, mu_init):
        # 🔥 使用 Zero Init 创建新子网
        new_net = DeepSubNet(self.hidden, is_new_addition=True).to(device)
        mu = nn.Parameter(torch.tensor(mu_init, device=device, dtype=dtype).view(1, 2))
        lg = nn.Parameter(torch.tensor([3.0], device=device, dtype=dtype))
        
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
# 3. 混合采样 (Hybrid Sampling) - 救命稻草
# ==========================================================
def pde_residual(model, x, t, current_nu):
    u = model(x, t)
    u_dt, u_dx = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    return u_dt + u * u_dx - current_nu * u_dxx

def redistribute_points_hybrid(model, n_total, current_nu):
    """ 🔥 混合采样：30% 均匀点 + 70% 自适应点 """
    model.eval()
    
    # 1. 均匀采样部分 (保底，防止平坦区域被遗忘)
    n_uniform = int(0.3 * n_total)
    x_uni = (-1 + 2 * torch.rand(n_uniform, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_uni = torch.rand(n_uniform, 1, device=device, dtype=dtype).requires_grad_(True)
    
    # 2. AAIS 自适应采样部分
    n_adaptive = n_total - n_uniform
    n_cand = n_adaptive * 5 # 候选池
    x_cand = (-1 + 2 * torch.rand(n_cand, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_cand = torch.rand(n_cand, 1, device=device, dtype=dtype).requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        res = pde_residual(model, x_cand, t_cand, current_nu)
        score = torch.abs(res).detach().cpu().numpy().flatten()
    
    # 基于残差概率采样
    score = score ** 2 
    pdf = score / (np.sum(score) + 1e-10)
    idx = np.random.choice(n_cand, size=n_adaptive, p=pdf, replace=False)
    
    x_adapt = x_cand[idx].detach().requires_grad_(True)
    t_adapt = t_cand[idx].detach().requires_grad_(True)
    
    # 3. 合并
    x_final = torch.cat([x_uni, x_adapt], dim=0)
    t_final = torch.cat([t_uni, t_adapt], dim=0)
    
    model.train()
    return x_final, t_final

# ==========================================================
# 4. 训练准备
# ==========================================================
target_nu = 0.01 / np.pi
start_nu = 10.0 * target_nu 
ADAM_EPOCHS = 20000 
N_total = 4000      

model = DynamicABPINN(hidden=64).to(device)

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# BC: 强行设为 0
t_bc = torch.rand(200, 1, device=device, dtype=dtype)
x_bc_l = -1.0 * torch.ones_like(t_bc)
x_bc_r = 1.0 * torch.ones_like(t_bc)
u_bc_val = torch.zeros_like(t_bc)

def compute_loss_val(model, x_r, t_r, nu_val):
    res = pde_residual(model, x_r, t_r, nu_val)
    loss_r = torch.mean(res**2)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    # 边界条件权重稍微给大一点
    loss_bc = torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + \
              torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2)
    return loss_r + 200.0 * (loss_ic + loss_bc), res

# 初始采样
curr_nu = start_nu
x_r, t_r = redistribute_points_hybrid(model, N_total, curr_nu)

# ==========================================================
# Phase 1: Adam
# ==========================================================
print(">>> Phase 1: Adam (Stable Mode)...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
start_time = time.time()

for epoch in range(ADAM_EPOCHS + 1):
    # 课程学习
    if epoch < 10000:
        curr_nu = start_nu - (start_nu - target_nu) * (epoch / 10000)
    else:
        curr_nu = target_nu

    # 动态采样 (频率略微降低，保证稳定)
    if epoch > 0 and epoch % 2000 == 0:
        x_r, t_r = redistribute_points_hybrid(model, N_total, curr_nu)

    optimizer.zero_grad()
    loss, res_tensor = compute_loss_val(model, x_r, t_r, curr_nu)
    loss.backward()
    optimizer.step()

    # 动态加网
    if epoch > 2000 and epoch % 3000 == 0:
        max_res = res_tensor.abs().max().item()
        # 只要残差还显著，且数量不超过 6
        if len(model.subnets) < 6 and max_res > 0.05:
            idx = torch.argmax(res_tensor.abs())
            # 这里的 x_r, t_r 是混合点，包含了高误差区域
            new_mu = [x_r[idx].item(), t_r[idx].item()]
            
            print(f"⚡️ 自适应触发: Max Res={max_res:.4f} @ {new_mu} -> Add SubNet {len(model.subnets)+1}")
            model.add_subdomain(new_mu)
            
            # 重置优化器 (因为有了 zero init，可以用稍大的 lr 继续)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if epoch % 1000 == 0:
        print(f"Ep {epoch:5d} | Loss: {loss.item():.5e} | Subnets: {len(model.subnets)}")

print(f"Adam Time: {time.time()-start_time:.1f}s")

# ==========================================================
# Phase 2: L-BFGS (决胜局)
# ==========================================================
print("\n>>> Phase 2: L-BFGS Fine-tuning...")

# 最后再采样一次高质量点
x_r, t_r = redistribute_points_hybrid(model, N_total, target_nu)

lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=10000, max_eval=10000, 
    history_size=100, 
    tolerance_grad=1e-15, tolerance_change=1e-15,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    loss, _ = compute_loss_val(model, x_r, t_r, target_nu)
    loss.backward()
    return loss

lbfgs.step(closure)
print(f"L-BFGS Final Loss: {closure().item():.5e}")

# ==========================================================
# 验证
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(len(t_exact), len(x_exact)).T

error_l2 = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)
print(f"\n✨✨✨ Final Relative L2 Error: {error_l2:.4e} ✨✨✨")

# 绘图
plt.figure(figsize=(14, 6))

# 解对比
plt.subplot(1, 2, 1)
idx = int(0.5 * len(t_exact))
plt.plot(x_exact, u_exact_all[:, idx], 'k-', linewidth=3, label="Exact")
plt.plot(x_exact, u_pred[:, idx], 'r--', linewidth=2, label="Prediction")
plt.title(f"t=0.5 | Error: {error_l2:.2e}")
plt.legend()
plt.grid(True, alpha=0.3)

# 残差与采样点分布
plt.subplot(1, 2, 2)
if not x_r.requires_grad: x_r.requires_grad_(True)
if not t_r.requires_grad: t_r.requires_grad_(True)
_, res_final = compute_loss_val(model, x_r, t_r, target_nu)
res_np = res_final.detach().cpu().numpy()

sc = plt.scatter(t_r.detach().cpu().numpy(), x_r.detach().cpu().numpy(), 
                 c=res_np, s=3, cmap='jet', alpha=0.7)
plt.colorbar(sc, label='Residual')

# 绘制中心
centers_np = np.array([[c.detach().cpu().numpy().flatten()[1], c.detach().cpu().numpy().flatten()[0]] for c in model.centers])
plt.scatter(centers_np[:, 0], centers_np[:, 1], c='white', edgecolors='black', s=120, marker='X', label='Centers', zorder=10)

plt.xlabel("t")
plt.ylabel("x")
plt.title(f"Subnets: {len(model.subnets)} | Hybrid Sampling")
plt.tight_layout()
plt.savefig('hybrid_fix_result.png', dpi=300)
print("✅ 结果已保存")