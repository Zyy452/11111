import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# ==========================================================
# 0. 全局配置与数据准备
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print(f"🔥 任务: Baseline vs AB-PINN vs AAIS+AB (公平对比实验)")

# --- 1. 读取真值 (全场考卷) ---
try:
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"
    data = scipy.io.loadmat(file_path)
    x_exact = data["x"].flatten()
    t_exact = data["tt"].flatten()
    u_exact_all = data["uu"]
    if u_exact_all.shape[0] != len(x_exact): u_exact_all = u_exact_all.T
except:
    x_exact = np.linspace(-1, 1, 512)
    t_exact = np.linspace(0, 1, 201)
    u_exact_all = np.zeros((512, 201))

# 生成全场验证集 (Test Set) - 这是唯一的“标准答案”
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact, indexing='ij')
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
u_exact_flat = u_exact_all.flatten()

# --- 2. 物理方程参数 ---
EPSILON = 0.0001
GAMMA = 5.0

def pde_residual(model, x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t - EPSILON * u_xx + GAMMA * (u**3 - u)

# ==========================================================
# 1. 基础组件定义
# ==========================================================
class PeriodicEmbedding(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(5.0, dtype=dtype))
    def forward(self, x): return torch.tanh(self.scale * x)

# --- 标准 PINN 网络 ---
class StandardPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, 1)
        )
        # Hard IC wrapper not included here for simplicity, using Soft IC/BC or Hard structure in forward
        # 为了公平，这里统一使用 Hard Constraint 结构
        
    def forward(self, x, t):
        x_emb = self.embed(x)
        u_net = self.net(torch.cat([x_emb, t], dim=1))
        # Hard IC
        u_0 = x**2 * torch.cos(np.pi * x)
        return torch.tanh(t) * u_net + u_0

# --- AB-PINN 组件 ---
class DeepSubNet(nn.Module):
    def __init__(self, hidden=64, is_new=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, 1)
        )
        if is_new:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
    def forward(self, x_emb, t): return self.net(torch.cat([x_emb, t], dim=1))

class DynamicABPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.hidden = hidden
        self.subnets = nn.ModuleList([DeepSubNet(hidden=hidden)])
        self.centers = nn.ParameterList([nn.Parameter(torch.tensor([0.0, 0.5], device=device, dtype=dtype))])
        self.log_gammas = nn.ParameterList([nn.Parameter(torch.tensor([[3.0, 3.0]], device=device, dtype=dtype))])

    def add_subdomain(self, mu_init):
        self.subnets.append(DeepSubNet(hidden=self.hidden, is_new=True).to(device))
        self.centers.append(nn.Parameter(torch.tensor(mu_init, device=device, dtype=dtype).view(1,2)))
        self.log_gammas.append(nn.Parameter(torch.tensor([[3.0, 3.0]], device=device, dtype=dtype)))

    def forward(self, x, t):
        x_emb = self.embed(x)
        xt = torch.cat([x, t], dim=1)
        logits = []
        for i in range(len(self.subnets)):
            diff_sq = (xt - self.centers[i])**2
            gamma = torch.exp(self.log_gammas[i])
            logits.append(-torch.sum(gamma * diff_sq, dim=1, keepdim=True))
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1)
        u_out = 0
        for i in range(len(self.subnets)):
            u_out += weights[:, i] * self.subnets[i](x_emb, t)
        u_0 = x**2 * torch.cos(np.pi * x)
        return torch.tanh(t) * u_out + u_0

# --- AAIS 生成器 ---
class AAIS_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)
        )
    def forward(self, z):
        raw = self.net(z)
        return torch.tanh(raw[:,0:1]), torch.sigmoid(raw[:,1:2])

# ==========================================================
# 2. 实验控制器
# ==========================================================
# 统一参数：保证公平
TOTAL_POINTS = 6000  # 无论什么方法，每一步只能看 6000 个点
EPOCHS = 15000       # 统一训练轮数
CHECK_INTERVAL = 500

def get_fixed_points(N, seed=1234):
    """生成固定的均匀采样点"""
    torch.manual_seed(seed)
    x = (-1 + 2 * torch.rand(N, 1, device=device, dtype=dtype)).requires_grad_(True)
    t = torch.rand(N, 1, device=device, dtype=dtype).requires_grad_(True)
    return x, t

def evaluate(model):
    model.eval()
    with torch.no_grad():
        u_pred = model(X_star, T_star).cpu().numpy().flatten()
        err = np.linalg.norm(u_exact_flat - u_pred) / np.linalg.norm(u_exact_flat)
    model.train()
    return err

# --- 实验 A: Baseline PINN (固定点) ---
def run_baseline():
    print(f"\n🚀 [Exp A] Baseline PINN (Fixed {TOTAL_POINTS} Uniform Points)...")
    model = StandardPINN(hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. 锁死训练集：这就是 Baseline 的特征，数据不动
    x_train, t_train = get_fixed_points(TOTAL_POINTS)
    
    err_history = []
    
    for ep in range(EPOCHS + 1):
        optim.zero_grad()
        res = pde_residual(model, x_train, t_train)
        loss = torch.mean(res**2)
        loss.backward()
        optim.step()
        
        if ep % CHECK_INTERVAL == 0:
            err = evaluate(model)
            err_history.append(err)
            print(f"   Ep {ep:5d} | Loss: {loss.item():.4e} | Err: {err:.4f}")
            
    return err_history

# --- 实验 B: AB-PINN (固定点 + 动网) ---
def run_ab_pinn():
    print(f"\n🚀 [Exp B] AB-PINN (Fixed {TOTAL_POINTS} Points + Dynamic Experts)...")
    model = DynamicABPINN(hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. 锁死训练集
    x_train, t_train = get_fixed_points(TOTAL_POINTS)
    
    err_history = []
    
    for ep in range(EPOCHS + 1):
        optim.zero_grad()
        res = pde_residual(model, x_train, t_train)
        loss = torch.mean(res**2)
        loss.backward()
        optim.step()
        
        # 动态加网逻辑
        if ep > 2000 and ep % 3000 == 0:
            with torch.no_grad():
                res_abs = torch.abs(res).flatten()
                mask = (t_train.flatten() > 0.05)
                if mask.sum() > 0:
                    res_masked = res_abs.clone(); res_masked[~mask] = -1.0
                    max_res = res_masked.max().item()
                    if len(model.subnets) < 8 and max_res > 0.01:
                        idx = torch.argmax(res_masked)
                        print(f"      ⚡️ Added Expert @ ({x_train[idx].item():.2f}, {t_train[idx].item():.2f})")
                        model.add_subdomain([x_train[idx].item(), t_train[idx].item()])
                        optim = torch.optim.Adam(model.parameters(), lr=8e-4)

        if ep % CHECK_INTERVAL == 0:
            err = evaluate(model)
            err_history.append(err)
            print(f"   Ep {ep:5d} | Loss: {loss.item():.4e} | Err: {err:.4f} | Subs: {len(model.subnets)}")
            
    return err_history

# --- 实验 C: AAIS + AB-PINN (动点 + 动网) ---
def run_aais_ab_pinn():
    print(f"\n🚀 [Exp C] AAIS + AB-PINN (Dynamic {TOTAL_POINTS} Points + Dynamic Experts)...")
    model = DynamicABPINN(hidden=64).to(device)
    generator = AAIS_Generator().to(device)
    
    optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)
    
    err_history = []
    
    for ep in range(EPOCHS + 1):
        # 1. AAIS 采样: 点是动的！
        if ep % 5 == 0:
            for _ in range(2):
                optim_gen.zero_grad()
                z = torch.randn(TOTAL_POINTS // 2, 2, device=device, dtype=dtype)
                xg, tg = generator(z)
                loss_gen = -torch.mean(pde_residual(model, xg, tg)**2)
                loss_gen.backward()
                optim_gen.step()
        
        # 2. 生成混合训练集
        N_aais = int(0.7 * TOTAL_POINTS)
        with torch.no_grad():
            z = torch.randn(N_aais, 2, device=device, dtype=dtype)
            x_adv, t_adv = generator(z)
        x_adv.requires_grad_(True); t_adv.requires_grad_(True)
        # 补充均匀点，保证总数一致
        x_uni, t_uni = get_fixed_points(TOTAL_POINTS - N_aais, seed=ep) # 随机种子变动模拟随机采样
        x_train = torch.cat([x_adv, x_uni], dim=0)
        t_train = torch.cat([t_adv, t_uni], dim=0)
        
        # 3. PINN Update
        optim_pinn.zero_grad()
        res = pde_residual(model, x_train, t_train)
        weights = torch.ones_like(res); weights[:N_aais] = 5.0 # AAIS 权重
        loss = torch.mean(weights * res**2)
        loss.backward()
        optim_pinn.step()
        
        # 4. 动态加网
        if ep > 2000 and ep % 3000 == 0:
            with torch.no_grad():
                res_abs = torch.abs(res).flatten()
                mask = (t_train.flatten() > 0.05)
                if mask.sum() > 0:
                    res_masked = res_abs.clone(); res_masked[~mask] = -1.0
                    max_res = res_masked.max().item()
                    if len(model.subnets) < 8 and max_res > 0.01:
                        idx = torch.argmax(res_masked)
                        print(f"      ⚡️ Added Expert @ ({x_train[idx].item():.2f}, {t_train[idx].item():.2f})")
                        model.add_subdomain([x_train[idx].item(), t_train[idx].item()])
                        optim_pinn = torch.optim.Adam(model.parameters(), lr=8e-4)

        if ep % CHECK_INTERVAL == 0:
            err = evaluate(model)
            err_history.append(err)
            print(f"   Ep {ep:5d} | Loss: {loss.item():.4e} | Err: {err:.4f} | Subs: {len(model.subnets)}")
            
    return err_history

# ==========================================================
# 3. 运行与绘图
# ==========================================================
hist_A = run_baseline()
hist_B = run_ab_pinn()
hist_C = run_aais_ab_pinn()

plt.figure(figsize=(10, 6))
epochs = np.arange(0, EPOCHS + 1, CHECK_INTERVAL)

plt.semilogy(epochs, hist_A, 'k--', label='Baseline (Static Points)', linewidth=2)
plt.semilogy(epochs, hist_B, 'b-.', label='AB-PINN (Static Points + Dynamic Net)', linewidth=2)
plt.semilogy(epochs, hist_C, 'r-', label='AAIS+AB (Dynamic Points + Dynamic Net)', linewidth=2)

plt.xlabel("Epochs")
plt.ylabel("Relative L2 Error")
plt.title(f"Performance Comparison (N={TOTAL_POINTS})")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("Comparison_Result.png", dpi=300)
print("\n✅ 对比完成，结果已保存为 Comparison_Result.png")