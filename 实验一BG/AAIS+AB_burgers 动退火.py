import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time
import glob
from PIL import Image
from torch.distributions import MultivariateNormal, Categorical

# ==========================================================
# 0. 基础配置
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cpu") 
dtype = torch.float64 
torch.set_default_dtype(dtype)

print(f"🔥 任务: AAIS + AB-PINN (Gamma Annealing) | Device: {device}")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备
# ==========================================================
file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一BG/burgers_shock.mat"

if not os.path.exists(file_path):
    print(f"❌ 错误: 文件不存在!")
    exit()

data = scipy.io.loadmat(file_path)
x_exact = data["x"].flatten()
t_exact = data["t"].flatten()
u_exact_all = data["usol"]

Nx = len(x_exact)
Nt = len(t_exact)
print(f"✅ 数据加载成功: Nx={Nx}, Nt={Nt}")

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact, indexing='ij')
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

t_bc = torch.rand(1000, 1, device=device, dtype=dtype)
x_bc_l = -1.0 * torch.ones_like(t_bc)
x_bc_r = 1.0 * torch.ones_like(t_bc)
u_bc_val = torch.zeros_like(t_bc)

loss_history = []

# ==========================================================
# 2. AAIS 采样器
# ==========================================================
class AAIS_Sampler:
    def __init__(self, n_components=12, z_dim=2, device=device):
        self.K = n_components 
        self.dim = z_dim
        self.device = device
        self.alpha = torch.ones(self.K, device=device, dtype=dtype) / self.K
        self.mu = torch.rand(self.K, self.dim, device=device, dtype=dtype)
        self.mu[:, 0] = self.mu[:, 0] * 2 - 1 
        self.mu[:, 1] = self.mu[:, 1]       
        self.cov_diag = torch.ones(self.K, self.dim, device=device, dtype=dtype) * 0.05

    def get_log_prob(self, samples):
        comp = MultivariateNormal(self.mu, torch.diag_embed(self.cov_diag))
        log_probs_comp = comp.log_prob(samples.unsqueeze(1)) 
        return torch.logsumexp(log_probs_comp + torch.log(self.alpha + 1e-16), dim=1)

    def sample(self, N):
        mix = Categorical(self.alpha)
        comp_indices = mix.sample((N,))
        samples = torch.zeros(N, self.dim, device=self.device, dtype=dtype)
        counts = torch.bincount(comp_indices, minlength=self.K)
        ptr = 0
        for k in range(self.K):
            if counts[k] > 0:
                s = MultivariateNormal(self.mu[k], torch.diag_embed(self.cov_diag[k])).sample((counts[k],))
                samples[ptr:ptr+counts[k]] = s
                ptr += counts[k]
        samples[:, 0] = torch.clamp(samples[:, 0], -1.0, 1.0)
        samples[:, 1] = torch.clamp(samples[:, 1], 0.0, 1.0)
        return samples

    def em_step(self, target_log_prob_func, lambda_k, n_samples=2000):
        X = self.sample(n_samples).detach()
        log_q = self.get_log_prob(X)
        log_Q_target = target_log_prob_func(X) 
        log_Q_target = torch.clamp(log_Q_target, max=30.0, min=-30.0)
        log_target_k = (1 - lambda_k) * log_q + lambda_k * log_Q_target
        w = torch.softmax(log_target_k - log_q, dim=0) 
        comp = MultivariateNormal(self.mu, torch.diag_embed(self.cov_diag))
        rho = torch.exp(torch.log(self.alpha + 1e-16) + comp.log_prob(X.unsqueeze(1)) - log_q.unsqueeze(1))
        rho = rho / (torch.sum(rho, dim=1, keepdim=True) + 1e-16)
        eff_w = w.unsqueeze(1) * rho 
        sum_eff_w = torch.sum(eff_w, dim=0) + 1e-16
        self.alpha = sum_eff_w / torch.sum(sum_eff_w)
        self.mu = (eff_w.T @ X) / sum_eff_w.unsqueeze(1)
        for k in range(self.K):
            diff = X - self.mu[k]
            w_k = eff_w[:, k].unsqueeze(1)
            cov_k = torch.sum(w_k * (diff**2), dim=0) / sum_eff_w[k]
            self.cov_diag[k] = torch.clamp(cov_k, min=1e-6, max=0.2)

# ==========================================================
# 3. Dynamic AB-PINN (Gamma Annealing)
# ==========================================================
class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype)) 
    def forward(self, x):
        return torch.tanh(self.scale * x)

class DeepSubNet(nn.Module):
    def __init__(self, hidden=50, is_new_addition=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        if is_new_addition:
            # 🔥 激活初始化：不要全零，给一点点噪声，让它"活着"
            nn.init.normal_(self.net[-1].weight, mean=0, std=0.01)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

class DynamicABPINN(nn.Module):
    def __init__(self, hidden=50):
        super().__init__()
        self.hidden = hidden
        self.subnets = nn.ModuleList([DeepSubNet(hidden), DeepSubNet(hidden)])
        self.centers = nn.ParameterList([
            nn.Parameter(torch.tensor([-0.5, 0.2], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([0.5, 0.8], device=device, dtype=dtype))
        ])
        # 🔥 全局动态 Gamma：控制子网的锐利程度
        self.global_gamma_val = 0.1 

    def update_gamma(self, progress):
        # 门控退火：从 0.1 (软) 线性增加到 5.0 (硬)
        # progress: 0.0 -> 1.0
        self.global_gamma_val = 0.1 + (5.0 - 0.1) * progress

    def add_subdomain(self, mu_init):
        new_net = DeepSubNet(self.hidden, is_new_addition=True).to(device)
        mu = nn.Parameter(torch.tensor(mu_init, device=device, dtype=dtype).view(1, 2))
        self.subnets.append(new_net)
        self.centers.append(mu)

    def get_subnet_weights(self, x, t):
        xt = torch.cat([x, t], dim=1)
        logits = []
        for i in range(len(self.subnets)):
            dist_sq = torch.sum((xt - self.centers[i])**2, dim=1, keepdim=True)
            # 使用动态 gamma
            logits.append(-self.global_gamma_val * dist_sq)
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1) 
        return weights.squeeze(2) 

    def forward(self, x, t):
        weights = self.get_subnet_weights(x, t) 
        u_out = 0
        for i in range(len(self.subnets)):
            u_out += weights[:, i:i+1] * self.subnets[i](x, t)
        return u_out

# ==========================================================
# 4. 辅助函数
# ==========================================================
def pde_residual(model, x, t, current_nu):
    u = model(x, t)
    u_dt, u_dx = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    return u_dt + u * u_dx - current_nu * u_dxx

def target_log_prob_wrapper(model, samples, nu):
    x = samples[:, 0:1].requires_grad_(True)
    t = samples[:, 1:2].requires_grad_(True)
    res = pde_residual(model, x, t, nu)
    return torch.log(res**2 + 1e-16).detach().squeeze()

def plot_snapshot(epoch, model, aais, current_nu):
    print(f"📸 Generating Snapshot at Epoch {epoch}...")
    plt.figure(figsize=(10, 8), dpi=100)
    
    with torch.no_grad():
        u_pred = model(X_star, T_star).cpu().numpy().reshape(Nx, Nt)
        if u_pred.shape != u_exact_all.shape: u_pred = u_pred.T
        err_map = np.abs(u_exact_all - u_pred)
        weights = model.get_subnet_weights(X_star, T_star).cpu().numpy()
        subnet_map = np.argmax(weights, axis=1).reshape(Nx, Nt)

    plt.imshow(err_map, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='jet', alpha=0.9)
    plt.colorbar(label='Abs Error')
    
    samps = aais.sample(2000).cpu().numpy()
    plt.scatter(samps[:, 1], samps[:, 0], s=1, c='white', alpha=0.3, label='AAIS') 

    T_np = T_star.reshape(Nx, Nt).cpu().numpy()
    X_np = X_star.reshape(Nx, Nt).cpu().numpy()
    plt.contour(T_np, X_np, subnet_map, levels=np.arange(len(model.subnets)), colors='black', linewidths=0.5, alpha=0.5)

    centers_list = []
    for c in model.centers:
        c_np = c.detach().cpu().numpy().flatten()
        centers_list.append(c_np)
    centers_np = np.array(centers_list)
    plt.scatter(centers_np[:, 1], centers_np[:, 0], c='red', marker='*', s=200, edgecolors='white', zorder=10)

    plt.xlabel("Time (t)"); plt.ylabel("Space (x)")
    plt.xlim(0, 1); plt.ylim(-1, 1)
    plt.title(f"Epoch {epoch} | Gamma={model.global_gamma_val:.2f} | Nets={len(model.subnets)}")
    plt.legend(loc='upper right', framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'Evolution_Ep{epoch:05d}.png')
    plt.close()

# ==========================================================
# 5. 主训练循环
# ==========================================================
target_nu = 0.01 / np.pi
start_nu = 2.0 * target_nu 
ADAM_EPOCHS = 15000 
BATCH_SIZE = 5000

model = DynamicABPINN(hidden=50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 暂时不用 Scheduler，手动控制
aais = AAIS_Sampler(n_components=12, z_dim=2, device=device)
lambda_schedule = np.linspace(0.1, 1.0, ADAM_EPOCHS)

# --- Phase 0: Warm-up ---
print("\n>>> Phase 0: Warm-up...")
plot_snapshot(0, model, aais, start_nu)

for epoch in range(2001):
    optimizer.zero_grad()
    x_uni = (-1 + 2*torch.rand(4000, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_uni = torch.rand(4000, 1, device=device, dtype=dtype).requires_grad_(True)
    res = pde_residual(model, x_uni, t_uni, start_nu)
    loss = torch.mean(res**2) + 100.0 * (torch.mean((model(x_ic, t_ic) - u_ic)**2) + 
           torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2))
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0: print(f"Warmup Ep {epoch} | Loss: {loss.item():.5e}")

# --- Phase 1: AAIS + Dynamic Growth ---
print("\n>>> Phase 1: AAIS-Driven Training (Annealing Gamma)...")
start_time = time.time()

for epoch in range(ADAM_EPOCHS + 1):
    curr_nu = start_nu - (start_nu - target_nu) * (epoch / (ADAM_EPOCHS // 2)) if epoch < (ADAM_EPOCHS // 2) else target_nu
    current_bc_weight = 100.0 + (5000.0 - 100.0) * min(1.0, epoch / 5000.0)

    # 🔥 门控退火 🔥
    model.update_gamma(epoch / ADAM_EPOCHS)

    if epoch % 30 == 0:
        curr_lambda = lambda_schedule[epoch] if epoch < len(lambda_schedule) else 1.0
        target_func = lambda s: target_log_prob_wrapper(model, s, curr_nu)
        aais.em_step(target_func, lambda_k=curr_lambda, n_samples=3000)

    optimizer.zero_grad()
    samples_aais = aais.sample(int(0.7 * BATCH_SIZE))
    x_ad, t_ad = samples_aais[:, 0:1].requires_grad_(True), samples_aais[:, 1:2].requires_grad_(True)
    x_uni = (-1 + 2*torch.rand(int(0.3 * BATCH_SIZE), 1, device=device, dtype=dtype)).requires_grad_(True)
    t_uni = torch.rand(int(0.3 * BATCH_SIZE), 1, device=device, dtype=dtype).requires_grad_(True)
    x_train = torch.cat([x_ad, x_uni], dim=0)
    t_train = torch.cat([t_ad, t_uni], dim=0)
    
    res = pde_residual(model, x_train, t_train, curr_nu)
    loss_r = torch.mean(res**2)
    loss_bc = torch.mean((model(x_ic, t_ic) - u_ic)**2) + \
              torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + \
              torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2)
    
    loss = loss_r + current_bc_weight * loss_bc
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    # Direct Coupling
    if epoch > 2000 and epoch % 1500 == 0: # 加网频率稍微快点
        scout_locs = aais.mu.detach()
        scout_x = scout_locs[:, 0:1].requires_grad_(True)
        scout_t = scout_locs[:, 1:2].requires_grad_(True)
        scout_res = pde_residual(model, scout_x, scout_t, curr_nu).abs()
        best_scout_idx = torch.argmax(scout_res)
        max_res_val = scout_res[best_scout_idx].item()
        
        if len(model.subnets) < 10 and max_res_val > 0.05:
            target_loc = scout_locs[best_scout_idx].detach().cpu().numpy().flatten()
            print(f"⚡️ AAIS引导加网: Ep {epoch} | Loc=({target_loc[0]:.2f}, {target_loc[1]:.2f}) | Res={max_res_val:.4f}")
            model.add_subdomain(target_loc)
            # 简单粗暴：加网时重置一下 LR，给新网一点动力
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3

    if epoch % 2500 == 0:
        print(f"Ep {epoch:5d} | Loss: {loss.item():.4e} | Nets: {len(model.subnets)} | Gamma: {model.global_gamma_val:.2f}")
        plot_snapshot(epoch, model, aais, curr_nu)

# --- Phase 2: L-BFGS ---
print("\n>>> Phase 2: L-BFGS Refinement (70/30 Balanced)...")
model.update_gamma(1.0) # L-BFGS 阶段把 Gamma 锁死在最硬状态
N_LBFGS = 20000 
samples_aais = aais.sample(int(0.7 * N_LBFGS))
x_ad, t_ad = samples_aais[:, 0:1].detach().requires_grad_(True), samples_aais[:, 1:2].detach().requires_grad_(True)
x_uni = (-1 + 2*torch.rand(int(0.3 * N_LBFGS), 1, device=device, dtype=dtype)).requires_grad_(True)
t_uni = torch.rand(int(0.3 * N_LBFGS), 1, device=device, dtype=dtype).requires_grad_(True)
x_final = torch.cat([x_ad, x_uni], dim=0)
t_final = torch.cat([t_ad, t_uni], dim=0)

lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=20000, history_size=150, 
    tolerance_grad=1e-15, tolerance_change=1e-15, line_search_fn="strong_wolfe"
)

iter_count = 0
FINAL_BC_WEIGHT = 50000.0

def closure():
    global iter_count
    lbfgs.zero_grad()
    res = pde_residual(model, x_final, t_final, target_nu)
    loss = torch.mean(res**2) + FINAL_BC_WEIGHT * (torch.mean((model(x_ic, t_ic) - u_ic)**2) + 
           torch.mean((model(x_bc_l, t_bc) - u_bc_val)**2) + torch.mean((model(x_bc_r, t_bc) - u_bc_val)**2))
    loss.backward()
    iter_count += 1
    if iter_count % 1000 == 0:
        print(f"L-BFGS Iter {iter_count:4d} | Loss: {loss.item():.5e}")
        loss_history.append(loss.item())
    return loss
lbfgs.step(closure)

plot_snapshot(99999, model, aais, target_nu)

# ==========================================================
# 6. 后处理
# ==========================================================
print("\n>>> Generating Analysis Results...")

plt.figure(figsize=(10, 5))
plt.semilogy(loss_history, 'b-', linewidth=1.5)
plt.xlabel("Iterations"); plt.ylabel("Total Loss"); plt.title("Training Loss Evolution")
plt.grid(True, alpha=0.3); plt.savefig("Training_Loss_History.png", dpi=150)

img_files = sorted(glob.glob("Evolution_Ep*.png"))
if len(img_files) > 0:
    frames = [Image.open(f) for f in img_files]
    frames[0].save('Training_Evolution.gif', format='GIF', 
                   append_images=frames[1:], save_all=True, duration=800, loop=0)
    print(f"✅ 动图已生成: Training_Evolution.gif ({len(frames)} frames)")

model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(Nx, Nt)
if u_pred.shape != u_exact_all.shape: u_pred = u_pred.T
error_l2 = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)
print(f"\n✨✨✨ Final Relative L2 Error: {error_l2:.4e} ✨✨✨")