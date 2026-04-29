import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import time

# ==========================================================
# 0. 基础配置
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda")
dtype = torch.float64
torch.set_default_dtype(dtype)

print("🔥 任务: B选项终极完全体 (因果时间推进 + RADS-GAN + AB-PINN)")
print("🚨 核心策略: 解决全局相位漂移，按时间轴由近及远逐步解锁训练域！")
torch.manual_seed(1234)
np.random.seed(1234)

# 注意修改路径（已为你保留原路径）
DATA_PATH = '/3241003007/zy/实验三KDV/KdV.mat'
data = scipy.io.loadmat(DATA_PATH)

# ==========================================================
# 1. 数据读取与预处理
# ==========================================================
data = scipy.io.loadmat(DATA_PATH)
x_exact = data["x"].flatten()
t_exact = data["tt"].flatten()
u_exact = data["uu"].real.T

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
x_min, x_max = x_exact.min(), x_exact.max()
t_min, t_max = t_exact.min(), t_exact.max()

LAMBDA_1 = 1.0
LAMBDA_2 = 0.0025

# ==========================================================
# 2. 对抗寻点生成器 (RADS-GAN)
# ==========================================================
class RADS_Generator(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, z, current_t_max):
        raw = self.net(z)
        x = torch.tanh(raw[:, 0:1]) * max(abs(x_min), abs(x_max)) 
        # 🌟 GAN 生成的点也被严格限制在当前的动态时间窗口内
        t = torch.sigmoid(raw[:, 1:2]) * (current_t_max - t_min) + t_min
        return x, t

# ==========================================================
# 3. 动态网络架构
# ==========================================================
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            # 🌟 KdV 存在高阶导数，增加正弦辅助能更好追踪波的相位，这里改回Tanh保持平稳
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.out = nn.Linear(layers[-2], layers[-1])
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t): 
        return self.out(self.net(torch.cat([x, t], dim=1)))

class DynamicABPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = MLP(layers=[2, 64, 64, 64, 64, 1]) 
        self.experts = nn.ModuleList()                     
        self.centers = []                                  
        self.gamma = 15.0  # 稍微放宽专家辐射范围，避免崎岖地形                              

    def forward(self, x, t):
        u_pred = self.base_net(x, t)
        for i, expert in enumerate(self.experts):
            cx, ct = self.centers[i]
            weight = torch.exp(-self.gamma * ((x - cx)**2 + (t - ct)**2))
            u_pred = u_pred + weight * expert(x, t)
        return u_pred
        
    def add_expert(self, cx, ct):
        new_expert = MLP(layers=[2, 40, 40, 40, 1]).to(device)
        nn.init.zeros_(new_expert.out.weight)
        nn.init.zeros_(new_expert.out.bias)
        self.experts.append(new_expert)
        self.centers.append((cx, ct))
        print(f"🌟 [战术空投] 第 {len(self.experts)} 个子网，锚点: x={cx:.3f}, t={ct:.3f}")
        return new_expert

model = DynamicABPINN().to(device)
generator = RADS_Generator().to(device)

# ==========================================================
# 4. 物理残差与计算
# ==========================================================
N_bc = 400
x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)

def pde_residual(model, x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

def compute_loss(x_f, t_f, t_bc=None):
    res = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(res**2) 
        
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    
    if t_bc is None: 
        t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        
    x_lb = (x_min * torch.ones_like(t_bc)).requires_grad_(True)
    x_ub = (x_max * torch.ones_like(t_bc)).requires_grad_(True)
    
    u_lb, u_ub = model(x_lb, t_bc), model(x_ub, t_bc)
    u_x_lb = torch.autograd.grad(u_lb, x_lb, torch.ones_like(u_lb), create_graph=True)[0]
    u_x_ub = torch.autograd.grad(u_ub, x_ub, torch.ones_like(u_ub), create_graph=True)[0]
    u_xx_lb = torch.autograd.grad(u_x_lb, x_lb, torch.ones_like(u_x_lb), create_graph=True)[0]
    u_xx_ub = torch.autograd.grad(u_x_ub, x_ub, torch.ones_like(u_x_ub), create_graph=True)[0]
    
    loss_bc_u = torch.mean((u_lb - u_ub)**2)
    loss_bc_ux = torch.mean((u_x_lb - u_x_ub)**2)
    loss_bc_uxx = torch.mean((u_xx_lb - u_xx_ub)**2)
    
    # 🌟 稍微调高物理边界和初始条件的权重，从源头稳住波的形状
    total_loss = loss_f + 50.0 * loss_ic + 20.0 * loss_bc_u + 1.0 * loss_bc_ux + 0.1 * loss_bc_uxx
    return total_loss, res

def get_uniform_points_causal(N, current_t_max):
    x = (torch.rand(N, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min).requires_grad_(True)
    # 🌟 核心：只在 [0, current_t_max] 内采样
    t = (torch.rand(N, 1, device=device, dtype=dtype) * (current_t_max - t_min) + t_min).requires_grad_(True)
    return x, t

# ==========================================================
# 5. Phase 1: 时间课程式对抗训练 (Causal Training)
# ==========================================================
TOTAL_ITERS = 20000 
BATCH_SIZE = 8000 
AAIS_RATIO = 0.3   # 降低 GAN 比例，让物理均匀采样主导，防止局部震荡
MAX_EXPERTS = 4    # 限制专家数量，防止后期地形过分崎岖

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3)
optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3)

print("\n>>> 🚀 Phase 1: 课程式时间推进 + RADS-GAN + 动态空投...")
start_time = time.time()

for it in range(TOTAL_ITERS + 1):
    # 🌟 动态计算当前的时间窗口 (前 15000 步逐渐解锁整个 t 轴)
    progress = min(1.0, it / 15000.0)
    current_t_max = t_min + progress * (t_max - t_min)

    # A. 训练 Generator
    for _ in range(2): 
        optim_gen.zero_grad()
        z = torch.randn(BATCH_SIZE // 2, 2, device=device, dtype=dtype)
        x_gen, t_gen = generator(z, current_t_max)
        res_val = pde_residual(model, x_gen, t_gen)
        loss_gen = -torch.mean(res_val**2) 
        loss_gen.backward()
        optim_gen.step()

    # B. 组合采样与训练 PINN
    optim_pinn.zero_grad()
    N_aais = int(BATCH_SIZE * AAIS_RATIO)
    with torch.no_grad():
        z = torch.randn(N_aais, 2, device=device, dtype=dtype)
        x_adv, t_adv = generator(z, current_t_max)
    x_adv.requires_grad_(True); t_adv.requires_grad_(True)
    
    x_uni, t_uni = get_uniform_points_causal(BATCH_SIZE - N_aais, current_t_max)
    
    x_train = torch.cat([x_adv, x_uni], dim=0)
    t_train = torch.cat([t_adv, t_uni], dim=0)
    
    loss, res = compute_loss(x_train, t_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim_pinn.step()

    # C. 空投专家逻辑 (在时间窗口扩张时进行支援)
    if it > 5000 and it % 3000 == 0 and len(model.experts) < MAX_EXPERTS:
        res_sq = res**2
        res_adv = res_sq[:N_aais].detach()
        max_val, max_idx = torch.max(res_adv, 0)
        
        best_x, best_t = x_adv[max_idx], t_adv[max_idx]
        print(f"🎯 [最高残差锁定: {max_val.item():.4f}] 当前推进时间 t_max: {current_t_max:.2f}")
        new_exp = model.add_expert(best_x.item(), best_t.item())
        optim_pinn.add_param_group({'params': new_exp.parameters(), 'lr': 1e-3})

    if it % 1000 == 0:
        with torch.no_grad():
            err = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
        print(f"Iter {it:5d} | t_max: {current_t_max:.2f} | Loss: {loss.item():.5e} | Error: {err:.4f} | Experts: {len(model.experts)}")

phase1_time = time.time() - start_time
print(f"⏱️ Phase 1 (Adam) 耗时: {phase1_time:.2f} 秒")

# ==========================================================
# 6. Phase 2: L-BFGS 高精度微调 
# ==========================================================
print("\n🌊 >>> 🚀 Phase 2: L-BFGS 全局开眼微调...")
x_lbfgs, t_lbfgs = get_uniform_points_causal(10000, t_max) # 此时 t_max 已经是完全体 1.0
fixed_t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min

lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=5000, max_eval=5000,
    history_size=50, tolerance_grad=1e-7, line_search_fn="strong_wolfe"
)

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    lbfgs.zero_grad()
    loss, _ = compute_loss(x_lbfgs, t_lbfgs, t_bc=fixed_t_bc)
    loss.backward()
    lbfgs_iter += 1
    if lbfgs_iter % 500 == 0:
        with torch.no_grad():
            e = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
        print(f"L-BFGS Iter {lbfgs_iter:4d} | Loss: {loss.item():.5e} | Error: {e:.5f}")
    return loss

lbfgs.step(closure)

total_time = time.time() - start_time
print(f"\n✨✨=========================================✨✨")
print(f"⏱️ 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

# ==========================================================
# 7. 评估与可视化
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)

print(f"✨ 终极完全体 最终相对 L2 误差: {final_error:.4e} ✨")

fig = plt.figure(figsize=(18, 5))

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"Causal Full-Version Predict (Error: {final_error:.4f})")
ax1.set_xlabel('t'); ax1.set_ylabel('x')
plt.colorbar(im1, ax=ax1)

ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title("Absolute Error")
ax2.set_xlabel('t'); ax2.set_ylabel('x')
plt.colorbar(im2, ax=ax2)

ax3 = plt.subplot(1, 3, 3)
idx_t1, idx_t2 = int(0.5 * len(t_exact)), int(0.8 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t1, :], 'k-', linewidth=2, label=f"Exact t={t_exact[idx_t1]:.2f}")
ax3.plot(x_exact, u_pred[idx_t1, :], 'r--', linewidth=2, label="Predict")
ax3.plot(x_exact, u_exact[idx_t2, :], 'b-', linewidth=2, label=f"Exact t={t_exact[idx_t2]:.2f}")
ax3.plot(x_exact, u_pred[idx_t2, :], 'g--', linewidth=2, label="Predict")
ax3.set_title("Wave Profile Slices")
ax3.set_xlabel('x'); ax3.set_ylabel('u(x,t)')
ax3.legend()

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, "Al44444.png")
plt.savefig(save_path, dpi=200)
print(f"✅ 图片已成功保存至: {save_path}")
plt.show()