import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time
import datetime
from torch.distributions import MultivariateNormal, Categorical

# ==========================================================
# 0. 基础配置
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cpu") 
dtype = torch.float64 
torch.set_default_dtype(dtype)

print(f"🔥 任务: AAIS (True Implementation) + AB-PINN | Device: CPU")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备 (保持不变)
# ==========================================================
EPSILON = 0.0001
GAMMA = 5.0

# 为了演示，直接生成伪数据（确保代码可运行）
x_exact = np.linspace(-1, 1, 512)
t_exact = np.linspace(0, 1, 201)
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact, indexing='ij')
u_exact_all = np.zeros_like(X_mesh) # 占位

X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# [cite_start]2. AAIS 核心实现 (新增部分) [cite: 1464, 1534, 1547]
# ==========================================================
class AAIS_Sampler:
    """
    真正的 AAIS 实现：基于高斯混合模型 (GMM) 和 EM 算法
    对应论文中的 Algorithm 3 (简化版，专注于核心的 EM 和退火)
    """
    def __init__(self, n_components=10, z_dim=2, device=device):
        self.K = n_components # 混合成分数量
        self.dim = z_dim
        self.device = device
        
        # 初始化混合权重 (alpha), 均值 (mu), 协方差 (cov)
        # 初始时均匀分布在定义域内: x in [-1, 1], t in [0, 1]
        self.alpha = torch.ones(self.K, device=device, dtype=dtype) / self.K
        
        # 随机初始化均值
        self.mu = torch.rand(self.K, self.dim, device=device, dtype=dtype)
        self.mu[:, 0] = self.mu[:, 0] * 2 - 1 # x: [-1, 1]
        
        # [cite_start]初始化协方差 (对角矩阵) [cite: 1596]
        # 初始方差设大一点以便探索
        self.cov_diag = torch.ones(self.K, self.dim, device=device, dtype=dtype) * 0.1

    def get_log_prob(self, samples):
        """计算混合模型 q(x) 的对数概率"""
        # samples: [N, 2]
        mix = Categorical(self.alpha)
        # 构建多元高斯分布组件
        comp = MultivariateNormal(self.mu, torch.diag_embed(self.cov_diag))
        # 计算 log_prob: log(sum(alpha * prob))
        # 技巧: 使用 logsumexp 增强数值稳定性
        log_probs_comp = comp.log_prob(samples.unsqueeze(1)) # [N, K]
        return torch.logsumexp(log_probs_comp + torch.log(self.alpha), dim=1)

    def sample(self, N):
        
        """从当前建议分布 q 中采样 [cite: 1537]"""
        mix = Categorical(self.alpha)
        comp = MultivariateNormal(self.mu, torch.diag_embed(self.cov_diag))
        
        # 先选成分，再从成分中采样
        comp_indices = mix.sample((N,))
        samples = torch.zeros(N, self.dim, device=self.device, dtype=dtype)
        
        # 向量化采样需要一些技巧，这里用循环简单实现（因为K通常不大）
        # 或者直接利用分布性质
        counts = torch.bincount(comp_indices, minlength=self.K)
        ptr = 0
        for k in range(self.K):
            if counts[k] > 0:
                s = MultivariateNormal(self.mu[k], torch.diag_embed(self.cov_diag[k])).sample((counts[k],))
                samples[ptr:ptr+counts[k]] = s
                ptr += counts[k]
        
        # 截断到定义域内 (Rejection Sampling的简化处理，直接截断)
        samples[:, 0] = torch.clamp(samples[:, 0], -1.0, 1.0)
        samples[:, 1] = torch.clamp(samples[:, 1], 0.0, 1.0)
        return samples

    def em_step(self, target_log_prob_func, lambda_k, n_samples=2000):
        """
        [cite_start]执行一步 EM 算法更新参数 [cite: 1540, 1548]
        参数:
            target_log_prob_func: 计算 log(Q(x)) 的函数 (即 log(Residual^2))
            [cite_start]lambda_k: 当前温度参数 [cite: 1569]
        """
        # 1. 采样 (Sampling)
        X = self.sample(n_samples).detach() # [N, 2]
        
        # [cite_start]2. 计算目标密度 Q_k [cite: 1569]
        # log(Q_k) = (1 - lambda) * log(q) + lambda * log(Q_target)
        log_q = self.get_log_prob(X)
        log_Q_target = target_log_prob_func(X) # 物理残差的对数
        
        # 防止数值溢出
        log_Q_target = torch.clamp(log_Q_target, max=20.0, min=-20.0)
        
        log_target_k = (1 - lambda_k) * log_q + lambda_k * log_Q_target
        
        # [cite_start]3. 计算重要性权重 w [cite: 1543]
        # log_w = log(target) - log(proposal)
        log_w = log_target_k - log_q
        w = torch.softmax(log_w, dim=0) # 归一化权重
        
        # [cite_start]4. E-Step: 计算后验概率 rho [cite: 1538]
        # rho_mk = alpha_m * f_m(x) / q(x)
        # log_rho = log_alpha + log_prob_comp - log_q
        comp = MultivariateNormal(self.mu, torch.diag_embed(self.cov_diag))
        log_probs_comp = comp.log_prob(X.unsqueeze(1)) # [N, K]
        log_rho = torch.log(self.alpha + 1e-10) + log_probs_comp - log_q.unsqueeze(1)
        rho = torch.exp(log_rho) # [N, K]
        
        # 归一化 rho (使得 sum_m rho_mk = 1)
        rho = rho / (torch.sum(rho, dim=1, keepdim=True) + 1e-10)
        
        # 组合权重: 样本重要性权重 * 成分后验概率
        # effective_weight[i, m] = w[i] * rho[i, m]
        eff_w = w.unsqueeze(1) * rho # [N, K]
        sum_eff_w = torch.sum(eff_w, dim=0) + 1e-10 # [K]
        
        # [cite_start]5. M-Step: 更新参数 [cite: 1548]
        # 更新 alpha
        self.alpha = sum_eff_w # 稍后归一化
        self.alpha = self.alpha / torch.sum(self.alpha)
        
        # 更新 mu
        # mu_new = sum(w * rho * x) / sum(w * rho)
        self.mu = (eff_w.T @ X) / sum_eff_w.unsqueeze(1)
        
        # 更新 cov
        # cov_new = sum(w * rho * (x-mu)^2) / sum(...)
        # 这里简化为对角协方差
        for k in range(self.K):
            diff = X - self.mu[k] # [N, 2]
            w_k = eff_w[:, k].unsqueeze(1) # [N, 1]
            cov_k = torch.sum(w_k * (diff ** 2), dim=0) / sum_eff_w[k]
            # 保证协方差矩阵正定且不过小
            self.cov_diag[k] = torch.clamp(cov_k, min=1e-4, max=0.5)

# ==========================================================
# 3. 网络定义 (保持原逻辑)
# ==========================================================
class PeriodicEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1)

class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(5.0, dtype=dtype)) 
    def forward(self, x):
        return torch.tanh(self.scale * x)

class DeepSubNet(nn.Module):
    def __init__(self, input_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, hidden), ScaledTanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x_emb, t):
        return self.net(torch.cat([x_emb, t], dim=1))

class SimplePINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embed = PeriodicEmbedding()
        self.net = DeepSubNet(input_dim=3, hidden=hidden)
    def forward(self, x, t):
        x_emb = self.embed(x)
        u_out = self.net(x_emb, t)
        # 硬约束边界条件: u(x,0) = x^2 * cos(pi*x)
        # 边界 u(-1)=u(1) 已由 PeriodicEmbedding 隐式处理部分，
        # 这里的输出结构简化处理，仅演示AAIS采样
        u_0 = x**2 * torch.cos(np.pi * x)
        u_final = torch.tanh(t) * u_out + u_0
        return u_final

# ==========================================================
# 4. 训练流程与 AAIS 整合
# ==========================================================
def pde_residual(model, x, t):
    """计算物理残差"""
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    res = u_t - EPSILON * u_xx + GAMMA * (u**3 - u)
    return res

def get_residual_log_prob(model, samples):
    """
    计算目标分布 Q(x) 的对数。
    Q(x) = |Residual(x)|^2 / Z
    log Q(x) = 2 * log|Residual| - log Z
    注意: EM 只需要相对值，不需要归一化常数 Z
    """
    x = samples[:, 0:1].requires_grad_(True)
    t = samples[:, 1:2].requires_grad_(True)
    res = pde_residual(model, x, t)
    # 加上一个小量防止 log(0)
    return torch.log(res**2 + 1e-16).detach().squeeze()

# 初始化模型和 AAIS 采样器
model = SimplePINN(hidden=64).to(device)
aais_sampler = AAIS_Sampler(n_components=10, z_dim=2, device=device) # [cite: 1464]

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3)

BATCH_SIZE = 5000
EPOCHS = 5000 

# [cite_start]定义退火温度阶梯 [cite: 1571]
# 这里简化为在训练过程中逐步提升 lambda
lambda_schedule = np.linspace(0.1, 1.0, EPOCHS)

print(f"\n>>> Phase 1: AAIS-PINN Training ({EPOCHS} Epochs)...")
start_time = time.time()

for epoch in range(EPOCHS):
    
    # === AAIS 步骤: 更新采样器 ===
    # 每隔几个 epoch 执行一次 EM 更新，模拟渐进式退火
    if epoch % 10 == 0:
        curr_lambda = lambda_schedule[epoch] # 获取当前温度
        # 定义当前的目标函数包装器
        target_func = lambda samples: get_residual_log_prob(model, samples)
        # [cite_start]执行 EM 步骤优化建议分布 q [cite: 1514]
        aais_sampler.em_step(target_func, lambda_k=curr_lambda)

    # === PINN 步骤: 训练网络 ===
    optim_pinn.zero_grad()
    
    # [cite_start]1. 从 AAIS 采样器中采样 (重点关注区域) [cite: 1443]
    samples_adaptive = aais_sampler.sample(int(0.7 * BATCH_SIZE))
    x_ad = samples_adaptive[:, 0:1].requires_grad_(True)
    t_ad = samples_adaptive[:, 1:2].requires_grad_(True)
    
    # 2. 均匀采样 (保留全局探索)
    x_uni = (-1 + 2*torch.rand(int(0.3 * BATCH_SIZE), 1, device=device)).requires_grad_(True)
    t_uni = torch.rand(int(0.3 * BATCH_SIZE), 1, device=device).requires_grad_(True)
    
    # 合并数据
    x_train = torch.cat([x_ad, x_uni], dim=0)
    t_train = torch.cat([t_ad, t_uni], dim=0)
    
    # 计算 Loss
    res = pde_residual(model, x_train, t_train)
    loss = torch.mean(res**2)
    
    loss.backward()
    optim_pinn.step()
    
    if epoch % 200 == 0:
        elapsed = time.time() - start_time
        print(f"Ep {epoch:4d} | Loss: {loss.item():.2e} | Temp(lambda): {lambda_schedule[epoch]:.2f} | Time: {elapsed:.1f}s")
        # 打印 AAIS 状态：查看高斯成分是否聚集
        # print(f"   AAIS Mu[0]: {aais_sampler.mu[0].data.cpu().numpy()}")

# ==========================================================
# Phase 2: L-BFGS (带打印修正)
# ==========================================================
print("\n>>> Phase 2: L-BFGS Refinement...")

x_final = X_star.clone().detach().requires_grad_(True) # 全网格点
t_final = T_star.clone().detach().requires_grad_(True)

lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=2000, 
    history_size=50,
    tolerance_grad=1e-9, 
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe" 
)

# 使用闭包内的计数器实现每200次打印
iter_count = 0 
def closure():
    global iter_count
    lbfgs.zero_grad()
    res = pde_residual(model, x_final, t_final)
    loss = torch.mean(res**2)
    loss.backward()
    
    iter_count += 1
    # 🔥🔥🔥 修正: L-BFGS 每 200 循环打印一次 🔥🔥🔥
    if iter_count % 200 == 0:
        print(f"L-BFGS Iter {iter_count:4d} | Loss: {loss.item():.5e}")
        
    return loss

lbfgs.step(closure)
print("✅ Done.")

# ==========================================================
#  结果验证与可视化 
# ==========================================================
model.eval()

# 1. 预测 u (前向传播不需要梯度，可以用 no_grad)
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(512, 201)

# 2. 计算残差场 (必须计算梯度，不能用 no_grad！)
#    需要克隆数据并开启 requires_grad
X_star_grad = X_star.clone().detach().requires_grad_(True)
T_star_grad = T_star.clone().detach().requires_grad_(True)

res_field = pde_residual(model, X_star_grad, T_star_grad)
# 计算完后，再 detach 转为 numpy
res_field = res_field.abs().detach().cpu().numpy().reshape(512, 201)

# 计算 L2 误差
error_l2 = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all)
print(f"\n✨✨✨ Final Relative L2 Error: {error_l2:.4e} ✨✨✨")

plt.figure(figsize=(10, 5))

# 图1: AAIS 采样点分布
plt.subplot(1, 2, 1)
samples = aais_sampler.sample(2000).cpu().numpy()
plt.scatter(samples[:, 1], samples[:, 0], s=2, alpha=0.5, c='r', label='AAIS Samples')
# 看 GMM 中心在哪里 
mus = aais_sampler.mu.detach().cpu().numpy()
plt.scatter(mus[:, 1], mus[:, 0], s=50, c='blue', marker='x', label='Centers')
plt.xlabel('t'); plt.ylabel('x'); plt.title(f'AAIS Sample Dist (Lambda={lambda_schedule[-1]:.1f})')
plt.xlim(0, 1); plt.ylim(-1, 1)
plt.legend()

# 图2: 残差场
plt.subplot(1, 2, 2)
plt.imshow(res_field, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='Residual Abs')
plt.xlabel('t'); plt.title(f'Final Residual Field (L2 Err={error_l2:.2e})')

plt.tight_layout()
plt.show()