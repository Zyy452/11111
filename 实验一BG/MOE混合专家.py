import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import time

# ==========================================
# 0. 配置与环境
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用 float32 保证速度，如果需要极致精度可改 float64
dtype = torch.float32 
print(f"🔥 设备: {device} | 架构: DeepSeekMoE (Shared + Routed Experts)")

nu = 0.05 / np.pi 

# ==========================================
# 1. DeepSeekMoE 核心层定义
# ==========================================
class Expert(nn.Module):
    """ 单个专家网络：简单的 MLP """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # 物理问题常用 Tanh
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class DeepSeekMoELayer(nn.Module):
    def __init__(self, hidden_dim, num_total_experts=8, num_shared=2, top_k=2):
        super().__init__()
        self.num_total_experts = num_total_experts
        self.num_shared = num_shared
        self.num_routed = num_total_experts - num_shared
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        # 1. 共享专家 (Shared Experts) - 永远激活
        self.shared_experts = nn.ModuleList([
            Expert(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_shared)
        ])

        # 2. 路由专家 (Routed Experts) - 待选
        self.routed_experts = nn.ModuleList([
            Expert(hidden_dim, hidden_dim, hidden_dim) for _ in range(self.num_routed)
        ])

        # 3. 门控网络 (Router)
        self.router = nn.Linear(hidden_dim, self.num_routed)

    def forward(self, x):
        # --- A. 共享专家路径 (Common Knowledge) ---
        shared_out = 0
        for expert in self.shared_experts:
            shared_out += expert(x)
        
        # --- B. 路由专家路径 (Specialized Tasks) ---
        # 计算路由分数
        router_logits = self.router(x) # [batch, num_routed]
        routing_weights = F.softmax(router_logits, dim=1)
        
        # 选出 Top-K
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        
        # 归一化权重 (让被选中的专家权重和为1，或者是原始softmax值)
        # DeepSeek 论文中通常直接使用 softmax 值，或者重新归一化
        # 这里为了梯度稳定，直接使用原始 softmax 权重
        
        routed_out = torch.zeros_like(x)
        
        # ⚠️ 注意：这里为了代码可读性使用了循环。
        # 在大规模 LLM 中会使用 torch.gather/scatter 进行并行优化。
        # 对于 PINN 这种规模，循环完全没问题。
        
        # 遍历所有路由专家，如果在 TopK 里就计算
        # (实际实现：全算或是这就根据 batch 索引算，这里简化为加权求和)
        
        # 为了演示清晰，我们这里计算所有专家，然后用 mask 过滤
        # 这样你可以看到“没被选中的专家”贡献确实为 0
        
        batch_size = x.shape[0]
        
        # 创建一个 mask [batch, num_routed]
        mask = torch.zeros(batch_size, self.num_routed, device=x.device)
        mask.scatter_(1, top_k_indices, 1.0) # 把 Top-K 的位置置为 1
        
        # 加权求和
        for i, expert in enumerate(self.routed_experts):
            expert_output = expert(x)
            # 只有当该专家在某样本的 Top-K 中时，mask[:, i] 才为 1
            # 权重是 routing_weights[:, i]
            weight = routing_weights[:, i] * mask[:, i]
            routed_out += weight.unsqueeze(1) * expert_output

        # --- C. 结果融合 ---
        return shared_out + routed_out

class DeepSeekPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 64) # x, t -> hidden
        
        # 堆叠 DeepSeekMoE 层
        self.moe_layers = nn.ModuleList([
            DeepSeekMoELayer(hidden_dim=64, num_total_experts=6, num_shared=2, top_k=2),
            DeepSeekMoELayer(hidden_dim=64, num_total_experts=6, num_shared=2, top_k=2),
            DeepSeekMoELayer(hidden_dim=64, num_total_experts=6, num_shared=2, top_k=2)
        ])
        
        self.output_layer = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        h = self.tanh(self.input_layer(xt))
        
        for layer in self.moe_layers:
            # 残差连接 (Residual Connection) 防止梯度消失
            h_moe = layer(h)
            h = self.tanh(h + h_moe) 
            
        u = self.output_layer(h)
        return u
    
    # 辅助函数：提取路由器的激活状态，用于画图
    def get_router_activation(self, x, t):
        xt = torch.cat([x, t], dim=1)
        h = self.tanh(self.input_layer(xt))
        activations = []
        
        for layer in self.moe_layers:
            # 重新计算一遍路由权重
            router_logits = layer.router(h)
            routing_weights = F.softmax(router_logits, dim=1)
            activations.append(routing_weights.detach().cpu())
            
            h_moe = layer(h)
            h = self.tanh(h + h_moe)
            
        return activations

# ==========================================
# 2. 精确解与数据 (同之前)
# ==========================================
def burgers_exact_scalar(x, t):
    if t == 0: return -np.sin(np.pi * x)
    def f(y): return np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
    def g(y): return np.exp(-(x - y)**2 / (4 * nu * t))
    res_num, _ = quad(lambda y: np.sin(np.pi * y) * f(y) * g(y), -1, 1, limit=100)
    res_den, _ = quad(lambda y: f(y) * g(y), -1, 1, limit=100)
    return -res_num / res_den

def get_data(batch_size=2000):
    # 混合采样：全域 + 激波附近
    x_rand = (torch.rand(batch_size, 1, device=device, dtype=dtype)*2 - 1).requires_grad_(True)
    t_rand = torch.rand(batch_size, 1, device=device, dtype=dtype).requires_grad_(True)
    
    # 激波加密
    x_shock = (torch.rand(1000, 1, device=device, dtype=dtype)*0.4 - 0.2).requires_grad_(True)
    t_shock = torch.rand(1000, 1, device=device, dtype=dtype).requires_grad_(True)
    
    x_f = torch.cat([x_rand, x_shock], dim=0)
    t_f = torch.cat([t_rand, t_shock], dim=0)
    
    # IC/BC
    x_ic = (torch.rand(1000, 1, device=device, dtype=dtype)*2 - 1)
    u_ic = -torch.sin(np.pi * x_ic)
    x_bc = torch.cat([torch.ones(500,1,device=device)*-1, torch.ones(500,1,device=device)*1], dim=0)
    t_bc = torch.rand(1000, 1, device=device, dtype=dtype)
    
    return x_f, t_f, x_ic, u_ic, x_bc, t_bc

# ==========================================
# 3. 训练流程
# ==========================================
model = DeepSeekPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def compute_loss(x, t, x_ic, u_ic, x_bc, t_bc):
    u = model(x, t)
    u_t, u_x = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    res = u_t + u * u_x - nu * u_xx
    loss_res = torch.mean(res**2)
    loss_ic = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - u_ic)**2)
    loss_bc = torch.mean(model(x_bc, t_bc)**2) # 假设边界为0 (近似)
    
    return loss_res + 100*loss_ic + 100*loss_bc

print(">>> DeepSeekMoE 启动训练...")
loss_history = []
start_time = time.time()

# 快速训练 3000 epoch 看看效果
for epoch in range(3001):
    x_f, t_f, xi, ui, xb, tb = get_data()
    
    optimizer.zero_grad()
    loss = compute_loss(x_f, t_f, xi, ui, xb, tb)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4e}")

print(f"训练耗时: {time.time()-start_time:.1f}s")

# ==========================================
# 4. L-BFGS 精修 (可选)
# ==========================================
print(">>> L-BFGS 精修...")
x_fix, t_fix, xi_fix, ui_fix, xb_fix, tb_fix = get_data(batch_size=5000)
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000, line_search_fn="strong_wolfe")

def closure():
    lbfgs.zero_grad()
    l = compute_loss(x_fix, t_fix, xi_fix, ui_fix, xb_fix, tb_fix)
    l.backward()
    return l
lbfgs.step(closure)

# ==========================================
# 5. 可视化：解 + 专家激活热力图
# ==========================================
# 生成精确解
x_val = np.linspace(-1, 1, 200)
u_true = np.array([burgers_exact_scalar(xi, 0.5) for xi in x_val])
x_t = torch.tensor(x_val, dtype=dtype, device=device).view(-1, 1)
with torch.no_grad():
    u_pred = model(x_t, torch.ones_like(x_t)*0.5).cpu().numpy().flatten()
    # 获取最后一层MoE的路由状态
    router_acts = model.get_router_activation(x_t, torch.ones_like(x_t)*0.5)
    last_layer_acts = router_acts[-1].numpy() # [200, 4] (4个路由专家)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 解对比
ax[0].plot(x_val, u_true, 'k-', linewidth=3, label='Exact')
ax[0].plot(x_val, u_pred, 'r--', linewidth=2.5, label='DeepSeekMoE')
ax[0].set_title(f"Result (t=0.5) | L2 Error: {np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true):.2%}")
ax[0].legend()

# 图2: 路由专家激活热力图
# 我们画出最后那一层 4 个路由专家在 x 轴上的权重分布
for i in range(last_layer_acts.shape[1]):
    ax[1].plot(x_val, last_layer_acts[:, i], label=f'Routed Expert {i+1}')
ax[1].set_title("Activation of Routed Experts (t=0.5)")
ax[1].set_xlabel("x")
ax[1].set_ylabel("Gate Probability")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

# 图3: 共享专家 vs 路由专家 概念图
ax[2].text(0.5, 0.7, "Shared Experts\n(Always Active)", ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", fc="cyan", ec="b"))
ax[2].text(0.5, 0.3, "Routed Experts\n(Top-K Active)", ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", fc="yellow", ec="orange"))
ax[2].annotate("", xy=(0.5, 0.6), xytext=(0.5, 0.4), arrowprops=dict(arrowstyle="->"))
ax[2].set_title("DeepSeekMoE Architecture")
ax[2].axis('off')

plt.tight_layout()
plt.show()