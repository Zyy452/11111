import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os

# ==========================================================
# 0. 实验模式选择 (SCI 论文消融实验核心)
# ==========================================================
# 请修改此处的变量以运行不同的实验：
# 'RDES_ONLY' : 仅开启 RDES 残差演化采样 (Baseline 2)
# 'AB_ONLY'   : 仅开启动态区域分解，使用均匀采样 (Baseline 3)
# 'COMBINED'  : 综合框架：RDES 采样 + 动态区域分解 (Proposed Method)

EXPERIMENT_MODE = 'COMBINED'  

USE_RDES = EXPERIMENT_MODE in ['RDES_ONLY', 'COMBINED']
USE_DYNAMIC_AB = EXPERIMENT_MODE in ['AB_ONLY', 'COMBINED']

print(f"🔥 当前运行模式: {EXPERIMENT_MODE}")
print(f"   - RDES 采样: {'开启' if USE_RDES else '关闭'}")
print(f"   - 动态区域分解: {'开启' if USE_DYNAMIC_AB else '关闭'}")

# ==========================================================
# 1. 基础配置与数据读取
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(1234)
np.random.seed(1234)

DATA_PATH = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/KdV.mat"
SAVE_DIR = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

data = scipy.io.loadmat(DATA_PATH)
x_exact = data["x"].flatten()
t_exact = data["tt"].flatten()
u_exact = data["uu"].real.T  # 转置为 (201, 512)

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

x_min, x_max = x_exact.min(), x_exact.max()
t_min, t_max = t_exact.min(), t_exact.max()

# ==========================================================
# 2. 动态自适应基网络架构 (Dynamic AB-PINN)
# ==========================================================
class MLP(nn.Module):
    """标准的 MLP 子网络"""
    def __init__(self, layers=[2, 40, 40, 40, 1]):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.out = nn.Linear(layers[-2], layers[-1])
        
        # 初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        return self.out(self.net(torch.cat([x, t], dim=1)))

class DynamicABPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = MLP(layers=[2, 50, 50, 50, 50, 1]) # 全局基网络
        self.experts = nn.ModuleList()                     # 专家子网络列表
        self.centers = []                                  # 专家锚点坐标 (x, t)
        self.gamma = 10.0                                  # RBF 影响范围参数

    def forward(self, x, t):
        u_pred = self.base_net(x, t)
        
        # 如果没有开启区域分解，或者当前没有专家，直接返回基网络输出
        if not USE_DYNAMIC_AB or len(self.experts) == 0:
            return u_pred
            
        # 叠加专家网络的局部修正 (基于 RBF 距离门控)
        for i, expert in enumerate(self.experts):
            cx, ct = self.centers[i]
            dist_sq = (x - cx)**2 + (t - ct)**2
            weight = torch.exp(-self.gamma * dist_sq)
            u_pred = u_pred + weight * expert(x, t)
            
        return u_pred
        
    def add_expert(self, cx, ct):
        """在指定锚点增加一个新的专家网络，并零初始化"""
        new_expert = MLP(layers=[2, 30, 30, 30, 1]).to(device)
        # 零初始化输出层，实现平滑插入 (Zero-Init)
        nn.init.zeros_(new_expert.out.weight)
        nn.init.zeros_(new_expert.out.bias)
        self.experts.append(new_expert)
        self.centers.append((cx, ct))
        print(f"🌟 [架构生长] 增加第 {len(self.experts)} 个专家网络，锚点: x={cx:.3f}, t={ct:.3f}")

model = DynamicABPINN().to(device)

# ==========================================================
# 3. 物理残差与多尺度演化采样 (RDES)
# ==========================================================
LAMBDA_1, LAMBDA_2 = 1.0, 0.0025

def pde_residual(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

N_f = 15000  # 配置点总数
N_ic, N_bc = 512, 400

def generate_points(model, use_rdes=False):
    """根据模式生成配置点 (均匀 or 混合 RDES)"""
    if not use_rdes:
        # 纯均匀采样
        x_f = (torch.rand(N_f, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min)
        t_f = (torch.rand(N_f, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min)
        return x_f, t_f
    else:
        # RDES 多尺度混合采样 (30% 均匀 + 70% 残差驱动)
        N_uni = int(0.3 * N_f)
        N_res = N_f - N_uni
        
        # 1. 30% 保底集
        x_uni = (torch.rand(N_uni, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min)
        t_uni = (torch.rand(N_uni, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min)
        
        # 2. 70% 追踪集 (从 10万个密集候选点中筛选)
        N_pool = 100000
        x_pool = (torch.rand(N_pool, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min)
        t_pool = (torch.rand(N_pool, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min)
        
        with torch.no_grad():
            res_pool = torch.abs(pde_residual(model, x_pool, t_pool))
            # 概率密度 p ∝ |R|^2
            prob = (res_pool ** 2).flatten()
            prob = prob / torch.sum(prob)
            
        # 根据概率分布进行多项式重采样
        idx = torch.multinomial(prob, N_res, replacement=True)
        x_res, t_res = x_pool[idx], t_pool[idx]
        
        x_f = torch.cat([x_uni, x_res], dim=0).detach()
        t_f = torch.cat([t_uni, t_res], dim=0).detach()
        x_f.requires_grad_(True)
        t_f.requires_grad_(True)
        return x_f, t_f

# 初始网格点生成
x_f, t_f = generate_points(model, use_rdes=False) # 预热阶段用均匀采样
x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)

# ==========================================================
# 4. 训练与动态演化逻辑
# ==========================================================
def compute_loss(x_f, t_f):
    res = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(res**2)
    
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    # 周期边界省略，这里用简化的软约束模拟 (为了突出重点，您可自行补充导数项)
    t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    u_lb = model(x_min * torch.ones_like(t_bc), t_bc)
    u_ub = model(x_max * torch.ones_like(t_bc), t_bc)
    loss_bc = torch.mean((u_lb - u_ub)**2)
    
    return loss_f + 100.0 * loss_ic + 10.0 * loss_bc, res

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ADAM_ITERS = 15000
RESAMPLE_FREQ = 2000 # 每 2000 次迭代更新一次采样点和检查架构

print("\n>>> 开始训练...")
for it in range(ADAM_ITERS + 1):
    # 【核心逻辑 1：RDES 采样点更新】
    if it > 0 and it % RESAMPLE_FREQ == 0:
        x_f, t_f = generate_points(model, use_rdes=USE_RDES)
        
        # 【核心逻辑 2：动态架构检查与生长】
        if USE_DYNAMIC_AB:
            with torch.no_grad():
                current_res = torch.abs(pde_residual(model, x_f, t_f))
                max_res_idx = torch.argmax(current_res)
                max_res_val = current_res[max_res_idx].item()
                
                # 如果最大残差仍然很高，说明单网络学不动了，空投专家！
                if max_res_val > 0.05 and len(model.experts) < 5: 
                    cx = x_f[max_res_idx].item()
                    ct = t_f[max_res_idx].item()
                    model.add_expert(cx, ct)
                    # 动态添加网络后，需要更新优化器的参数列表
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss, _ = compute_loss(x_f, t_f)
    loss.backward()
    optimizer.step()
    
    if it % 1000 == 0:
        with torch.no_grad():
            u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
            err = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
        print(f"Iter {it:5d} | Loss: {loss.item():.5e} | Rel L2 Error: {err:.4f} | Experts: {len(model.experts)}")

# （后续 L-BFGS 和绘图逻辑与前文 Baseline 一致，此处由于篇幅限制略去绘图代码，
# 您可以直接将前一个回答中的 L-BFGS 和 plt 代码粘贴至此处）

print(f"✅ {EXPERIMENT_MODE} 训练完毕！")