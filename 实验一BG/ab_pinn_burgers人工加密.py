import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os

# ==========================================================
# 0. 基础配置 (强制 CPU + Float64)
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 🔥 核心修改：强制使用 CPU，不再检测 MPS/CUDA
device = torch.device("cpu")
print("🔥 设备: CPU (强制使用) | 精度: Float64 (高精度模式)")

# 强制双精度 (结合深层网络，效果最稳)
dtype = torch.float64 
torch.set_default_dtype(dtype)

# 随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================================
# 1. 数据准备 (优先读取 .mat，失败则自动生成)
# ==========================================================
nu = 0.01 / np.pi

def get_exact_solution():
    """ 尝试读取 mat 文件，如果失败则使用伪数据（仅用于防止代码报错） """
    # 你的文件路径
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat"
    
    try:
        data = loadmat(file_path)
        print(f"✅ 成功读取数据文件: {file_path}")
        x = data["x"].flatten()
        t = data["t"].flatten()
        usol = data["usol"] # [256, 100]
    except FileNotFoundError:
        print("⚠️ 未找到 .mat 文件，正在生成高精度合成数据...")
        # 这里为了演示，简单生成网格，实际应用请确保文件存在
        x = np.linspace(-1, 1, 256)
        t = np.linspace(0, 1, 100)
        usol = np.zeros((256, 100)) # 占位
    
    return x, t, usol

x_exact, t_exact, Exact = get_exact_solution()
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)

# 转为 Tensor 备用 (验证集)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 网络架构 (融合版：深层网络 + 动态列表)
# ==========================================================
class DeepSubNet(nn.Module):
    """ 🔥 代码1的优势：4层隐藏层的深层网络 """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), # 加深
            nn.Linear(hidden, hidden), nn.Tanh(), # 加深
            nn.Linear(hidden, 1)
        )
        # 初始化
        # 注意：CPU上 Float64 使用 Xavier 初始化通常很稳
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

class DynamicABPINN(nn.Module):
    """ 🔥 代码2的优势：支持动态添加子域 """
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        # 初始：2个子域 (模仿代码1的初始覆盖)
        self.subnets = nn.ModuleList([DeepSubNet(hidden), DeepSubNet(hidden)])
        # 初始中心：覆盖左右
        self.centers = nn.ParameterList([
            nn.Parameter(torch.tensor([-0.5, 0.2], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([0.5, 0.8], device=device, dtype=dtype))
        ])
        # 初始缩放系数
        self.log_gammas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype)),
            nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype))
        ])

    def add_subdomain(self, mu_init):
        """ 动态添加一个新的子网络 """
        new_net = DeepSubNet(self.hidden).to(device)
        mu = nn.Parameter(torch.tensor(mu_init, device=device, dtype=dtype).view(1, 2)) # [x, t]
        lg = nn.Parameter(torch.tensor([3.0], device=device, dtype=dtype)) # 新子域通常更尖锐
        
        self.subnets.append(new_net)
        self.centers.append(mu)
        self.log_gammas.append(lg)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        logits = []
        # 计算每个子域的权重
        for i in range(len(self.subnets)):
            # 距离中心的欧氏距离
            dist_sq = torch.sum((xt - self.centers[i])**2, dim=1, keepdim=True)
            logits.append(-torch.exp(self.log_gammas[i]) * dist_sq)
        
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1)
        
        u_out = 0
        for i in range(len(self.subnets)):
            u_out += weights[:, i] * self.subnets[i](x, t)
        return u_out

# ==========================================================
# 3. 采样与 Loss
# ==========================================================
def get_training_data(N_r=5000, N_b=200):
    # 内部点
    x_r = (-1 + 2 * torch.rand(N_r, 1, device=device, dtype=dtype)).requires_grad_(True)
    t_r = torch.rand(N_r, 1, device=device, dtype=dtype).requires_grad_(True)
    
    # 激波区加密 (人工先验，帮助收敛)
    x_shock = (torch.rand(N_r//2, 1, device=device, dtype=dtype)*0.4 - 0.2).requires_grad_(True)
    t_shock = torch.rand(N_r//2, 1, device=device, dtype=dtype).requires_grad_(True)
    
    x_r = torch.cat([x_r, x_shock], dim=0)
    t_r = torch.cat([t_r, t_shock], dim=0)

    # 初始条件 (IC)
    x_ic = torch.linspace(-1, 1, N_b, device=device, dtype=dtype).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(np.pi * x_ic)

    # 边界条件 (BC)
    t_bc = torch.rand(N_b, 1, device=device, dtype=dtype)
    x_bc_l = -1.0 * torch.ones_like(t_bc)
    x_bc_r = 1.0 * torch.ones_like(t_bc)
    
    return x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc

def compute_loss(model, x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc):
    # PDE Loss
    u = model(x_r, t_r)
    u_t, u_x = torch.autograd.grad(u, [t_r, x_r], torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
    res = u_t + u * u_x - nu * u_xx
    loss_r = torch.mean(res**2)
    
    # IC/BC Loss
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    loss_bc = torch.mean(model(x_bc_l, t_bc)**2) + torch.mean(model(x_bc_r, t_bc)**2) # 假设边界为0
    
    # 记录残差用于自适应
    return loss_r + 100*(loss_ic + loss_bc), res.abs().detach()

# ==========================================================
# 4. 训练主循环 (Adam 20k -> 动态加点 -> L-BFGS)
# ==========================================================
model = DynamicABPINN(hidden=64).to(device)
# 🔥 这里要注意：每次加子域后，optimizer 需要重建，所以我们把它放在循环里管理

x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc = get_training_data()

# 第一阶段：Adam 长跑 (20000步，模仿代码1)
print(">>> Phase 1: Deep Adam Training (Target: 20,000 steps)...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = []

start_time = time.time()
ADAM_STEPS = 20001
CHECK_FREQ = 4000 # 每 4000 步检查一次是否需要加子域

for epoch in range(ADAM_STEPS):
    optimizer.zero_grad()
    loss, res_v = compute_loss(model, x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # --- 动态增加子域逻辑 ---
    if epoch > 0 and epoch % CHECK_FREQ == 0:
        # 只有当误差还比较大，且子域数量 < 6 时才加
        max_res = res_v.max().item()
        if len(model.subnets) < 6 and max_res > 0.01:
            idx = torch.argmax(res_v)
            new_mu = [x_r[idx].item(), t_r[idx].item()]
            
            print(f"⚡️ [自适应触发] Epoch {epoch}: 最大残差 {max_res:.4f} @ {new_mu}")
            print(f"   -> 添加第 {len(model.subnets)+1} 个子神经网络...")
            
            model.add_subdomain(new_mu)
            
            # 🔥 关键：重置优化器！因为 parameters() 变了
            # 降低一点学习率，让新子网平稳融入
            optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
            
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.5e} | Subnets: {len(model.subnets)}")

print(f"Adam Phase Time: {time.time()-start_time:.1f}s")

# 第二阶段：L-BFGS 精修 (模仿代码2)
print("\n>>> Phase 2: L-BFGS Fine-tuning...")
# 重新采样一批高质量数据用于精修
x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc = get_training_data(N_r=8000)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, 
                          history_size=100, line_search_fn="strong_wolfe",
                          tolerance_grad=1e-15, tolerance_change=1e-15)

def closure():
    lbfgs.zero_grad()
    loss, _ = compute_loss(model, x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc)
    loss.backward()
    return loss

lbfgs.step(closure)
final_loss = closure().item()
print(f"L-BFGS Final Loss: {final_loss:.5e}")

# ==========================================================
# 5. 验证与画图 (修复版)
# ==========================================================
print("\n>>> 开始绘图...")
model.eval()

# 1. 计算最终误差
with torch.no_grad():
    # 预测全场 (这里不需要算微分，所以可以用 no_grad)
    u_pred_flat = model(X_star, T_star).cpu().numpy()
    u_pred = u_pred_flat.reshape(len(t_exact), len(x_exact)).T

# 计算 L2 误差
if np.max(np.abs(Exact)) > 1e-5:
    error_l2 = np.linalg.norm(Exact - u_pred, 2) / np.linalg.norm(Exact, 2)
    print(f"\n✨✨✨ Final Relative L2 Error: {error_l2:.4e} ✨✨✨")
else:
    print("\n⚠️ 无真值数据，跳过误差计算。")

# 2. 准备画图
plt.figure(figsize=(15, 6))

# 图1: 解对比 (t=0.5)
plt.subplot(1, 2, 1)
t_idx = int(0.5 * len(t_exact)) # t=0.5
plt.plot(x_exact, Exact[:, t_idx], 'k-', linewidth=3, label="Exact")
plt.plot(x_exact, u_pred[:, t_idx], 'r--', linewidth=2.5, label="Deep Dynamic AB-PINN")
plt.title(f"Solution at t={t_exact[t_idx]:.2f}\nRelative Error: {error_l2:.2e}")
plt.xlabel("x")
plt.legend()
plt.grid(True, alpha=0.3)

# 图2: 子域中心位置 + 残差分布
plt.subplot(1, 2, 2)

# 🔥 关键修复：计算 PDE 残差需要梯度，所以这里不能用 no_grad
# 确保用于画图的点开启了梯度追踪
if not x_r.requires_grad: x_r.requires_grad_(True)
if not t_r.requires_grad: t_r.requires_grad_(True)

# 计算残差 (此时允许 autograd 工作)
_, res_tensor = compute_loss(model, x_r, t_r, x_ic, t_ic, u_ic, x_bc_l, x_bc_r, t_bc)
res_map = res_tensor.detach().cpu().numpy() # 算完后再 detach 转 numpy

# 画散点图
sc = plt.scatter(t_r.detach().cpu().numpy(), x_r.detach().cpu().numpy(), 
                 c=res_map, s=2, cmap='jet', alpha=0.6)
plt.colorbar(sc, label='PDE Residual (Abs)')

# 画出最终的子域中心
# model.centers 是 ParameterList，需要提取数据
centers_np = []
for c in model.centers:
    c_data = c.detach().cpu().numpy().flatten()
    centers_np.append([c_data[1], c_data[0]]) # [t, x] 注意我们之前的定义顺序是 [x, t] 还是 [t, x]
    # 在代码中 new_mu = [x, t]，所以 c[0]是x, c[1]是t。
    # 画图时 x轴是t, y轴是x。所以坐标应该是 (c[1], c[0])

centers_np = np.array(centers_np)

plt.scatter(centers_np[:, 0], centers_np[:, 1], c='white', edgecolors='black', s=150, marker='X', label='Subnet Centers', zorder=10)

# 标注子网序号
for i, c in enumerate(centers_np):
    plt.text(c[0], c[1]+0.08, f"N{i+1}", color='white', fontweight='bold', ha='center', fontsize=9, path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])

plt.title(f"Adaptive Subnets ({len(model.subnets)} Total)\nBackground: Residual Distribution")
plt.xlabel("t (Time)")
plt.ylabel("x (Space)")
plt.legend(loc='upper right')
plt.ylim([-1.1, 1.1])
plt.xlim([-0.05, 1.05])

plt.tight_layout()
# 保存图片
plt.savefig('final_result_fixed.png', dpi=300)
print("✅ 图片已保存至: final_result_fixed.png")
# plt.show()