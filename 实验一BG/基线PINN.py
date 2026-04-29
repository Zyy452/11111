import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# ==========================================================
# 0. 设备配置与环境初始化
# ==========================================================
# 检查并设置 MPS 设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用设备: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用设备: CUDA")
else:
    device = torch.device("cpu")
    print("使用设备: CPU")

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)
if device.type == "mps":
    torch.mps.manual_seed(42)

# ==========================================================
# 1. 读取 Burgers 方程真解
# ==========================================================
# 请确保路径正确，或将文件置于脚本同级目录
data_path = "/3241003007/zy/实验一BG/burgers_shock.mat"
data = loadmat(data_path)
x_exact = data["x"].flatten()             # shape [256]
t_exact = data["t"].flatten()             # shape [100]
Exact = data["usol"]                      # shape [256, 100]

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)

# ==========================================================
# 2. PINN 网络结构
# ==========================================================
class MLP(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 1]):
        super().__init__()
        net = []
        for i in range(len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, x, t):
        # 显式拼接
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# 实例化并移动到 MPS
model = MLP().to(device)

# 物理参数
nu = 0.01 / np.pi

# ==========================================================
# 3. 准备训练数据 (全部搬运至 device)
# ==========================================================
# 内部点 (Residual Points)
N_r = 20000
x_r = (-1 + 2 * torch.rand(N_r, 1, dtype=torch.float32)).to(device)
t_r = torch.rand(N_r, 1, dtype=torch.float32).to(device)

# 初始条件 (Initial Condition)
N_ic = 200
x_ic = torch.linspace(-1, 1, N_ic, dtype=torch.float32).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic = -torch.sin(np.pi * x_ic).to(device)

# 边界条件 (Boundary Condition)
N_bc = 200
t_bc = torch.rand(N_bc, 1, dtype=torch.float32).to(device)
x_bc_L = -1.0 * torch.ones_like(t_bc).to(device)
x_bc_R =  1.0 * torch.ones_like(t_bc).to(device)
u_L_val = torch.zeros_like(t_bc).to(device)
u_R_val = torch.zeros_like(t_bc).to(device)

# ==========================================================
# 4. PDE 残差计算函数
# ==========================================================
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)

    # 自动微分：MPS 支持 create_graph=True
    u_grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)
    u_x = u_grads[0]
    u_t = u_grads[1]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]

    return u_t + u * u_x - nu * u_xx

# ==========================================================
# 5. 训练循环
# ==========================================================
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start_time = time.time()
print("开始训练...")

for epoch in range(30001):
    optimizer.zero_grad()

    # PDE 损失
    res = pde_residual(model, x_r, t_r)
    loss_r = loss_fn(res, torch.zeros_like(res))

    # 初始条件损失
    loss_ic = loss_fn(model(x_ic, t_ic), u_ic)

    # 边界条件损失
    pred_bc_L = model(x_bc_L, t_bc)
    pred_bc_R = model(x_bc_R, t_bc)
    loss_bc = loss_fn(pred_bc_L, u_L_val) + loss_fn(pred_bc_R, u_R_val)

    # 总损失
    loss = loss_r + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | PDE: {loss_r.item():.3e} | 用时: {elapsed:.1f}s")

# ==========================================================
# 6. 结果可视化与误差计算
# ==========================================================
model.eval()

# 准备绘图数据
times = [0.0, 0.25, 0.50, 0.75, 1.0]
xx_plot = np.linspace(-1, 1, 256)
xx_plot_t = torch.tensor(xx_plot, dtype=torch.float32).view(-1, 1).to(device)

plt.figure(figsize=(15, 10))

for i, t0 in enumerate(times):
    tt_plot_t = torch.ones_like(xx_plot_t) * t0
    
    with torch.no_grad():
        u_pred_plot = model(xx_plot_t, tt_plot_t).cpu().numpy()
    
    # 获取真解索引（100个时间步对应 0.0 到 0.99）
    t_idx = int(t0 * 99)
    u_true_plot = Exact[:, t_idx]

    plt.subplot(2, 3, i+1)
    plt.plot(xx_plot, u_true_plot, "k-", label="Exact", linewidth=2)
    plt.plot(xx_plot, u_pred_plot, "r--", label="PINN", linewidth=2)
    plt.title(f"Time t = {t0}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.grid(True, alpha=0.3)

# ----------------------------------------------------------
# 7. 计算全时空域 Relative L2 Error
# ----------------------------------------------------------
# 构建推理网格
x_test = torch.tensor(X_mesh.flatten()[:, None], dtype=torch.float32).to(device)
t_test = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32).to(device)

with torch.no_grad():
    # 预测并转换维度对齐 Exact [256, 100]
    # meshgrid 生成的是 (100, 256)，扁平化预测后再转置
    u_pred_all = model(x_test, t_test).cpu().numpy().reshape(len(t_exact), len(x_exact)).T

error_l2 = np.linalg.norm(Exact - u_pred_all, 2) / np.linalg.norm(Exact, 2)
print(f"\n--- 训练完成 ---")
print(f"最终相对 L2 误差: {error_l2:.3e}")

plt.tight_layout()
plt.show()