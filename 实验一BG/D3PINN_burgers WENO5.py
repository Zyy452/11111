import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import math

# ==========================================================
# 0. 设备配置与环境初始化
# ==========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用设备: MPS (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用设备: CUDA")
else:
    device = torch.device("cpu")
    print("使用设备: CPU")

torch.manual_seed(42)
np.random.seed(42)
if device.type == "mps":
    torch.mps.manual_seed(42)

# ==========================================================
# 1. 读取真解 (用于对比评估)
# ==========================================================
data_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat" # 请确保路径正确
data = loadmat(data_path)
x_exact = data["x"].flatten()     # shape [256]
t_exact = data["t"].flatten()     # shape [100]
Exact = data["usol"]              # shape [256, 100]

# ==========================================================
# 2. 物理参数与网格设置
# ==========================================================
nu = 0.01 / math.pi
M = 256  # 空间点数，与 Exact 一致
dx = x_exact[1] - x_exact[0]
T = 1.0  # 模拟总时长，与基准线对齐

# 初始条件 u(x,0)
u0 = -np.sin(np.pi * x_exact).astype(np.float64)

# RK4 时间步长 (为了稳定性，步长需满足 CFL 条件)
dt = 0.0005 
nsteps = int(T / dt)

# ==========================================================
# 3. WENO5 对流项格式 (固定算法)
# ==========================================================
def flux(u): return 0.5 * u**2

def weno5_left(v0, v1, v2, v3, v4):
    p0 = (1/3)*v0 - (7/6)*v1 + (11/6)*v2
    p1 = -(1/6)*v1 + (5/6)*v2 + (1/3)*v3
    p2 = (1/3)*v2 + (5/6)*v3 - (1/6)*v4
    b0, b1, b2 = (13/12)*(v0-2*v1+v2)**2+0.25*(v0-4*v1+3*v2)**2, \
                 (13/12)*(v1-2*v2+v3)**2+0.25*(v1-v3)**2, \
                 (13/12)*(v2-2*v3+v4)**2+0.25*(3*v2-4*v3+v4)**2
    eps_w = 1e-6
    a = np.array([0.1/(eps_w+b0)**2, 0.6/(eps_w+b1)**2, 0.3/(eps_w+b2)**2])
    w = a / np.sum(a)
    return w[0]*p0 + w[1]*p1 + w[2]*p2

def weno5_right(v0, v1, v2, v3, v4): return weno5_left(v4, v3, v2, v1, v0)

def weno5_flux_x(u):
    Mlen = len(u)
    f = flux(u)
    f_ext = np.pad(f, (3, 3), 'edge')
    u_ext = np.pad(u, (3, 3), 'edge')
    alpha = max(np.max(np.abs(u)), 1e-6)
    f_p = 0.5*(flux(u_ext) + alpha * u_ext)
    f_m = 0.5*(flux(u_ext) - alpha * u_ext)
    
    flux_iface = np.zeros(Mlen + 1)
    for i in range(Mlen + 1):
        k = i + 3
        fp = weno5_left(f_p[k-3], f_p[k-2], f_p[k-1], f_p[k], f_p[k+1])
        fm = weno5_right(f_m[k+2], f_m[k+1], f_m[k], f_m[k-1], f_m[k-2])
        flux_iface[i] = fp + fm
    return (flux_iface[1:] - flux_iface[:-1]) / dx

# ==========================================================
# 4. 离散算子神经网络 (对齐基准线参数)
# ==========================================================
class StencilNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # 对齐基准线的 [64, 64, 64, 64] 结构
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

model = StencilNN(hidden=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 对齐学习率

def build_stencils(u_arr):
    st = [u_arr[i-2:i+3] for i in range(2, len(u_arr)-2)]
    return torch.tensor(np.array(st), dtype=torch.float32, device=device)

def fd_u_xx(u):
    uxx = np.zeros_like(u)
    uxx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    return uxx

# ==========================================================
# 5. 预训练与在线修正 (微训练)
# ==========================================================
print("正在预训练算子网络...")
st_init = build_stencils(u0)
target_init = torch.tensor(fd_u_xx(u0)[2:-2].reshape(-1,1), dtype=torch.float32, device=device)

for ep in range(2000):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(st_init), target_init)
    loss.backward(); optimizer.step()

# ==========================================================
# 6. RK4 时间步进模拟
# ==========================================================
u_d3 = u0.copy()
start_t = time.time()

def get_u_t(u_curr):
    fx = weno5_flux_x(u_curr)
    # NN 预测 u_xx
    st = build_stencils(u_curr)
    with torch.no_grad():
        uxx_inner = model(st).cpu().numpy().flatten()
    uxx = np.zeros_like(u_curr)
    uxx[2:-2] = uxx_inner
    uxx[0], uxx[1], uxx[-1], uxx[-2] = fd_u_xx(u_curr)[[0,1,-1,-2]] # 边界用差分
    return -fx + nu * uxx

print(f"开始时间步进计算 (总步数: {nsteps})...")
for n in range(1, nsteps + 1):
    # RK4 
    k1 = get_u_t(u_d3)
    k2 = get_u_t(u_d3 + 0.5*dt*k1)
    k3 = get_u_t(u_d3 + 0.5*dt*k2)
    k4 = get_u_t(u_d3 + dt*k3)
    u_d3 += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # 在线微训练 (保持 NN 对物理算子的敏感度)
    if n % 50 == 0:
        st_online = build_stencils(u_d3)
        target_online = torch.tensor(fd_u_xx(u_d3)[2:-2].reshape(-1,1), dtype=torch.float32, device=device)
        optimizer.zero_grad()
        nn.MSELoss()(model(st_online), target_online).backward()
        optimizer.step()

# ==========================================================
# 7. 最终评估：Relative L2 Error
# ==========================================================
u_true_final = Exact[:, -1] # 真解在 t=1.0 的值
error_l2 = np.linalg.norm(u_true_final - u_d3, 2) / np.linalg.norm(u_true_final, 2)

print(f"\n--- Final Result ---")
print(f"D3PINN (Code B) Relative L2 Error at t=1.0: {error_l2:.3e}")
print(f"计算总耗时: {time.time() - start_t:.2f}s")

# 可视化对比
plt.figure(figsize=(8, 5))
plt.plot(x_exact, u_true_final, 'k-', label="True (t=1.0)")
plt.plot(x_exact, u_d3, 'r--', label="D3PINN (t=1.0)")
plt.title(f"D3PINN Result Comparison (Error: {error_l2:.2e})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()