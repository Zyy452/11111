import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ==========================================================
# 1. 读取 Burgers 方程真解（用经典数据集）
# ==========================================================

data = loadmat("/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat")
x = data["x"].flatten()             # shape [256]
t = data["t"].flatten()             # shape [100]
Exact = data["usol"]                # shape [256, 100]

X, T = np.meshgrid(x, t)

# ==========================================================
# 2. PINN 网络
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
        return self.net(torch.cat([x, t], dim=1))

model = MLP()

# viscosity
nu = 0.01 / np.pi

# ==========================================================
# 3. 训练数据
# ==========================================================
N_r = 20000
x_r = -1 + 2 * torch.rand(N_r, 1)
t_r = torch.rand(N_r, 1)

# initial condition u(x,0)
N_ic = 200
x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# boundary BC
N_bc = 200
t_bc = torch.rand(N_bc, 1)
x_bc_L = -1 * torch.ones_like(t_bc)
x_bc_R =  1 * torch.ones_like(t_bc)
u_L = torch.zeros_like(t_bc)
u_R = torch.zeros_like(t_bc)

# ==========================================================
# 4. PDE residual
# ==========================================================
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0]

    return u_t + u*u_x - nu*u_xx

# ==========================================================
# 5. 训练
# ==========================================================
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30000):

    optimizer.zero_grad()

    # PDE loss
    r = pde_residual(model, x_r, t_r)
    loss_r = loss_fn(r, torch.zeros_like(r))

    # IC loss
    loss_ic = loss_fn(model(x_ic, t_ic), u_ic)

    # BC loss
    loss_bc = loss_fn(model(x_bc_L, t_bc), u_L) + \
              loss_fn(model(x_bc_R, t_bc), u_R)

    loss = loss_r + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.3e}, PDE={loss_r:.3e}")

# ==========================================================
# 6. 因果性破坏可视化（核心）
# ==========================================================
times = [0.0, 0.25, 0.50, 0.75, 1.0]

xx = np.linspace(-1, 1, 256)
xx_t = torch.tensor(xx, dtype=torch.float32).view(-1,1)

plt.figure(figsize=(14,10))

for i, t0 in enumerate(times):
    tt_t = torch.ones_like(xx_t)*t0
    
    pred = model(xx_t, tt_t).detach().numpy()
    true = Exact[:, int(t0*99)]       # 抽取真实解

    plt.subplot(2,3,i+1)
    plt.plot(xx, true, "k-", label="True")
    plt.plot(xx, pred, "r--", label="PINN")
    plt.title(f"t = {t0}")
    plt.legend()

plt.tight_layout()
plt.show()
