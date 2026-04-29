import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. 构造 Heat Equation 真解
# ==========================================================
def u_true(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

# ==========================================================
# 2. PINN 网络定义
# ==========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

model = MLP()

# ==========================================================
# 3. 训练数据：内部点 + 初始条件
# ==========================================================
N_r = 2000
x_r = torch.rand(N_r, 1)
t_r = torch.rand(N_r, 1)

N_ic = 200
x_ic = torch.linspace(0, 1, N_ic).view(-1, 1)
t_ic = torch.zeros_like(x_ic)
u_ic = torch.sin(np.pi * x_ic)  # u(x,0)=sin(pi x)

x_r = x_r.float()
t_r = t_r.float()
x_ic = x_ic.float()
t_ic = t_ic.float()
u_ic = u_ic.float()

# ==========================================================
# 4. 定义 PDE 残差
# ==========================================================
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)

    # 计算 u_t
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                             retain_graph=True, create_graph=True)[0]
    # 计算 u_x
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    # 计算 u_xx
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0]

    # 热方程 PDE 残差：u_t - u_xx = 0
    return u_t - u_xx

# ==========================================================
# 5. 优化器 & 损失函数
# ==========================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ==========================================================
# 6. 开始训练 PINN
# ==========================================================
for epoch in range(4000):
    optimizer.zero_grad()
    
    # PDE 残差损失
    r = pde_residual(model, x_r, t_r)
    loss_r = loss_fn(r, torch.zeros_like(r))
    
    # 初始条件损失
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    loss = loss_r + loss_ic
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# ==========================================================
# 7. 计算残差 r(x,t) 和真实误差 e(x,t)
# ==========================================================
N_test = 2000
x_test = np.random.rand(N_test, 1)
t_test = np.random.rand(N_test, 1)

x_t = torch.tensor(x_test, dtype=torch.float32)
t_t = torch.tensor(t_test, dtype=torch.float32)

# PINN 预测值
u_pred = model(x_t, t_t).detach().numpy()

# 真实值
u_truth = u_true(x_test, t_test)

# 真实误差
error = np.abs(u_pred - u_truth)

# 计算 PDE 残差（绝对值）
r_val = pde_residual(model, x_t, t_t).detach().numpy()
residual = np.abs(r_val)

# ==========================================================
# 8. Scatter plot 残差 vs 真实误差
# ==========================================================
plt.figure(figsize=(7,6))
plt.scatter(residual, error, s=5, alpha=0.4)
plt.xlabel("Residual |u_t - u_xx|")
plt.ylabel("Error |u_pred - u_true|")
plt.title("Residual vs True Error (Misalignment)")
plt.grid(True)
plt.show()
