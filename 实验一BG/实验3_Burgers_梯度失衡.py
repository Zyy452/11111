import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. PINN 网络结构
# ==========================================================
class MLP(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super().__init__()
        l = []
        for i in range(len(layers)-1):
            l.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                l.append(nn.Tanh())
        self.net = nn.Sequential(*l)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

model = MLP()

# ==========================================================
# 2. Burgers PDE 参数
# ==========================================================
nu = 0.01 / np.pi    # viscosity

# ==========================================================
# 3. 训练数据
# ==========================================================
N_r = 2000
x_r = -1 + 2 * torch.rand(N_r, 1)
t_r = torch.rand(N_r, 1)

# initial condition u(x,0) = -sin(pi x)
N_ic = 200
x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# 转为 float
x_r = x_r.float()
t_r = t_r.float()
x_ic = x_ic.float()
t_ic = t_ic.float()
u_ic = u_ic.float()

# ==========================================================
# 4. PDE Residual
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

    # Burgers PDE:
    # u_t + u u_x = nu u_xx
    return u_t + u * u_x - nu * u_xx

# ==========================================================
# 5. 训练配置
# ==========================================================
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

grad_r_list = []
grad_ic_list = []
ratio_list = []
epochs = []

# 梯度范数计算
def get_grad_norm(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    return torch.norm(torch.cat(grads)).item()

# ==========================================================
# 6. Training Loop
# ==========================================================
for epoch in range(4000):

    # --------------------------
    # PDE gradient
    # --------------------------
    optimizer.zero_grad()
    r = pde_residual(model, x_r, t_r)
    loss_r = loss_fn(r, torch.zeros_like(r))
    loss_r.backward(retain_graph=True)
    grad_r = get_grad_norm(model)

    # --------------------------
    # IC gradient
    # --------------------------
    optimizer.zero_grad()
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    loss_ic.backward(retain_graph=True)
    grad_ic = get_grad_norm(model)

    # --------------------------
    # train both
    # --------------------------
    optimizer.zero_grad()
    (loss_r + loss_ic).backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss_r={loss_r.item():.2e}, Loss_ic={loss_ic.item():.2e}")
        print(f"    Grad PDE g_r={grad_r:.3e},  Grad IC g_ic={grad_ic:.3e}")
        print("------------------------------------")

    # 记录
    epochs.append(epoch)
    grad_r_list.append(grad_r)
    grad_ic_list.append(grad_ic)
    ratio_list.append(grad_ic / (grad_r + 1e-12))

# ==========================================================
# 7. 可视化
# ==========================================================

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, grad_r_list)
plt.title("PDE Gradient Norm (g_r)")
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.plot(epochs, grad_ic_list)
plt.title("IC Gradient Norm (g_ic)")
plt.yscale('log')

plt.subplot(1, 3, 3)
plt.plot(epochs, ratio_list)
plt.title("Gradient Ratio g_ic / g_r")
plt.yscale('log')

plt.tight_layout()
plt.show()
