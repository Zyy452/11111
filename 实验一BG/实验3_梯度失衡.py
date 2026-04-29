import torch
import torch.nn as nn
import numpy as np

# ==========================================================
# 1. 构造 Heat Equation 真解
# ==========================================================
def u_true(x, t):
    return torch.exp(-np.pi**2 * t) * torch.sin(np.pi * x)

# ==========================================================
# 2. PINN 网络
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
# 3. 训练点
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
# 4. PDE 残差
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
    return u_t - u_xx

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# 5. 训练 + 打印梯度范数
# ==========================================================
def get_grad_norm(model, loss):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().clone().flatten())
    grads = torch.cat(grads)
    return torch.norm(grads).item()

for epoch in range(3000):

    optimizer.zero_grad()

    # PDE残差损失
    r = pde_residual(model, x_r, t_r)
    loss_r = loss_fn(r, torch.zeros_like(r))
    
    # 初始条件损失
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)

    # --------------------
    # 计算梯度：分别回传
    # --------------------
    # PDE梯度
    optimizer.zero_grad()
    loss_r.backward(retain_graph=True)
    g_r = get_grad_norm(model, loss_r)

    # IC梯度
    optimizer.zero_grad()
    loss_ic.backward(retain_graph=True)
    g_ic = get_grad_norm(model, loss_ic)

    # --------------------
    # 正式训练：同时优化
    # --------------------
    optimizer.zero_grad()
    (loss_r + loss_ic).backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss_r={loss_r.item():.2e}, Loss_ic={loss_ic.item():.2e}")
        print(f"    Grad PDE   g_r  = {g_r:.3e}")
        print(f"    Grad IC    g_ic = {g_ic:.3e}")
        print("--------------------------------------------------")
