import os
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import time

# ==========================================================
# 0. 目录、设备配置与随机种子
# ==========================================================
save_dir = "/3241003007/zy/save"
os.makedirs(save_dir, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

torch.manual_seed(42)
np.random.seed(42)
if device.type == "mps":
    torch.mps.manual_seed(42)

# ==========================================================
# 1. 读取 Burgers 方程真解
# ==========================================================
try:
    data = loadmat("/3241003007/zy/实验一BG/burgers_shock.mat")
except FileNotFoundError:
    raise FileNotFoundError("请确保 burgers_shock.mat 文件路径正确")

x_exact = data["x"].flatten()             # [256]
t_exact = data["t"].flatten()             # [100]
Exact = data["usol"]                      # [256, 100]
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)

# ==========================================================
# 2. AB-PINN 结构定义
# ==========================================================
def make_mlp(layers):
    seq = []
    for i in range(len(layers)-1):
        seq.append(nn.Linear(layers[i], layers[i+1]))
        if i < len(layers)-2:
            seq.append(nn.Tanh())
    return nn.Sequential(*seq)

class SubNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = make_mlp([2, hidden, hidden, hidden, hidden, 1])
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

class AB_PINN(nn.Module):
    def __init__(self, K=2, hidden=64):
        super().__init__()
        self.K = K
        self.subnets = nn.ModuleList([SubNet(hidden) for _ in range(K)])
        
        self.c = nn.Parameter(torch.tensor([-0.5, 0.5], dtype=torch.float32))
        self.log_gamma = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

    def get_weights(self, x):
        gamma = torch.exp(self.log_gamma)
        diff = x - self.c.view(1, -1) 
        logits = - gamma.view(1, -1) * (diff**2)
        return torch.softmax(logits, dim=1)

    def forward(self, x, t):
        w = self.get_weights(x)
        u_out = 0
        for i in range(self.K):
            u_out += w[:, i:i+1] * self.subnets[i](x, t)
        return u_out

model = AB_PINN(K=2, hidden=64).to(device)

# ==========================================================
# 3. 实验参数与数据准备
# ==========================================================
nu = 0.01 / np.pi
N_r = 20000
N_ic = 200
N_bc = 200
EPOCHS = 30000 

x_r = (-1 + 2 * torch.rand(N_r, 1, dtype=torch.float32)).to(device)
t_r = torch.rand(N_r, 1, dtype=torch.float32).to(device)

x_ic = torch.linspace(-1, 1, N_ic, dtype=torch.float32).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic = -torch.sin(np.pi * x_ic).to(device)

t_bc = torch.rand(N_bc, 1, dtype=torch.float32).to(device)
x_bc_L = -1.0 * torch.ones_like(t_bc).to(device)
x_bc_R = 1.0 * torch.ones_like(t_bc).to(device)
u_bc_val = torch.zeros_like(t_bc).to(device)

# ==========================================================
# 4. PDE 残差优化
# ==========================================================
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    ones = torch.ones_like(u)
    u_grads = torch.autograd.grad(u, [x, t], grad_outputs=ones, create_graph=True)
    u_x, u_t = u_grads[0], u_grads[1]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    return u_t + u * u_x - nu * u_xx

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ==========================================================
# 5. 训练循环
# ==========================================================
start_time = time.time()
print("--- Starting Training ---")

for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    
    res = pde_residual(model, x_r, t_r)
    loss_r = loss_fn(res, torch.zeros_like(res))
    loss_ic = loss_fn(model(x_ic, t_ic), u_ic)
    loss_bc = loss_fn(model(x_bc_L, t_bc), u_bc_val) + loss_fn(model(x_bc_R, t_bc), u_bc_val)
    
    loss = loss_r + loss_ic + loss_bc
    loss.backward()
    optimizer.step()
    
    if epoch % 2000 == 0:
        elapsed = time.time() - start_time
        c_vals = model.c.detach().cpu().numpy()
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | Centers: {c_vals} | 累计用时: {elapsed:.1f}s")

total_train_time = time.time() - start_time

# ==========================================================
# 6. 最终评估与数据保存 (包含自适应参数)
# ==========================================================
model.eval()

x_test_tensor = torch.tensor(X_mesh.flatten()[:, None], dtype=torch.float32).to(device)
t_test_tensor = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32).to(device)
xx_eval = torch.tensor(x_exact, dtype=torch.float32).view(-1, 1).to(device)

with torch.no_grad():
    u_pred_raw = model(x_test_tensor, t_test_tensor).cpu().numpy()
    u_pred = u_pred_raw.reshape(len(t_exact), len(x_exact)).T
    w_final = model.get_weights(xx_eval).cpu().numpy()
    
    # 提取模型学习到的中心 c 和宽度参数 gamma
    c_final = model.c.cpu().numpy()
    gamma_final = torch.exp(model.log_gamma).cpu().numpy()

error_l2 = np.linalg.norm(Exact - u_pred, 2) / np.linalg.norm(Exact, 2)

print(f"\n--- 训练完成 ---")
print(f"总训练耗时: {total_train_time:.2f} 秒")
print(f"AB-PINN 最终相对 L2 误差: {error_l2:.3e}")
print(f"最终学习到的中心 c: {c_final}")

model_save_path = os.path.join(save_dir, "ab_pinn_model.pth")
torch.save(model.state_dict(), model_save_path)

data_save_path = os.path.join(save_dir, "ab_pinn_results.npz")
np.savez(data_save_path, 
         u_pred=u_pred, 
         u_exact=Exact, 
         x=x_exact, 
         t=t_exact,
         w_final=w_final,  
         c=c_final,           # 新增：子域中心点
         gamma=gamma_final,   # 新增：子域宽度参数
         train_time=total_train_time,
         error_l2=error_l2)
print(f"预测结果及参数数据已保存至: {data_save_path}")