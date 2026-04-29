import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# ==========================================================
# 0. 设备配置 (MPS 适配)
# ==========================================================
# 检查是否支持 MPS (Mac GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用设备: MPS (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用设备: CUDA")
else:
    device = torch.device("cpu")
    print("使用设备: CPU")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================================
# 1. 读取真解
# ==========================================================
# 请确保该路径正确
data_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验一/burgers_shock.mat"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"未找到数据文件: {data_path}")

data = loadmat(data_path)
x_exact = data["x"].flatten()
t_exact = data["t"].flatten()
Exact = data["usol"]
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)

# ==========================================================
# 2. 网络结构定义
# ==========================================================
class SubNet(nn.Module):
    def __init__(self, layers=[2, 40, 40, 40, 1]):
        super().__init__()
        net = []
        for i in range(len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# ==========================================================
# 3. D3PINN 实验参数与数据生成
# ==========================================================
nu = 0.01 / np.pi
N_total_r = 20000
EPOCHS = 30000
t_split = 0.5

# 初始化子模型并移至设备
model1 = SubNet().to(device)
model2 = SubNet().to(device)

optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()

def get_residual_points(n, t_range):
    x = -1 + 2 * torch.rand(n, 1)
    t = t_range[0] + (t_range[1] - t_range[0]) * torch.rand(n, 1)
    # 注意：明确指定 device 提高效率
    return x.to(device), t.to(device)

x_r1, t_r1 = get_residual_points(N_total_r // 2, [0, t_split])
x_r2, t_r2 = get_residual_points(N_total_r // 2, [t_split, 1.0])

x_ic = torch.linspace(-1, 1, 200).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic = -torch.sin(np.pi * x_ic).to(device)

t_bc1 = (t_split * torch.rand(100, 1)).to(device)
t_bc2 = (t_split + t_split * torch.rand(100, 1)).to(device)
x_bc_L = -1 * torch.ones(100, 1).to(device)
x_bc_R = 1 * torch.ones(100, 1).to(device)
u_bc_val = torch.zeros(100, 1).to(device)

x_int = torch.linspace(-1, 1, 400).view(-1, 1).to(device)
t_int = (torch.ones_like(x_int) * t_split).to(device)

# ==========================================================
# 4. PDE 残差函数
# ==========================================================
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    # MPS 在 autograd 某些情况下可能比 CUDA 敏感，确保 grad_outputs 在同设备
    ones = torch.ones_like(u).to(device)
    
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    
    # 计算二阶导
    ones_ux = torch.ones_like(u_x).to(device)
    u_xx = torch.autograd.grad(u_x, x, ones_ux, create_graph=True)[0]
    
    return u_t + u * u_x - nu * u_xx

# ==========================================================
# 5. 训练循环
# ==========================================================
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    res1 = pde_residual(model1, x_r1, t_r1)
    res2 = pde_residual(model2, x_r2, t_r2)
    loss_r = loss_fn(res1, torch.zeros_like(res1)) + loss_fn(res2, torch.zeros_like(res2))
    
    loss_ic = loss_fn(model1(x_ic, t_ic), u_ic)
    loss_bc = (loss_fn(model1(x_bc_L, t_bc1), u_bc_val) + loss_fn(model1(x_bc_R, t_bc1), u_bc_val) +
              loss_fn(model2(x_bc_L, t_bc2), u_bc_val) + loss_fn(model2(x_bc_R, t_bc2), u_bc_val))
    
    # 界面损失
    u_int1 = model1(x_int, t_int)
    u_int2 = model2(x_int, t_int)
    loss_int = loss_fn(u_int1, u_int2) # 这里使用双向约束更稳定
    
    # 综合 Loss (界面权重可根据 MPS 训练情况微调，通常 5.0 左右)
    loss = loss_r + loss_ic + loss_bc + 5.0 * loss_int
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Total Loss: {loss.item():.3e} | Interface: {loss_int.item():.3e}")

# ==========================================================
# 6. 最终评估与可视化
# ==========================================================
def predict_full(x_pts, t_pts):
    # 此步骤涉及逻辑判断，建议在 CPU 上合并结果
    u_pred = np.zeros(len(t_pts))
    t_pts_cpu = t_pts.cpu().numpy().flatten()
    idx1 = (t_pts_cpu <= t_split)
    idx2 = (t_pts_cpu > t_split)
    
    model1.eval()
    model2.eval()
    with torch.no_grad():
        if any(idx1):
            u_pred[idx1] = model1(x_pts[idx1], t_pts[idx1]).cpu().numpy().flatten()
        if any(idx2):
            u_pred[idx2] = model2(x_pts[idx2], t_pts[idx2]).cpu().numpy().flatten()
    return u_pred

# 计算误差
x_test = torch.tensor(X_mesh.flatten()[:, None], dtype=torch.float32).to(device)
t_test = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32).to(device)
u_final_pred = predict_full(x_test, t_test).reshape(len(t_exact), len(x_exact)).T

error_l2 = np.linalg.norm(Exact - u_final_pred, 2) / np.linalg.norm(Exact, 2)
print(f"\nD3PINN (MPS) Final Relative L2 Error: {error_l2:.3e}")

# 多时刻绘图
times = [0.0, 0.25, 0.50, 0.75, 1.0]
xx_eval = torch.linspace(-1, 1, 256).view(-1, 1).to(device)

plt.figure(figsize=(12, 8))
for i, t0 in enumerate(times):
    tt_eval = torch.ones_like(xx_eval) * t0
    pred = predict_full(xx_eval, tt_eval)
    true = Exact[:, int(t0 * (len(t_exact)-1))]
    
    plt.subplot(2, 3, i+1)
    plt.plot(x_exact, true, 'k-', label="True")
    plt.plot(x_exact, pred, 'b--', label="D3PINN")
    plt.axvline(x=0, color='gray', alpha=0.2)
    plt.title(f"t = {t0}")
    plt.legend()

plt.tight_layout()
plt.show()