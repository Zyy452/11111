import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os

# ==========================================================
# 0. 基础配置
# ==========================================================
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# AC 方程非常敏感，必须用 Float64
dtype = torch.float64 
torch.set_default_dtype(dtype)

#
device = torch.device("cpu")

print(f"任务: Baseline PINN for Allen-Cahn (Fixed Data & Backward) | 设备: {device}")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备 (AC.mat只用于检测)
# ==========================================================
file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"
have_ground_truth = False

if os.path.exists(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        
        # 适配数据变量名 ---
        x_exact = data["x"].flatten()   # (512,)
        t_exact = data["tt"].flatten()  # (201,)  <-- 数据里叫 'tt'
        u_exact_raw = data["uu"]        # (512, 201) <-- 数据里叫 'uu'
        
        # 维度对齐 
        # np.meshgrid(x, t) 默认生成的形状是 (len(t), len(x))，即 (201, 512)
        # uu 是 (512, 201)，所以需要转置 (.T)
        X, T = np.meshgrid(x_exact, t_exact)
        u_exact_all = u_exact_raw.T  # 变成 (201, 512) 以匹配 X, T
        
        #以此生成用于评估的张量
        X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
        T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)
        
        have_ground_truth = True
        print(f"成功加载 AC.mat (u shape: {u_exact_all.shape})")
        
    except Exception as e:
        print(f"读取出错: {e}")
else:
    print(f"未找到文件: {file_path}")
    # 备用假数据
    x_exact = np.linspace(-1, 1, 200)
    t_exact = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_exact, t_exact)
    X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
    T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)

# ==========================================================
# 2. 模型定义
# ==========================================================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                # Tanh 激活函数对高阶导数更平滑
                self.net.add_module(f'tanh_{i}', nn.Tanh())
        
        # 初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# 网络结构
layers = [2, 128, 128, 128, 128, 1]
model = PINN(layers).to(device)

# ==========================================================
# 3. 物理约束 (Allen-Cahn)
# ==========================================================
def pde_residual(model, x, t):
    """
    Allen-Cahn: u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0
    """
    u = model(x, t)
    
    # 计算一阶导 (t, x)
    u_grads = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dt = u_grads[0]
    u_dx = u_grads[1]
    
    # 计算二阶导 (xx)
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    
    # PDE Residual
    f = u_dt - 0.0001 * u_dxx + 5.0 * (u**3 - u)
    return f

# ==========================================================
# 4. 训练数据采样
# ==========================================================
N_f = 10000 
N_b = 200   

# 内部配点
x_f = (-1 + 2 * torch.rand(N_f, 1, device=device, dtype=dtype)).requires_grad_(True)
t_f = torch.rand(N_f, 1, device=device, dtype=dtype).requires_grad_(True)

# 初始条件 (t=0)
# u(x,0) = x^2 * cos(pi * x)
x_ic = (-1 + 2 * torch.rand(N_b, 1, device=device, dtype=dtype))
t_ic = torch.zeros_like(x_ic)
u_ic_exact = x_ic**2 * torch.cos(np.pi * x_ic)

# 周期性边界条件 (Periodic BC)
# t 在 [0, 1] 随机采样
t_bc = torch.rand(N_b, 1, device=device, dtype=dtype).requires_grad_(True)
x_bc_left = -1.0 * torch.ones_like(t_bc, requires_grad=True)
x_bc_right = 1.0 * torch.ones_like(t_bc, requires_grad=True)

# ==========================================================
# 5. 训练循环
# ==========================================================
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

start_time = time.time()
loss_history = []

print(">>> 开始训练 (Phase 1: Adam)...")

for epoch in range(5001): # 先跑 5000 次看看
    optimizer_adam.zero_grad()
    
    # 1. PDE Loss
    res = pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(res**2)
    
    # 2. IC Loss
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic_exact)**2)
    
    # 3. BC Loss (Periodic)
    u_left = model(x_bc_left, t_bc)
    u_right = model(x_bc_right, t_bc)
    loss_bc_u = torch.mean((u_left - u_right)**2)
    
    # 导数边界
    u_x_left = torch.autograd.grad(u_left, x_bc_left, torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_bc_right, torch.ones_like(u_right), create_graph=True)[0]
    loss_bc_ux = torch.mean((u_x_left - u_x_right)**2)
    
    # 总 Loss
    loss = loss_pde + 100.0 * loss_ic + 10.0 * (loss_bc_u + loss_bc_ux)
    
    loss.backward(retain_graph=True)
    
    optimizer_adam.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Ep {epoch:5d} | Loss: {loss.item():.5e} | PDE: {loss_pde.item():.5e} | IC: {loss_ic.item():.5e}")

print(f"Adam Time: {time.time()-start_time:.1f}s")

# ==========================================================
# Phase 2: L-BFGS
# ==========================================================
print("\n>>> 开始微调 (Phase 2: L-BFGS)...")

lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=2000, 
    max_eval=2000, 
    history_size=50,
    tolerance_grad=1e-7, 
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    
    res = pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(res**2)
    
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic_exact)**2)
    
    u_left = model(x_bc_left, t_bc)
    u_right = model(x_bc_right, t_bc)
    loss_bc_u = torch.mean((u_left - u_right)**2)
    
    u_x_left = torch.autograd.grad(u_left, x_bc_left, torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_bc_right, torch.ones_like(u_right), create_graph=True)[0]
    loss_bc_ux = torch.mean((u_x_left - u_x_right)**2)
    
    loss = loss_pde + 100.0 * loss_ic + 10.0 * (loss_bc_u + loss_bc_ux)
    
    # 这里同样需要 retain_graph
    loss.backward(retain_graph=True)
    return loss

lbfgs.step(closure)
final_loss = closure().item()
print(f"L-BFGS Final Loss: {final_loss:.5e}")

# ==========================================================
# 6. 结果可视化与评估
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star)
    # 变成 (201, 512)
    u_pred_np = u_pred.cpu().numpy().reshape(201, 512)

# 计算 L2 Error
if have_ground_truth:
    # u_exact_all 已经是 (201, 512)
    error_l2 = np.linalg.norm(u_exact_all - u_pred_np) / np.linalg.norm(u_exact_all)
    print(f"\n Baseline Error (Allen-Cahn): {error_l2:.4e} ")
    title_str = f"Baseline PINN (Error: {error_l2:.2e})"
else:
    title_str = "Baseline PINN (Prediction)"

# 绘图
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.semilogy(loss_history)
plt.title("Loss History")
plt.xlabel("Epoch")

# 预测值
plt.subplot(1, 3, 2)
plt.imshow(u_pred_np.T, interpolation='nearest', cmap='jet', 
           extent=[0, 1, -1, 1], origin='lower', aspect='auto')
plt.colorbar(label='u_pred')
plt.title(title_str)
plt.xlabel("t")
plt.ylabel("x")

# 真实值（对比）
if have_ground_truth:
    plt.subplot(1, 3, 3)
    plt.imshow(u_exact_all.T, interpolation='nearest', cmap='jet', 
               extent=[0, 1, -1, 1], origin='lower', aspect='auto')
    plt.colorbar(label='u_exact')
    plt.title("Ground Truth")
    plt.xlabel("t")
    plt.ylabel("x")

plt.tight_layout()
plt.savefig('baseline_ac_result.png', dpi=300)
print("✅ 结果已保存为 baseline_ac_result.png")