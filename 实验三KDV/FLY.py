import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

# ================= 1. 基础配置 =================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cpu") 
dtype = torch.float64
torch.set_default_dtype(dtype)

# 保持随机数种子绝对一致，确保公平对比
torch.manual_seed(8888)
np.random.seed(8888)

# 👇 注意修改为你的实际路径
DATA_PATH = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/KdV.mat"
SAVE_DIR = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验三KDV/"
data = scipy.io.loadmat(DATA_PATH)
x_exact, t_exact = data["x"].flatten(), data["tt"].flatten()
u_exact = data["uu"].real.T
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
x_min, x_max, t_min, t_max = x_exact.min(), x_exact.max(), t_exact.min(), t_exact.max()

# ================= 2. 纯傅里叶基网络 (无专家网络) =================
class FourierMLP(nn.Module):
    def __init__(self, layers, fourier_dim=32, sigma=2.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(2, fourier_dim, dtype=dtype, device=device) * sigma, requires_grad=False)
        self.net = nn.Sequential()
        current_dim = fourier_dim * 2 
        for i in range(1, len(layers) - 1): 
            self.net.add_module(f'linear_{i}', nn.Linear(current_dim, layers[i]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
            current_dim = layers[i]
        self.out = nn.Linear(current_dim, layers[-1])
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        projected = 2.0 * np.pi * (inputs @ self.B)
        fourier_features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)
        return self.out(self.net(fourier_features))

model = FourierMLP(layers=[2, 80, 80, 80, 80, 1], fourier_dim=32, sigma=2.0).to(device)

# ================= 3. 物理残差与 纯随机采样 (无 RDES) =================
LAMBDA_1, LAMBDA_2 = 1.0, 0.0025

def pde_residual(model, x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + LAMBDA_1 * u * u_x + LAMBDA_2 * u_xxx

# ⚠️ 核心修改：闭眼盲狙，100% 纯均匀/随机采样
def generate_uniform_points(n_points=6000): 
    x_f = torch.rand(n_points, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
    t_f = torch.rand(n_points, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
    x_f.requires_grad_(True); t_f.requires_grad_(True)
    return x_f, t_f

x_ic = torch.tensor(x_exact[:, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min
u_ic = torch.tensor(u_exact[0, :, None], device=device, dtype=dtype)
N_bc = 400

def compute_loss(x_f, t_f, t_bc=None):
    res = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(res**2)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    
    if t_bc is None:
        t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        
    x_lb = (torch.ones_like(t_bc) * x_min).requires_grad_(True)
    x_ub = (torch.ones_like(t_bc) * x_max).requires_grad_(True)
    u_lb, u_ub = model(x_lb, t_bc), model(x_ub, t_bc)
    
    u_x_lb = torch.autograd.grad(u_lb, x_lb, torch.ones_like(u_lb), create_graph=True)[0]
    u_x_ub = torch.autograd.grad(u_ub, x_ub, torch.ones_like(u_ub), create_graph=True)[0]
    u_xx_lb = torch.autograd.grad(u_x_lb, x_lb, torch.ones_like(u_x_lb), create_graph=True)[0]
    u_xx_ub = torch.autograd.grad(u_x_ub, x_ub, torch.ones_like(u_x_ub), create_graph=True)[0]
    
    loss_bc_u = torch.mean((u_lb - u_ub)**2)
    loss_bc_ux = torch.mean((u_x_lb - u_x_ub)**2)
    loss_bc_uxx = torch.mean((u_xx_lb - u_xx_ub)**2)
              
    total_loss = loss_f + 100.0 * loss_ic + 50.0 * loss_bc_u + 10.0 * loss_bc_ux + 1.0 * loss_bc_uxx
    return total_loss, res

# ================= 4. Phase 1: Adam =================
print("🔥 开始 Phase 1: 纯 Fourier 测试 (6000点，纯随机采样，无专家网络)")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_f, t_f = generate_uniform_points(n_points=6000)

for it in range(15001):
    if it > 2000 and it % 2000 == 0 and it < 14000:
        x_f, t_f = generate_uniform_points(n_points=6000) # 重新盲目撒点

    optimizer.zero_grad()
    loss, _ = compute_loss(x_f, t_f)
    loss.backward()
    optimizer.step()
    
    if it % 1000 == 0:
        with torch.no_grad():
            err = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
        print(f"Adam Iter {it:5d} | Loss: {loss.item():.5e} | Rel L2 Error: {err:.4f}")

# ================= 5. Phase 2: L-BFGS =================
print("\n🚀 开始 Phase 2: L-BFGS...")
x_anchor = torch.rand(6000, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min 
t_anchor = torch.rand(6000, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
x_anchor.requires_grad_(True); t_anchor.requires_grad_(True)
fixed_t_bc = torch.rand(N_bc, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min

lbfgs_iter = 0
for stage in range(3):
    print(f"\n🌊 --- 进入 L-BFGS 第 {stage + 1} 阶段 ---")
    x_rad, t_rad = generate_uniform_points(n_points=6000) # L-BFGS 阶段依然是随机采样
    x_f_lbfgs = torch.cat([x_anchor, x_rad], dim=0)
    t_f_lbfgs = torch.cat([t_anchor, t_rad], dim=0)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=8000, max_eval=8000, 
        history_size=50, tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn="strong_wolfe"
    )

    def closure():
        global lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss, _ = compute_loss(x_f_lbfgs, t_f_lbfgs, t_bc=fixed_t_bc)
        loss.backward()
        lbfgs_iter += 1
        if lbfgs_iter % 500 == 0:
            with torch.no_grad():
                e = np.linalg.norm(u_exact - model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)) / np.linalg.norm(u_exact)
            print(f"L-BFGS Iter {lbfgs_iter:5d} | Stage {stage+1} Loss: {loss.item():.5e} | Rel L2 Error: {e:.5f}")
        return loss
    optimizer_lbfgs.step(closure)

# ================= 6. 可视化 =================
print("\n📊 开始绘制结果...")
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_exact.shape)
    final_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    max_error = np.max(np.abs(u_exact - u_pred))
    
    print(f"✅ =============================================")
    print(f"✅ [纯Fourier] 最终相对 L2 误差 (平均): {final_error:.6f}")
    print(f"🔥 [纯Fourier] 最终最大绝对误差 (最差): {max_error:.6f}")
    print(f"✅ =============================================")

fig = plt.figure(figsize=(18, 5))

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
ax1.set_title(f"Pure Fourier Predict (Error: {final_error:.4f})")
ax1.set_xlabel('t'); ax1.set_ylabel('x')
plt.colorbar(im1, ax=ax1)

ax2 = plt.subplot(1, 3, 2)
err_map = np.abs(u_exact - u_pred)
im2 = ax2.contourf(T_mesh, X_mesh, err_map, 100, cmap='inferno')
ax2.set_title("Absolute Error (Pure Fourier 6000 pts)")
ax2.set_xlabel('t'); ax2.set_ylabel('x')
plt.colorbar(im2, ax=ax2)

ax3 = plt.subplot(1, 3, 3)
idx_t1, idx_t2 = int(0.5 * len(t_exact)), int(0.8 * len(t_exact))
ax3.plot(x_exact, u_exact[idx_t1, :], 'k-', linewidth=2, label=f"Exact t={t_exact[idx_t1]:.2f}")
ax3.plot(x_exact, u_pred[idx_t1, :], 'r--', linewidth=2, label="Predict")
ax3.plot(x_exact, u_exact[idx_t2, :], 'b-', linewidth=2, label=f"Exact t={t_exact[idx_t2]:.2f}")
ax3.plot(x_exact, u_pred[idx_t2, :], 'g--', linewidth=2, label="Predict")
ax3.set_title("Wave Profile Slices")
ax3.set_xlabel('x'); ax3.set_ylabel('u(x,t)')
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Fourier_6000_KdV.png"), dpi=300)
print(f"🖼️ 图像已保存至文件夹中！")
plt.show()