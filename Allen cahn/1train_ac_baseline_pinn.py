import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import os

# ==========================================================
# 0. 基础配置与目录设置
# ==========================================================
save_dir = "/3241003007/zy/save"
os.makedirs(save_dir, exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# AC 方程非常敏感，必须用 Float64
dtype = torch.float64 
torch.set_default_dtype(dtype)

device = torch.device("cuda")
print(f"任务: Baseline PINN for Allen-Cahn | 设备: {device} | 精度: Float64")

torch.manual_seed(1234)
np.random.seed(1234)

# ==========================================================
# 1. 数据准备 (AC.mat 只用于检测)
# ==========================================================
file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat"
if not os.path.exists(file_path): 
    file_path = "/3241003007/zy/实验二AC/AC.mat"

have_ground_truth = False

if os.path.exists(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        x_exact = data["x"].flatten()   # (512,)
        t_exact = data["tt"].flatten()  # (201,) 
        u_exact_raw = data["uu"]        # (512, 201)
        
        # 维度对齐：np.meshgrid 默认生成 (201, 512)
        X, T = np.meshgrid(x_exact, t_exact)
        u_exact_all = u_exact_raw.T     # 转换为 (201, 512)
        
        X_star = torch.tensor(X.flatten()[:, None], device=device, dtype=dtype)
        T_star = torch.tensor(T.flatten()[:, None], device=device, dtype=dtype)
        
        have_ground_truth = True
        print(f"✅ 成功加载 AC.mat (u shape: {u_exact_all.shape})")
    except Exception as e:
        print(f"❌ 读取出错: {e}")
        exit()
else:
    print(f"❌ 未找到文件: {file_path}")
    exit()

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
                self.net.add_module(f'tanh_{i}', nn.Tanh())
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

layers = [2, 128, 128, 128, 128, 1]
model = PINN(layers).to(device)

# ==========================================================
# 3. 物理约束 (Allen-Cahn)
# ==========================================================
def pde_residual(model, x, t):
    """ Allen-Cahn: u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0 """
    u = model(x, t)
    u_grads = torch.autograd.grad(u, [t, x], torch.ones_like(u), create_graph=True)
    u_dt = u_grads[0]
    u_dx = u_grads[1]
    u_dxx = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph=True)[0]
    
    f = u_dt - 0.0001 * u_dxx + 5.0 * (u**3 - u)
    return f

# ==========================================================
# 4. 训练数据采样
# ==========================================================
N_f = 10000 
N_b = 200   

x_f = (-1 + 2 * torch.rand(N_f, 1, device=device, dtype=dtype)).requires_grad_(True)
t_f = torch.rand(N_f, 1, device=device, dtype=dtype).requires_grad_(True)

x_ic = (-1 + 2 * torch.rand(N_b, 1, device=device, dtype=dtype))
t_ic = torch.zeros_like(x_ic)
u_ic_exact = x_ic**2 * torch.cos(np.pi * x_ic)

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
for epoch in range(5001): 
    optimizer_adam.zero_grad()
    
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
    loss.backward(retain_graph=True)
    optimizer_adam.step()
    
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Ep {epoch:5d} | Loss: {loss.item():.5e} | PDE: {loss_pde.item():.5e} | IC: {loss_ic.item():.5e}")

print(f"Adam Time: {time.time()-start_time:.1f}s")

# ==========================================================
# Phase 2: L-BFGS
# ==========================================================
print("\n>>> 开始微调 (Phase 2: L-BFGS)...")

lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, 
    max_iter=2000, max_eval=2000, history_size=50,
    tolerance_grad=1e-7, line_search_fn="strong_wolfe"
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
    loss.backward(retain_graph=True)
    return loss

lbfgs.step(closure)
final_loss = closure().item()
loss_history.append(final_loss)
print(f"L-BFGS Final Loss: {final_loss:.5e}")

total_train_time = time.time() - start_time

# ==========================================================
# 6. 最终验证与数据保存
# ==========================================================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star)
    u_pred_np = u_pred.cpu().numpy().reshape(201, 512)

error_l2 = np.linalg.norm(u_exact_all - u_pred_np) / np.linalg.norm(u_exact_all)
print(f"\n✨✨✨ Baseline PINN (Allen-Cahn) Final Relative L2 Error: {error_l2:.4e} ✨✨✨")
print(f"总训练耗时: {total_train_time:.1f}s")

# 提取用于可视化的配置点
x_f_plot = x_f.detach().cpu().numpy().flatten()
t_f_plot = t_f.detach().cpu().numpy().flatten()

# 保存数据 (加入 ac_ 前缀)
model_save_path = os.path.join(save_dir, "ac_baseline_pinn_model.pth")
torch.save(model.state_dict(), model_save_path)

data_save_path = os.path.join(save_dir, "ac_baseline_pinn_results.npz")
np.savez(data_save_path, 
         u_pred=u_pred_np, 
         u_exact=u_exact_all, 
         x=x_exact, 
         t=t_exact,
         x_f=x_f_plot,
         t_f=t_f_plot,
         loss_history=np.array(loss_history),
         train_time=total_train_time,
         error_l2=error_l2)
print(f"预测结果及参数数据已保存至: {data_save_path}")