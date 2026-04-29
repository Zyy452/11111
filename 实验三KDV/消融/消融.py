import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import time
import datetime
import matplotlib.pyplot as plt

# ============================================================
# 🛠️ 核心控制面板 (修改这里跑不同的实验)
# ============================================================
N_TRAIN_POINTS = 15000    # 采样点数: 6000 或 15000
USE_FOURIER    = True    # 是否使用傅里叶特征(True/False)
USE_RAD        = True    # 是否使用自适应采样
USE_AB_EXPERTS = False    # 是否启用专家网络
# ============================================================

# 基础环境配置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(8888); np.random.seed(8888)

# 🌟 路径设置
BASE_SAVE_PATH = '/3241003007/zy/save/KDV'
config_str = f"pts{N_TRAIN_POINTS}_F{int(USE_FOURIER)}_R{int(USE_RAD)}_E{int(USE_AB_EXPERTS)}"
EXP_DIR = os.path.join(BASE_SAVE_PATH, config_str)
os.makedirs(EXP_DIR, exist_ok=True)

DATA_PATH = '/3241003007/zy/实验三KDV/KdV.mat'
print(f"🚀 实验启动: {config_str}")

# 加载数据
try:
    data = scipy.io.loadmat(DATA_PATH)
    x_raw, t_raw = data["x"].flatten(), data["tt"].flatten()
    u_raw = data["uu"].real.T # 形状 (Time, Space)
except Exception as e:
    raise FileNotFoundError(f"无法读取数据，请检查路径: {DATA_PATH}")

X_mesh, T_mesh = np.meshgrid(x_raw, t_raw)
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype)
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype)
x_min, x_max, t_min, t_max = x_raw.min(), x_raw.max(), t_raw.min(), t_raw.max()

# ================= 1. 网络架构定义 =================
class FlexibleNet(nn.Module):
    def __init__(self, layers, fourier_dim=32, sigma=2.0):
        super().__init__()
        self.use_fourier = USE_FOURIER
        if self.use_fourier:
            # 傅里叶特征映射矩阵
            self.B = nn.Parameter(torch.randn(2, fourier_dim, dtype=dtype, device=device) * sigma, requires_grad=False)
            current_dim = fourier_dim * 2
        else:
            current_dim = 2
            
        self.net = nn.Sequential()
        for i in range(1, len(layers) - 1):
            self.net.add_module(f'lin_{i}', nn.Linear(current_dim, layers[i]))
            self.net.add_module(f'act_{i}', nn.Tanh())
            current_dim = layers[i]
        self.out = nn.Linear(current_dim, layers[-1])

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        if self.use_fourier:
            proj = 2.0 * np.pi * (inp @ self.B)
            feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        else:
            feat = inp
        return self.out(self.net(feat))

class ABModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = FlexibleNet(layers=[2, 80, 80, 80, 80, 1])
        self.experts = nn.ModuleList()
        self.centers = []
        self.gamma = 60.0

    def forward(self, x, t):
        u = self.base_net(x, t)
        if USE_AB_EXPERTS:
            for i, exp in enumerate(self.experts):
                cx, ct = self.centers[i]
                # 高斯基函数作为权重
                w = torch.exp(-self.gamma * ((x - cx)**2 + (t - ct)**2))
                u = u + w * exp(x, t)
        return u
        
    def add_expert(self, cx, ct):
        new_exp = FlexibleNet(layers=[2, 40, 40, 40, 1], fourier_dim=16).to(device)
        nn.init.zeros_(new_exp.out.weight); nn.init.zeros_(new_exp.out.bias)
        self.experts.append(new_exp)
        self.centers.append((cx, ct))
        return new_exp

model = ABModel().to(device)

# ================= 2. 物理残差与采样逻辑 =================
def pde_res(model, x, t):
    # 必须开启计算图以支持偏导
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    # KdV 方程: u_t + u*u_x + 0.0025*u_xxx = 0
    return u_t + 1.0 * u * u_x + 0.0025 * u_xxx

def get_points():
    if USE_RAD:
        n_uni, n_res = int(0.2 * N_TRAIN_POINTS), int(0.8 * N_TRAIN_POINTS)
        x_u = torch.rand(n_uni, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
        t_u = torch.rand(n_uni, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        
        # RAD 采样池
        x_p = torch.rand(50000, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
        t_p = torch.rand(50000, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        x_p.requires_grad_(True); t_p.requires_grad_(True)
        
        res_p = pde_res(model, x_p, t_p)
        prob = (torch.abs(res_p)**2).flatten().detach()
        prob = prob / (prob.sum() + 1e-10)
        idx = torch.multinomial(prob, n_res, replacement=True)
        return torch.cat([x_u, x_p[idx].detach()], 0), torch.cat([t_u, t_p[idx].detach()], 0)
    else:
        xf = torch.rand(N_TRAIN_POINTS, 1, device=device, dtype=dtype) * (x_max - x_min) + x_min
        tf = torch.rand(N_TRAIN_POINTS, 1, device=device, dtype=dtype) * (t_max - t_min) + t_min
        return xf, tf

# ================= 3. 训练准备 =================
x_ic = torch.tensor(x_raw[:, None], device=device, dtype=dtype)
u_ic = torch.tensor(u_raw[0, :, None], device=device, dtype=dtype)
t_ic = torch.ones_like(x_ic) * t_min

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_f, t_f = get_points()

start_time = time.time()
print(f"⏲️ 正在开始训练...")

for it in range(10001):
    # 🌟 修复关键：确保每一轮迭代输入点都开启梯度计算
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)
    
    optimizer.zero_grad()
    
    # 物理损失
    res = pde_res(model, x_f, t_f)
    loss_f = torch.mean(res**2)
    
    # 初值损失
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    
    loss = loss_f + 100.0 * loss_ic
    loss.backward()
    optimizer.step()
    
    # 动态专家生长逻辑 (防止 RuntimeError 的增强处理)
    if USE_AB_EXPERTS and it > 0 and it % 2000 == 0 and len(model.experts) < 4:
        # 1. 寻找最大残差位置
        temp_x = x_f.detach().requires_grad_(True)
        temp_t = t_f.detach().requires_grad_(True)
        temp_res = pde_res(model, temp_x, temp_t)
        
        r_abs = torch.abs(temp_res).detach()
        m_idx = torch.argmax(r_abs)
        
        # 2. 如果最大残差超过阈值，添加专家
        if r_abs[m_idx] > 0.05:
            cx, ct = temp_x[m_idx].item(), temp_t[m_idx].item()
            new_e = model.add_expert(cx, ct)
            # 将新专家的参数加入优化器
            optimizer.add_param_group({'params': new_e.parameters(), 'lr': 1e-3})
            # ✅ 修复处的代码：加入了 .item() 转换
            print(f"🎯 [Iter {it}] 残差峰值 {r_abs[m_idx].item():.4f} -> 增加专家 @(x={cx:.2f}, t={ct:.2f})")

    if it % 1000 == 0:
        with torch.no_grad():
            u_p_test = model(X_star, T_star).cpu().numpy().reshape(u_raw.shape)
            err_now = np.linalg.norm(u_raw - u_p_test) / np.linalg.norm(u_raw)
        print(f"Iter {it:5d} | Loss: {loss.item():.3e} | Rel L2: {err_now:.4f} | Experts: {len(model.experts)}")

# ================= 4. 评估、保存与可视化 =================
model.eval()
with torch.no_grad():
    u_pred = model(X_star, T_star).cpu().numpy().reshape(u_raw.shape)
    final_err = np.linalg.norm(u_raw - u_pred) / np.linalg.norm(u_raw)

# 保存 .pt 数据
save_dict = {
    "u_pred": u_pred, "u_exact": u_raw, "x": x_raw, "t": t_raw,
    "error": final_err, "config": config_str, "time": time.time()-start_time
}
pt_path = os.path.join(EXP_DIR, f"KDV_{config_str}.pt")
torch.save(save_dict, pt_path)

# 快速预览图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.contourf(T_mesh, X_mesh, u_pred, 100, cmap='jet')
plt.title(f"Predict: {config_str}")
plt.subplot(1, 2, 2)
plt.contourf(T_mesh, X_mesh, np.abs(u_pred - u_raw), 100, cmap='inferno')
plt.title(f"Abs Error (L2: {final_err:.4f})")
plt.savefig(os.path.join(EXP_DIR, f"QuickView_{config_str}.png"), dpi=150)
plt.close()

print(f"\n✅ 实验配置 {config_str} 完成！")
print(f"📊 最终相对 L2 误差: {final_err:.6f}")
print(f"📁 结果目录: {EXP_DIR}")