import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import os 
import time 
import datetime 

# ========================================================== 
# 0. 基础配置 
# ========================================================== 
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
device = torch.device("cpu") 
dtype = torch.float64 
torch.set_default_dtype(dtype) 

print(f" 任务: RADS + AB-PINN (Timer & Precision Fix) | Device: CPU") 

torch.manual_seed(1234) 
np.random.seed(1234) 

# ========================================================== 
# 1. 数据读取 
# ========================================================== 
EPSILON = 0.0001 
GAMMA = 5.0 

try: 
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat" 
    data = scipy.io.loadmat(file_path) 
    x_exact = data["x"].flatten() 
    t_exact = data["tt"].flatten() 
    u_exact_all = data["uu"] 
    # 强制修正形状为 (Nx, Nt) 
    if u_exact_all.shape[0] != len(x_exact): 
        u_exact_all = u_exact_all.T 
    print(f" 数据加载成功: x={x_exact.shape}, t={t_exact.shape}, u={u_exact_all.shape}") 
except: 
    print(" 警告: 使用伪数据") 
    x_exact = np.linspace(-1, 1, 512) 
    t_exact = np.linspace(0, 1, 201) 
    u_exact_all = np.zeros((512, 201)) 

# 生成全场网格 (512 x 201 = 102,912 points) 
X_mesh, T_mesh = np.meshgrid(x_exact, t_exact, indexing='ij') 
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype) 
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype) 

# ========================================================== 
# 2. 网络定义 (Scaled Tanh) 
# ========================================================== 
class RADS_Generator(nn.Module): 
    def __init__(self, z_dim=2): 
        super().__init__() 
        self.net = nn.Sequential( 
            nn.Linear(z_dim, 64), nn.Tanh(), 
            nn.Linear(64, 64), nn.Tanh(), 
            nn.Linear(64, 64), nn.Tanh(), 
            nn.Linear(64, 2) 
        ) 
        
    def forward(self, z): 
        raw = self.net(z) 
        x = torch.tanh(raw[:, 0:1]) 
        t = torch.sigmoid(raw[:, 1:2]) #tanh
        return x, t 

class PeriodicEmbedding(nn.Module): 
    def __init__(self): 
        super().__init__() 
        
    def forward(self, x): 
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=1) 

class ScaledTanh(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.scale = nn.Parameter(torch.tensor(5.0, dtype=dtype)) 
        
    def forward(self, x): 
        return torch.tanh(self.scale * x) 

class DeepSubNet(nn.Module): 
    def __init__(self, input_dim=3, hidden=64, is_new_addition=False): 
        super().__init__() 
        self.net = nn.Sequential( 
            nn.Linear(input_dim, hidden), ScaledTanh(), 
            nn.Linear(hidden, hidden), ScaledTanh(), 
            nn.Linear(hidden, hidden), ScaledTanh(), 
            nn.Linear(hidden, hidden), ScaledTanh(), 
            nn.Linear(hidden, 1) 
        ) 
        for m in self.net.modules(): 
            if isinstance(m, nn.Linear): 
                nn.init.xavier_normal_(m.weight) 
                nn.init.zeros_(m.bias) 
        if is_new_addition: 
            nn.init.zeros_(self.net[-1].weight) 
            nn.init.zeros_(self.net[-1].bias) 

    def forward(self, x_emb, t): 
        return self.net(torch.cat([x_emb, t], dim=1)) 

class DynamicABPINN(nn.Module): 
    def __init__(self, hidden=64): 
        super().__init__() 
        self.embed = PeriodicEmbedding() 
        self.hidden = hidden 
        self.subnets = nn.ModuleList([DeepSubNet(input_dim=3, hidden=hidden)]) 
        self.centers = nn.ParameterList([nn.Parameter(torch.tensor([0.0, 0.5], device=device, dtype=dtype))]) 
        self.log_gammas = nn.ParameterList([nn.Parameter(torch.tensor([[3.0, 3.0]], device=device, dtype=dtype))]) 

    def add_subdomain(self, mu_init): 
        new_net = DeepSubNet(input_dim=3, hidden=self.hidden, is_new_addition=True).to(device) 
        mu = nn.Parameter(torch.tensor(mu_init, device=device, dtype=dtype).view(1, 2)) 
        lg = nn.Parameter(torch.tensor([[3.0, 3.0]], device=device, dtype=dtype)) 
        self.subnets.append(new_net) 
        self.centers.append(mu) 
        self.log_gammas.append(lg) 

    def get_window_values(self, x, t): 
        xt = torch.cat([x, t], dim=1) 
        values = [] 
        for i in range(len(self.subnets)): 
            diff_sq = (xt - self.centers[i])**2 
            gamma = torch.exp(self.log_gammas[i]) 
            dist_weighted = torch.sum(gamma * diff_sq, dim=1, keepdim=True) 
            values.append(torch.exp(-dist_weighted)) 
        return values 

    def forward(self, x, t): 
        x_emb = self.embed(x) 
        xt = torch.cat([x, t], dim=1) 
        logits = [] 
        for i in range(len(self.subnets)): 
            diff_sq = (xt - self.centers[i])**2 
            gamma = torch.exp(self.log_gammas[i]) 
            logits.append(-torch.sum(gamma * diff_sq, dim=1, keepdim=True)) 
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1) 
        u_out = 0 
        for i in range(len(self.subnets)): 
            u_out += weights[:, i] * self.subnets[i](x_emb, t) 
        u_0 = x**2 * torch.cos(np.pi * x) 
        u_final = torch.tanh(t) * u_out + u_0 
        return u_final 

# ========================================================== 
# 3. 训练流程 
# ========================================================== 
def pde_residual(model, x, t): 
    u = model(x, t) 
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] 
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] 
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0] 
    return u_t - EPSILON * u_xx + GAMMA * (u**3 - u) 

ADAM_EPOCHS = 25000 
BATCH_SIZE = 6000 

model = DynamicABPINN(hidden=64).to(device) 
generator = RADS_Generator().to(device) 

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3) 
optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3) 

def get_uniform_points(N): 
    x = (-1 + 2 * torch.rand(N, 1, device=device, dtype=dtype)).requires_grad_(True) 
    t = torch.rand(N, 1, device=device, dtype=dtype).requires_grad_(True) 
    return x, t 

loss_history = [] 
err_history = [] 
iters = [] 

print(f"\n>>> Phase 1: RADS Adversarial Training ({ADAM_EPOCHS} Epochs)...") 
phase1_start = time.time() 
start_time = time.time() # Loop start 

for epoch in range(ADAM_EPOCHS + 1): 
    # A. Generator Update 
    if epoch % 5 == 0: 
        for _ in range(2): 
            optim_gen.zero_grad() 
            z = torch.randn(BATCH_SIZE // 2, 2, device=device, dtype=dtype) 
            x_gen, t_gen = generator(z) 
            loss_gen = -torch.mean(pde_residual(model, x_gen, t_gen)**2) 
            loss_gen.backward() 
            optim_gen.step() 

    # B. PINN Update 
    optim_pinn.zero_grad() 
    N_aais = int(0.6 * BATCH_SIZE) 
    with torch.no_grad(): 
        z = torch.randn(N_aais, 2, device=device, dtype=dtype) 
        x_adv, t_adv = generator(z) 
        
    x_adv.requires_grad_(True)
    t_adv.requires_grad_(True) 
    x_uni, t_uni = get_uniform_points(BATCH_SIZE - N_aais) 
    x_train = torch.cat([x_adv, x_uni], dim=0) 
    t_train = torch.cat([t_adv, t_uni], dim=0) 
    res = pde_residual(model, x_train, t_train) 
    weights = torch.ones_like(res)
    weights[:N_aais] = 5.0 
    loss = torch.mean(weights * res**2) 
    loss.backward() 
    optim_pinn.step() 
    
    # C. Dynamic Expert 
    if epoch > 2000 and epoch % 3000 == 0: 
        with torch.no_grad(): 
            res_abs = torch.abs(res).flatten() 
            mask = (t_train.flatten() > 0.05) 
            if mask.sum() > 0: 
                res_masked = res_abs.clone()
                res_masked[~mask] = -1.0 
                max_res = res_masked.max().item() 
                if len(model.subnets) < 10 and max_res > 0.01: 
                    idx = torch.argmax(res_masked) 
                    print(f" Added Expert @ ({x_train[idx].item():.2f}, {t_train[idx].item():.2f}) | Res={max_res:.4f}") 
                    model.add_subdomain([x_train[idx].item(), t_train[idx].item()]) 
                    optim_pinn = torch.optim.Adam(model.parameters(), lr=8e-4) 

    # D. Logging with Timer 
    if epoch % 500 == 0: 
        model.eval() 
        with torch.no_grad(): 
            u_check = model(X_star, T_star).cpu().numpy().reshape(512, 201) 
            curr_err = np.linalg.norm(u_exact_all - u_check) / np.linalg.norm(u_exact_all) 
        model.train() 
        loss_history.append(loss.item()) 
        err_history.append(curr_err) 
        iters.append(epoch) 
        
        # 计算时间 
        elapsed = time.time() - start_time 
        avg_time_per_step = elapsed / 500 if epoch > 0 else 0 
        remaining_steps = ADAM_EPOCHS - epoch 
        eta_seconds = avg_time_per_step * remaining_steps 
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds))) 
        print(f"Ep {epoch:5d}/{ADAM_EPOCHS} | Loss: {loss.item():.2e} | Err: {curr_err:.4f} | Experts: {len(model.subnets)} | ETA: {eta_str}") 
        start_time = time.time() # Reset timer for next block 

phase1_time = time.time() - phase1_start 
print(f"✅ Phase 1 Finished in {str(datetime.timedelta(seconds=int(phase1_time)))}") 

# ========================================================== 
# Phase 2: L-BFGS (Full-Grid Saturation) 
# ========================================================== 
print("\n>>> Phase 2: L-BFGS Full-Grid Refinement ...") 
print(" OK ") 


x_final = X_star.clone().detach().requires_grad_(True) 
t_final = T_star.clone().detach().requires_grad_(True) 

print(f" L-BFGS Start. Max Iter: 5000. Please wait...") 
phase2_start = time.time() 

lbfgs = torch.optim.LBFGS( 
    model.parameters(), 
    lr=1.0, 
    max_iter=5000, 
    history_size=100, 
    tolerance_grad=1e-11, 
    tolerance_change=1e-11, 
    line_search_fn="strong_wolfe" 
) 

#  控制 L-BFGS 的打印频率
iter_count_lbfgs = 0

def closure(): 
    global iter_count_lbfgs
    lbfgs.zero_grad() 
    res = pde_residual(model, x_final, t_final) 
    loss = torch.mean(res**2) # 全场残差 
    loss.backward() 
    
    # 每 500 轮打印一次进度
    iter_count_lbfgs += 1
    if iter_count_lbfgs % 500 == 0:
        print(f"  L-BFGS Iter {iter_count_lbfgs:4d}/5000 | Loss: {loss.item():.5e}")
        
    return loss 

lbfgs.step(closure) 
final_loss = closure().item() 

phase2_time = time.time() - phase2_start 
print(f"Phase 2 Finished in {str(datetime.timedelta(seconds=int(phase2_time)))}") 
print(f" L-BFGS Final Loss: {final_loss:.5e}") 

# ========================================================== 
# 5. 验证与绘图 
# ========================================================== 
model.eval() 
with torch.no_grad(): 
    u_pred = model(X_star, T_star).cpu().numpy().reshape(512, 201) 

final_error = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all) 
print(f"\n✨✨✨ Final Relative L2 Error: {final_error:.4e} ✨✨✨") 
print(f"Total Time: {str(datetime.timedelta(seconds=int(phase1_time + phase2_time)))}") 

plt.figure(figsize=(24, 6)) 

# 1. Error Curve 
plt.subplot(1, 4, 1) 
plt.semilogy(iters, err_history, 'b-', label='L2 Error') 
plt.title("Error Convergence") 
plt.xlabel("Epochs"); plt.grid(True, alpha=0.3) 

# 2. Slice t=0.5 
plt.subplot(1, 4, 2) 
t_idx = int(0.5 * 201) 
plt.plot(x_exact, u_exact_all[:, t_idx], 'k-', linewidth=3, label="Exact") 
plt.plot(x_exact, u_pred[:, t_idx], 'r--', linewidth=2, label="Pred") 
plt.title(f"t=0.5 Slice (Err={final_error:.2%})") 
plt.legend(); plt.grid(True, alpha=0.3) 

# 3. Error Map + Experts 
plt.subplot(1, 4, 3) 
err_map = np.abs(u_exact_all - u_pred) 
plt.imshow(err_map, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='jet') 
plt.colorbar(label='Abs Error') 

X_grid = X_star.reshape(512, 201).cpu().numpy() 
T_grid = T_star.reshape(512, 201).cpu().numpy() 
with torch.no_grad(): 
    window_values = model.get_window_values(X_star, T_star) 
    colors = ['white', 'cyan', 'magenta', 'yellow', 'lime', 'orange'] 
    for i, val_flat in enumerate(window_values): 
        val_map = val_flat.reshape(512, 201).cpu().numpy() 
        col = colors[i % len(colors)] 
        plt.contour(T_grid, X_grid, val_map, levels=[0.5], colors=[col], linewidths=1.5, alpha=0.8) 
        c = model.centers[i].detach().cpu().numpy().flatten() 
        plt.plot(c[1], c[0], marker='x', color=col, markersize=8) 
plt.xlabel("t"); plt.ylabel("x"); plt.title("Error & Experts") 

# 4. RADS Distribution 
plt.subplot(1, 4, 4) 
with torch.no_grad(): 
    z_vis = torch.randn(3000, 2, device=device, dtype=dtype) 
    x_vis, t_vis = generator(z_vis) 
    plt.scatter(t_vis.cpu().numpy(), x_vis.cpu().numpy(), s=1, c='blue', alpha=0.3) 
plt.xlabel("t"); plt.ylabel("x"); plt.title("RADS Focus Areas") 
plt.xlim(0, 1); plt.ylim(-1, 1) 

plt.tight_layout() 
plt.savefig('AC_Timed_Final.png', dpi=150) 
print(" Done.")