import torch 
import torch.nn as nn 
import numpy as np 
import scipy.io 
import os 
import time 
import datetime 

# ========================================================== 
# 0. 基础配置与目录设置 
# ========================================================== 
BASE_SAVE_DIR = "/3241003007/zy/save"
EXP_DIR = os.path.join(BASE_SAVE_DIR, "AC_Experiment")
os.makedirs(EXP_DIR, exist_ok=True)
print(f"📁 实验数据将保存在: {EXP_DIR}")

os.environ['KMP_DUPLICATE_LIB_OK']='True' 

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}，正在使用 GPU 训练！")
else:
    device = torch.device("cpu")
    print(f"⚠️ 未检测到 GPU，已回退至 CPU 训练。")

dtype = torch.float64 
torch.set_default_dtype(dtype) 
torch.manual_seed(1234) 
np.random.seed(1234) 

# ========================================================== 
# 1. 数据读取与网格生成
# ========================================================== 
EPSILON = 0.0001 
GAMMA = 5.0 

def get_exact_data():
    file_path = "/Users/zyy/Library/CloudStorage/OneDrive-cuit.edu.cn/代码/zy/实验二AC/AC.mat" 
    if not os.path.exists(file_path): 
        file_path = "/3241003007/zy/实验二AC/AC.mat"
        
    try: 
        data = scipy.io.loadmat(file_path) 
        x = data["x"].flatten() 
        t = data["tt"].flatten() 
        uu = data["uu"] 
        if uu.shape[0] != len(x): 
            uu = uu.T 
        return x, t, uu
    except: 
        print("❌ 警告: 找不到 AC.mat，使用伪数据占位 (仅供代码跑通测试)") 
        x = np.linspace(-1, 1, 512) 
        t = np.linspace(0, 1, 201) 
        uu = np.zeros((512, 201)) 
        return x, t, uu

x_exact, t_exact, u_exact_all = get_exact_data()

X_mesh, T_mesh = np.meshgrid(x_exact, t_exact, indexing='ij') 
X_star = torch.tensor(X_mesh.flatten()[:, None], device=device, dtype=dtype) 
T_star = torch.tensor(T_mesh.flatten()[:, None], device=device, dtype=dtype) 

# ========================================================== 
# 2. 网络定义 (Scaled Tanh & 动态架构) 
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
        t = torch.sigmoid(raw[:, 1:2]) 
        return x, t 

class PeriodicEmbedding(nn.Module): 
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
        
        # 这里的零初始化写得非常对！保证了新网络初始输出为 0
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
        self.subnets = nn.ModuleList([DeepSubNet(input_dim=3, hidden=hidden).to(device)]) 
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

    # ================= 🚀 核心修复区域 =================
    def forward(self, x, t): 
        x_emb = self.embed(x) 
        xt = torch.cat([x, t], dim=1) 
        
        u_out = 0 
        for i in range(len(self.subnets)): 
            diff_sq = (xt - self.centers[i])**2 
            gamma = torch.exp(self.log_gammas[i]) 
            
            # 使用高斯径向基函数直接作为权重（不进行 Softmax 归一化）
            # 这样确保了 Proposition 1 中“纯加法基底”的数学性质
            weight = torch.exp(-torch.sum(gamma * diff_sq, dim=1, keepdim=True)) 
            u_out += weight * self.subnets[i](x_emb, t) 
            
        u_0 = x**2 * torch.cos(np.pi * x) 
        u_final = torch.tanh(t) * u_out + u_0 
        return u_final 
    # ===================================================

# ========================================================== 
# 3. 损失函数与工具
# ========================================================== 
def pde_residual(model, x, t): 
    u = model(x, t) 
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] 
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] 
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0] 
    return u_t - EPSILON * u_xx + GAMMA * (u**3 - u) 

def get_uniform_points(N): 
    x = (-1 + 2 * torch.rand(N, 1, device=device, dtype=dtype)).requires_grad_(True) 
    t = torch.rand(N, 1, device=device, dtype=dtype).requires_grad_(True) 
    return x, t 

# ========================================================== 
# 4. Phase 1: RADS 对抗训练
# ========================================================== 
ADAM_EPOCHS = 25000 
BATCH_SIZE = 6000 

model = DynamicABPINN(hidden=64).to(device) 
generator = RADS_Generator().to(device) 

optim_pinn = torch.optim.Adam(model.parameters(), lr=1e-3) 
optim_gen = torch.optim.Adam(generator.parameters(), lr=1e-3) 

loss_history = [] 
err_history = [] 
iters = [] 

print(f"\n>>> Phase 1: RADS Adversarial Training ({ADAM_EPOCHS} Epochs)...") 
phase1_start = time.time() 
start_time = time.time() 

for epoch in range(ADAM_EPOCHS + 1): 
    # 训练生成器 (Generator)
    if epoch % 5 == 0: 
        for _ in range(2): 
            optim_gen.zero_grad() 
            z = torch.randn(BATCH_SIZE // 2, 2, device=device, dtype=dtype) 
            x_gen, t_gen = generator(z) 
            loss_gen = -torch.mean(pde_residual(model, x_gen, t_gen)**2) 
            loss_gen.backward() 
            optim_gen.step() 

    # 训练 PINN (Model)
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
    
    # 动态添加 Expert 子网络
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
                    print(f" 🌟 Added Expert @ (x={x_train[idx].item():.2f}, t={t_train[idx].item():.2f}) | Res={max_res:.4f}") 
                    model.add_subdomain([x_train[idx].item(), t_train[idx].item()]) 
                    # 重置优化器以包含新加入的参数
                    optim_pinn = torch.optim.Adam(model.parameters(), lr=8e-4) 

    # 日志打印与数据记录
    if epoch % 500 == 0: 
        model.eval() 
        with torch.no_grad(): 
            u_check = model(X_star, T_star).cpu().numpy().reshape(512, 201) 
            curr_err = np.linalg.norm(u_exact_all - u_check) / np.linalg.norm(u_exact_all) 
        model.train() 
        loss_history.append(loss.item()) 
        err_history.append(curr_err) 
        iters.append(epoch) 
        
        elapsed = time.time() - start_time 
        avg_time_per_step = elapsed / 500 if epoch > 0 else 0 
        remaining_steps = ADAM_EPOCHS - epoch 
        eta_seconds = avg_time_per_step * remaining_steps 
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds))) 
        print(f"Ep {epoch:5d}/{ADAM_EPOCHS} | Loss: {loss.item():.2e} | Err: {curr_err:.4f} | Experts: {len(model.subnets)} | ETA: {eta_str}") 
        start_time = time.time() 

phase1_time = time.time() - phase1_start 
print(f"✅ Phase 1 Finished in {str(datetime.timedelta(seconds=int(phase1_time)))}") 

# ========================================================== 
# 5. Phase 2: L-BFGS 微调
# ========================================================== 
print("\n>>> Phase 2: L-BFGS Refinement ...") 

N_LBFGS = 15000 
rand_idx = torch.randperm(X_star.shape[0])[:N_LBFGS]
x_final = X_star[rand_idx].clone().detach().requires_grad_(True) 
t_final = T_star[rand_idx].clone().detach().requires_grad_(True) 

print(f" L-BFGS Start (Subsampled {N_LBFGS} points). Max Iter: 5000. Please wait...") 
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

iter_count_lbfgs = 0
def closure(): 
    global iter_count_lbfgs
    lbfgs.zero_grad() 
    res = pde_residual(model, x_final, t_final) 
    loss = torch.mean(res**2)  
    loss.backward() 
    
    iter_count_lbfgs += 1
    if iter_count_lbfgs % 500 == 0:
        print(f"  L-BFGS Iter {iter_count_lbfgs:4d}/5000 | Loss: {loss.item():.5e}")
        
    return loss 

lbfgs.step(closure) 

phase2_time = time.time() - phase2_start 
print(f"✅ Phase 2 Finished in {str(datetime.timedelta(seconds=int(phase2_time)))}") 

# ========================================================== 
# 6. 数据提取与保存 (供外部脚本做消融对比图使用)
# ========================================================== 
model.eval() 
generator.eval()

with torch.no_grad(): 
    u_pred = model(X_star, T_star).cpu().numpy().reshape(512, 201) 
    
    z_vis = torch.randn(3000, 2, device=device, dtype=dtype) 
    x_vis, t_vis = generator(z_vis) 
    x_vis_np = x_vis.cpu().numpy()
    t_vis_np = t_vis.cpu().numpy()

final_error = np.linalg.norm(u_exact_all - u_pred) / np.linalg.norm(u_exact_all) 
total_time = phase1_time + phase2_time

centers_final = torch.stack([c.view(-1).detach() for c in model.centers]).cpu().numpy()
gammas_final = torch.stack([torch.exp(lg).view(-1).detach() for lg in model.log_gammas]).cpu().numpy()

# 核心：将用于画图的必要指标保存下来
save_data = {
    "config_name": "RAD_AB_PINN_AC",
    "x_exact": x_exact,
    "t_exact": t_exact,
    "u_exact_all": u_exact_all,
    "u_pred": u_pred,
    "x_vis": x_vis_np,
    "t_vis": t_vis_np,
    "centers": centers_final,   
    "gammas": gammas_final,     
    "final_error": final_error,
    "time_adam": phase1_time,
    "time_lbfgs": phase2_time,
    "time_total": total_time,
    "err_history": np.array(err_history),    # 供对比图使用的 L2 误差曲线
    "loss_history": np.array(loss_history),  # 供对比图使用的 Loss 曲线
    "iters": np.array(iters)                 # 供对比图使用的 X 轴坐标
}

# 按要求保存文件和打印
DATA_SAVE_PATH = os.path.join(EXP_DIR, "ac_rads_abpinn_results2.pt")
torch.save(save_data, DATA_SAVE_PATH)
print(f"\n✨✨ RAD-AB-PINN (Ours) Final Rel L2 Error: {final_error:.4e} | 保存至 {DATA_SAVE_PATH}")