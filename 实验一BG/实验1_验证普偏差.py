import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1. 构造数据集
# ======================================================
N = 2000
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + 0.2*np.sin(50*np.pi*x)  # 低频 + 高频

x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ======================================================
# 2. 定义网络
# ======================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()

# ======================================================
# 3. 优化器与损失
# ======================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 保存不同阶段的预测
snapshots = {}
snapshot_steps = [10, 100, 500, 2000, 5000]

# ======================================================
# 4. 训练循环
# ======================================================
for epoch in range(6000):
    optimizer.zero_grad()
    pred = model(x_t)
    loss = loss_fn(pred, y_t)
    loss.backward()
    optimizer.step()
    
    if epoch in snapshot_steps:
        snapshots[epoch] = model(x_t).detach().numpy()
    
    if (epoch % 1000 == 0):
        print(f"Epoch {epoch}, loss = {loss.item():.6f}")

# ======================================================
# 5. 画图：不同训练阶段的拟合情况
# ======================================================
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'k', label="True function", linewidth=2)

for ep, pred in snapshots.items():
    plt.plot(x, pred, label=f'Epoch {ep}')

plt.legend()
plt.title("Learning Dynamics (Spectral Bias)")
plt.show()

# ======================================================
# 6. 频谱分析（FFT）
# ======================================================
def plot_spectrum(signal, title):
    fft_vals = np.abs(np.fft.rfft(signal.flatten()))
    freqs = np.fft.rfftfreq(len(signal), d=1/len(signal))
    plt.plot(freqs, fft_vals)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

plt.figure(figsize=(12, 8))

i = 1
for ep, pred in snapshots.items():
    plt.subplot(2, 3, i)
    plot_spectrum(pred, f"Spectrum @ epoch {ep}")
    i += 1

plt.tight_layout()
plt.show()
