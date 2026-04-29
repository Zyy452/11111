# d3pinn_heat_rk4_fixed.py
# Fixed and robust version for D3PINN prototype (1D heat equation)
# - conservative dt choice
# - float64 FD baseline
# - local-stencil NN approximates u_xx (pretrain + optional online micro-train)
# - RK4 time stepping for both baseline and D3PINN
# - safety checks for NaN/inf/overflow
# Author: ChatGPT (for your experiment)

import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=6, suppress=True)

# -------------------------
# PDE / grid / time params
# -------------------------
kappa = 0.1       # diffusion coef
L = 1.0
T = 0.5

M = 128           # spatial points
x = np.linspace(0.0, L, M)
dx = x[1] - x[0]

# conservative dt for explicit FD + RK4 (heat eq stability)
dt_cfl = dx**2 / (4.0 * kappa)   # conservative factor 4 in denom
dt = min(1e-4, dt_cfl)
nsteps = int(np.ceil(T / dt))
dt = T / nsteps   # adjust so that nsteps*dt == T

print(f"Grid M={M}, dx={dx:.3e}, chosen dt={dt:.3e}, nsteps={nsteps}, kappa={kappa}")

eps = 1e-12

# analytic solution for heat eq with u(x,0)=sin(pi x), Dirichlet 0 at boundaries
def u_exact(x, t):
    return np.exp(- (np.pi**2) * kappa * t) * np.sin(np.pi * x)

# initial
u0 = u_exact(x, 0).astype(np.float64)

# -------------------------
# FD baseline functions (numpy float64)
# -------------------------
def fd_u_xx(u_arr):
    # second derivative with Dirichlet BC u(0)=u(L)=0
    u_xx = np.zeros_like(u_arr, dtype=np.float64)
    # interior central difference
    u_xx[1:-1] = (u_arr[2:] - 2.0*u_arr[1:-1] + u_arr[:-2]) / (dx**2)
    # boundaries: assume Dirichlet u=0 => second derivative may be approximated 0 or one-sided
    u_xx[0] = 0.0
    u_xx[-1] = 0.0
    return u_xx

def rk4_step_fd(u):
    k1 = kappa * fd_u_xx(u)
    k2 = kappa * fd_u_xx(u + 0.5*dt*k1)
    k3 = kappa * fd_u_xx(u + 0.5*dt*k2)
    k4 = kappa * fd_u_xx(u + dt*k3)
    u_new = u + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    # sanity checks
    if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)) or np.max(np.abs(u_new)) > 1e8:
        raise RuntimeError("Numerical blow-up detected in FD RK4 integration.")
    return u_new

# -------------------------
# Local-stencil NN (PyTorch) mapping [u_{i-1}, u_i, u_{i+1}] -> u_xx_i
# -------------------------
class LocalStencilNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, stencil):  # stencil: (N_points, 3) float32
        return self.net(stencil)

# build model
model = LocalStencilNN(hidden=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Helper to build stencils (numpy -> tensor)
def build_stencils(u_arr):
    # u_arr: numpy array float64 length M
    stencils = []
    for i in range(1, M-1):
        stencils.append([u_arr[i-1], u_arr[i], u_arr[i+1]])
    st = np.array(stencils, dtype=np.float32)  # convert to float32 for torch
    return torch.tensor(st, dtype=torch.float32, device=device)  # shape (M-2,3)

def nn_predict_u_xx(u_arr):
    # u_arr: numpy float64
    st = build_stencils(u_arr)               # (M-2,3) float32
    with torch.no_grad():
        pred = model(st).cpu().numpy().flatten().astype(np.float64)
    uxx = np.zeros_like(u_arr, dtype=np.float64)
    uxx[1:-1] = pred
    uxx[0] = 0.0
    uxx[-1] = 0.0
    return uxx

# -------------------------
# Pretrain NN on FD targets (use initial u0 or a few snapshots)
# -------------------------
# prepare training dataset from initial u0
st = build_stencils(u0)  # float32 tensor
targets = fd_u_xx(u0)[1:-1].reshape(-1,1).astype(np.float32)  # float32
tgt = torch.tensor(targets, dtype=torch.float32, device=device)

print("Pretraining NN on initial FD stencil targets...")
for ep in range(400):
    opt.zero_grad()
    pred = model(st)
    loss = loss_fn(pred, tgt)
    loss.backward()
    opt.step()
    if ep % 100 == 0:
        print(f" pretrain ep {ep}, loss {loss.item():.3e}")
print("Pretrain done.")

# -------------------------
# Time integration loop (both FD baseline and D3PINN)
# -------------------------
# initial copies
u_fd = u0.copy()
u_d3 = u0.copy()

# record errors at specific times
save_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
save_steps = set([int(np.round(t / dt)) for t in save_times])

times_out = []
errs_fd = []
errs_d3 = []

# online micro-train settings (optional)
online_microtrain = True
micro_every = 50       # every micro_every steps, do micro-steps
micro_steps = 5        # gradient steps per micro-train
micro_lr = 1e-3

print("Beginning time integration...")
t0 = time.time()
for n in range(1, nsteps+1):
    # baseline FD RK4
    try:
        u_fd = rk4_step_fd(u_fd)
    except RuntimeError as e:
        print("FD baseline blow-up at step", n)
        raise

    # D3PINN: use NN to predict u_xx then RK4
    # k1
    uxx1 = nn_predict_u_xx(u_d3)
    k1 = kappa * uxx1
    # k2
    u_tmp = u_d3 + 0.5*dt*k1
    uxx2 = nn_predict_u_xx(u_tmp)
    k2 = kappa * uxx2
    # k3
    u_tmp = u_d3 + 0.5*dt*k2
    uxx3 = nn_predict_u_xx(u_tmp)
    k3 = kappa * uxx3
    # k4
    u_tmp = u_d3 + dt*k3
    uxx4 = nn_predict_u_xx(u_tmp)
    k4 = kappa * uxx4

    u_new = u_d3 + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    # safety
    if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)) or np.max(np.abs(u_new)) > 1e8:
        raise RuntimeError(f"D3PINN numerical blow-up at step {n}, max abs {np.max(np.abs(u_new)):.3e}")
    u_d3 = u_new

    # optional online micro-train: use FD baseline as teacher to refine NN periodically
    if online_microtrain and (n % micro_every == 0):
        # prepare training batch from current FD solution (trusted)
        st_tr = build_stencils(u_fd)
        tg_tr = fd_u_xx(u_fd)[1:-1].reshape(-1,1).astype(np.float32)
        tg_tr_t = torch.tensor(tg_tr, dtype=torch.float32, device=device)
        # do a few gradient steps
        for m in range(micro_steps):
            opt.zero_grad()
            pred = model(st_tr)
            loss = loss_fn(pred, tg_tr_t)
            loss.backward()
            opt.step()
        # optionally print microtrain loss
        # print(f"online microtrain at step {n}, loss {loss.item():.3e}")

    # save errors at selected times
    if n in save_steps:
        tcur = n * dt
        u_true = u_exact(x, tcur)
        denom = max(np.linalg.norm(u_true), eps)
        err_fd = np.linalg.norm(u_fd - u_true) / denom
        err_d3 = np.linalg.norm(u_d3 - u_true) / denom
        times_out.append(tcur)
        errs_fd.append(err_fd)
        errs_d3.append(err_d3)
        print(f"t={tcur:.3f}  FD_err={err_fd:.3e} D3_err={err_d3:.3e}")

t1 = time.time()
print("Integration finished in {:.2f}s".format(t1 - t0))

# -------------------------
# plot error vs time
# -------------------------
plt.figure(figsize=(6,4))
plt.plot(times_out, errs_fd, '-o', label='FD baseline')
plt.plot(times_out, errs_d3, '-s', label='D3PINN (NN spatial op)')
plt.xlabel('time')
plt.ylabel('relative L2 error')
plt.legend()
plt.title('Error vs time: D3PINN vs FD baseline')
plt.grid(True)
plt.show()
