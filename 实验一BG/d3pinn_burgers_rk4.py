# d3pinn_burgers_rk4_stableA.py
# Stable D3PINN variant (A): NN only learns viscous term u_xx.
# Baseline: WENO5 for convective flux derivative (f_x), central FD for u_xx.
# D3PINN: du/dt = - WENO5_flux_x(u) + nu * uxx_nn

import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import time, math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=6, suppress=True)

# -------------------------
# Problem params
# -------------------------
nu = 0.01 / math.pi        # viscosity
a, b = -1.0, 1.0
T = 0.5

# spatial grid
M = 256
x = np.linspace(a, b, M)
dx = x[1] - x[0]

# initial condition
def u_init(x):
    return -np.sin(np.pi * x)

u0 = u_init(x).astype(np.float64)

# time step selection (safe)
max_u = float(np.max(np.abs(u0))) + 1e-8
dt_conv = dx / max_u
dt_diff = dx**2 / (4.0 * nu + 1e-12)
dt = min(5e-5, 0.4 * min(dt_conv, dt_diff))
nsteps = int(np.ceil(T / dt))
dt = T / nsteps

print(f"M={M}, dx={dx:.3e}, dt={dt:.3e}, nsteps={nsteps}, nu={nu:.3e}")
eps = 1e-12

# -------------------------
# flux and WENO5 (fixed, 3 ghost cells)
# -------------------------
def flux(u):
    return 0.5 * u**2

def weno5_left(v0, v1, v2, v3, v4):
    p0 = (1/3)*v0 - (7/6)*v1 + (11/6)*v2
    p1 = -(1/6)*v1 + (5/6)*v2 + (1/3)*v3
    p2 = (1/3)*v2 + (5/6)*v3 - (1/6)*v4
    b0 = (13/12)*(v0 - 2*v1 + v2)**2 + 0.25*(v0 - 4*v1 + 3*v2)**2
    b1 = (13/12)*(v1 - 2*v2 + v3)**2 + 0.25*(v1 - v3)**2
    b2 = (13/12)*(v2 - 2*v3 + v4)**2 + 0.25*(3*v2 - 4*v3 + v4)**2
    epsw = 1e-6
    a0 = 0.1 / (epsw + b0)**2
    a1 = 0.6 / (epsw + b1)**2
    a2 = 0.3 / (epsw + b2)**2
    wsum = a0 + a1 + a2
    return (a0*p0 + a1*p1 + a2*p2) / wsum

def weno5_right(v0, v1, v2, v3, v4):
    # symmetry: right-biased reconstruction
    return weno5_left(v4, v3, v2, v1, v0)

def weno5_flux_x(u):
    Mlen = len(u)
    f = flux(u)

    # extend with 3 ghost cells (constant extrapolation)
    f_ext = np.zeros(Mlen + 6, dtype=np.float64)
    f_ext[3:-3] = f
    f_ext[0:3] = f[0]
    f_ext[-3:] = f[-1]

    u_ext = np.zeros_like(f_ext)
    u_ext[3:-3] = u
    u_ext[0:3] = u[0]
    u_ext[-3:] = u[-1]

    alpha = max(np.max(np.abs(u)), 1e-6)
    f_plus = 0.5*(0.5*u_ext**2 + alpha * u_ext)
    f_minus = 0.5*(0.5*u_ext**2 - alpha * u_ext)

    # compute interface fluxes f_{i+1/2} for i=0..M
    flux_iface = np.zeros(Mlen + 1, dtype=np.float64)
    # interface i -> index in ext arrays: i + 3
    for i in range(Mlen + 1):
        k = i + 3
        fp = weno5_left(f_plus[k-3], f_plus[k-2], f_plus[k-1], f_plus[k], f_plus[k+1])
        fm = weno5_right(f_minus[k+2], f_minus[k+1], f_minus[k], f_minus[k-1], f_minus[k-2])
        flux_iface[i] = fp + fm

    f_x = np.zeros(Mlen, dtype=np.float64)
    for i in range(Mlen):
        f_x[i] = (flux_iface[i+1] - flux_iface[i]) / dx
    return f_x

# -------------------------
# viscous FD operator (u_xx)
# -------------------------
def fd_u_xx(u_arr):
    uxx = np.zeros_like(u_arr, dtype=np.float64)
    uxx[1:-1] = (u_arr[2:] - 2.0*u_arr[1:-1] + u_arr[:-2]) / (dx**2)
    uxx[0] = 0.0
    uxx[-1] = 0.0
    return uxx

# FD baseline time derivative
def fd_time_derivative(u_arr):
    fx = weno5_flux_x(u_arr)
    uxx = fd_u_xx(u_arr)
    return -fx + nu * uxx

def rk4_step_fd(u):
    k1 = fd_time_derivative(u)
    k2 = fd_time_derivative(u + 0.5*dt*k1)
    k3 = fd_time_derivative(u + 0.5*dt*k2)
    k4 = fd_time_derivative(u + dt*k3)
    u_new = u + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)) or np.max(np.abs(u_new)) > 1e8:
        raise RuntimeError("FD baseline blow-up")
    return u_new

# -------------------------
# NN: local stencil -> predict u_xx only
# -------------------------
class LocalStencilNN_u_xx(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)   # output: u_xx
        )
    def forward(self, stencil):
        return self.net(stencil)

model = LocalStencilNN_u_xx(hidden=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def build_stencils(u_arr):
    st = []
    for i in range(1, M-1):
        st.append([u_arr[i-1], u_arr[i], u_arr[i+1]])
    st = np.array(st, dtype=np.float32)
    return torch.tensor(st, dtype=torch.float32, device=device)  # shape (M-2,3)

def nn_predict_uxx(u_arr):
    st = build_stencils(u_arr)
    with torch.no_grad():
        out = model(st).cpu().numpy().astype(np.float64).reshape(-1)  # length M-2
    uxx = np.zeros_like(u_arr, dtype=np.float64)
    uxx[1:-1] = out[:]  # interior predictions
    # boundaries: fallback to FD
    uxx[0] = 0.0
    uxx[-1] = 0.0
    return uxx

# -------------------------
# Pretrain NN on FD uxx target (initial)
# -------------------------
st_init = build_stencils(u0)
uxx_init = fd_u_xx(u0)
tgt_init = uxx_init[1:-1].astype(np.float32)
tgt_init_t = torch.tensor(tgt_init, dtype=torch.float32, device=device).unsqueeze(1)  # shape (M-2,1)

print("Pretraining NN on initial FD u_xx targets...")
for ep in range(600):
    opt.zero_grad()
    pred = model(st_init)  # (M-2,1)
    loss = loss_fn(pred, tgt_init_t)
    loss.backward()
    opt.step()
    if ep % 150 == 0:
        print(f" pretrain ep {ep}, loss {loss.item():.3e}")
print("Pretrain done.")

# -------------------------
# Time integration: baseline FD + D3PINN (stable A)
# -------------------------
u_fd = u0.copy()
u_d3 = u0.copy()

# save times
save_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
save_steps = set([int(round(t / dt)) for t in save_times])

times_out = []
errs_d3_vs_fd = []

# online micro-train settings (teacher = FD u_xx)
online_microtrain = True
micro_every = 50
micro_steps = 5

print("Starting time integration...")
t0 = time.time()
try:
    for n in range(1, nsteps+1):
        # FD baseline step
        u_fd = rk4_step_fd(u_fd)

        # D3PINN step: convective part via WENO5 (same as baseline), viscous via NN
        # Stage 1
        shock1 = weno5_flux_x(u_d3)
        uxx_nn1 = nn_predict_uxx(u_d3)
        k1 = - shock1 + nu * uxx_nn1

        # Stage 2
        u_tmp = u_d3 + 0.5 * dt * k1
        shock2 = weno5_flux_x(u_tmp)
        uxx_nn2 = nn_predict_uxx(u_tmp)
        k2 = - shock2 + nu * uxx_nn2

        # Stage 3
        u_tmp = u_d3 + 0.5 * dt * k2
        shock3 = weno5_flux_x(u_tmp)
        uxx_nn3 = nn_predict_uxx(u_tmp)
        k3 = - shock3 + nu * uxx_nn3

        # Stage 4
        u_tmp = u_d3 + dt * k3
        shock4 = weno5_flux_x(u_tmp)
        uxx_nn4 = nn_predict_uxx(u_tmp)
        k4 = - shock4 + nu * uxx_nn4

        u_new = u_d3 + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)) or np.max(np.abs(u_new)) > 1e8:
            raise RuntimeError(f"D3PINN blow-up at step {n}")
        u_d3 = u_new

        # optional online micro-train using FD teacher uxx
        if online_microtrain and (n % micro_every == 0):
            st_tr = build_stencils(u_fd)
            uxx_tr = fd_u_xx(u_fd)
            targets_tr = uxx_tr[1:-1].astype(np.float32).reshape(-1,1)
            tgt_tr_t = torch.tensor(targets_tr, dtype=torch.float32, device=device)
            for m in range(micro_steps):
                opt.zero_grad()
                pred = model(st_tr)
                loss = loss_fn(pred, tgt_tr_t)
                loss.backward()
                opt.step()

        if n in save_steps:
            tcur = n * dt
            denom = max(np.linalg.norm(u_fd), eps)
            err = np.linalg.norm(u_d3 - u_fd) / denom
            times_out.append(tcur)
            errs_d3_vs_fd.append(err)
            print(f"t={tcur:.3f}  D3_vs_FD_relL2={err:.3e}")

except RuntimeError as e:
    print("Aborted due to runtime error:", e)

t1 = time.time()
print("Finished integration in {:.2f}s".format(t1 - t0))

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x, u_fd, label='FD WENO5 baseline')
plt.plot(x, u_d3, '--', label='D3PINN (NN learns u_xx)')
plt.title(f"Solution at t={T:.3f}")
plt.legend()
plt.subplot(1,2,2)
plt.plot(times_out, errs_d3_vs_fd, '-s', label='D3 vs FD (rel L2)')
plt.xlabel('time')
plt.ylabel('relative L2 error')
plt.title('Error vs time (D3 vs FD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Compare u_xx at final time
uxx_fd_final = fd_u_xx(u_fd)
uxx_nn_final = nn_predict_uxx(u_d3)



plt.figure(figsize=(8,4))
plt.plot(x, uxx_fd_final, label='FD u_xx')
plt.plot(x, uxx_nn_final, '--', label='NN u_xx')
plt.legend()
plt.title('Comparison of u_xx at final time')
plt.show()

