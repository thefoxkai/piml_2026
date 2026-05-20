"""
pinn_multi_traj.py
==================
PINN vs plain NN on the damped mass-spring-damper (MCK),
trained on multiple trajectories with different initial conditions.

System   : ẍ + 2δẋ + ω₀²x = 0   (δ=2, ω₀=20, underdamped)
Network  : (t, x₀, ẋ₀) → x(t; x₀, ẋ₀)   — ICs are network inputs
Data     : --n_train trajectories × 20 pts from t ∈ [0, 0.36]
           ICs drawn uniformly: x₀ ∈ [−0.5, 0.5],  ẋ₀ ∈ [−0.1, 0.1]
Eval     : 8 held-out ICs (same ranges, different RNG seed)

PINN losses
  - data  : MSE on observed trajectory points
  - phys  : ODE residual on collocation pts (t, x₀, ẋ₀ from training set)
  - ic    : x(0;x₀,ẋ₀)=x₀  and  ẋ(0;x₀,ẋ₀)=ẋ₀

VanillaNN: same architecture, data loss only.

Key insight: the PINN uses the ODE to constrain the solution manifold and
generalises to unseen ICs. The plain NN can only interpolate/extrapolate
from the training trajectories it has seen.

Usage
-----
  python excercises/pinn/pinn_multi_traj.py --n_train 10
  python excercises/pinn/pinn_multi_traj.py --n_train 50 --seed 7
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from systems import MassSpringDamper

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "PINN": "#3498db", "NN": "#e74c3c"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Multi-trajectory PINN vs NN on MCK")
parser.add_argument("--n_train", type=int, default=10,
                    help="Number of training trajectories (default 10)")
parser.add_argument("--seed", type=int, default=0,
                    help="Master RNG seed (default 0)")
args = parser.parse_args()

SEED         = args.seed
N_TRAIN_TRAJ = max(1, args.n_train)
N_EVAL_TRAJ  = 8

torch.manual_seed(SEED)
rng_train = np.random.default_rng(SEED)
rng_eval  = np.random.default_rng(SEED + 999)   # separate stream → always held-out

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"n_train={N_TRAIN_TRAJ}  seed={SEED}  device={DEVICE}")

# ---------------------------------------------------------------------------
# System + analytical solution with arbitrary ICs
# ---------------------------------------------------------------------------

mck = MassSpringDamper(delta=2.0, omega0=20.0)
δ, ω0, ω = mck.delta, mck.omega0, mck.omega


def exact(t: np.ndarray, x0: float, v0: float) -> np.ndarray:
    """Closed-form: x'' + 2δx' + ω₀²x = 0,  x(0)=x0, x'(0)=v0."""
    return np.exp(-δ * t) * (x0 * np.cos(ω * t) + (v0 + δ * x0) / ω * np.sin(ω * t))


# ---------------------------------------------------------------------------
# Sample ICs
# ---------------------------------------------------------------------------

X0_RANGE = (-0.5,  0.5)
V0_RANGE = (-0.1,  0.1)

x0_train = rng_train.uniform(*X0_RANGE, N_TRAIN_TRAJ)
v0_train = rng_train.uniform(*V0_RANGE, N_TRAIN_TRAJ)

x0_eval  = rng_eval.uniform(*X0_RANGE, N_EVAL_TRAJ)
v0_eval  = rng_eval.uniform(*V0_RANGE, N_EVAL_TRAJ)

# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

T_TRAIN    = (0.0, 0.36)
T_EVAL     = (0.0, 1.0)
N_PTS_TRAJ = 20    # observations per training trajectory
N_EVAL_PTS = 500   # time-grid density for evaluation

t_pts = np.linspace(*T_TRAIN, N_PTS_TRAJ)

_t_list, _x0_list, _v0_list, _x_list = [], [], [], []
for x0, v0 in zip(x0_train, v0_train):
    _t_list.append(t_pts)
    _x0_list.append(np.full(N_PTS_TRAJ, x0))
    _v0_list.append(np.full(N_PTS_TRAJ, v0))
    _x_list.append(exact(t_pts, x0, v0))

train_t  = np.concatenate(_t_list)
train_x0 = np.concatenate(_x0_list)
train_v0 = np.concatenate(_v0_list)
train_x  = np.concatenate(_x_list)

t_tr  = torch.tensor(train_t[:,  None], dtype=torch.float32, device=DEVICE)
x0_tr = torch.tensor(train_x0[:, None], dtype=torch.float32, device=DEVICE)
v0_tr = torch.tensor(train_v0[:, None], dtype=torch.float32, device=DEVICE)
x_tr  = torch.tensor(train_x[:,  None], dtype=torch.float32, device=DEVICE)

# IC boundary rows (one per training trajectory)
t_ic  = torch.zeros(N_TRAIN_TRAJ, 1, dtype=torch.float32, device=DEVICE)
x0_ic = torch.tensor(x0_train[:, None], dtype=torch.float32, device=DEVICE)
v0_ic = torch.tensor(v0_train[:, None], dtype=torch.float32, device=DEVICE)


def sample_colloc(n: int = 2000):
    """t ∈ [0, T_EVAL], ICs sampled from the training set."""
    t_c  = torch.rand(n, 1, device=DEVICE) * T_EVAL[1]
    idx  = torch.randint(0, N_TRAIN_TRAJ, (n,))
    x0_c = x0_ic[idx]
    v0_c = v0_ic[idx]
    return t_c, x0_c, v0_c


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _mlp(in_dim, hidden, out_dim, n_layers, act=nn.Tanh):
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class MultiICPINN(nn.Module):
    """Physics-Informed NN: (t, x₀, ẋ₀) → x(t; x₀, ẋ₀)."""

    def __init__(self, delta, omega0, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.delta, self.omega0, self.t_scale = delta, omega0, t_scale
        self.net = _mlp(3, hidden_dim, 1, n_layers)

    def forward(self, t, x0, v0):
        return self.net(torch.cat([t / self.t_scale, x0, v0], dim=-1))

    def residual(self, t_c, x0_c, v0_c):
        t_r = t_c.detach().requires_grad_(True)
        x   = self.forward(t_r, x0_c, v0_c)
        xd  = torch.autograd.grad(x.sum(),  t_r, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), t_r, create_graph=True)[0]
        return (xdd + 2*self.delta*xd + self.omega0**2 * x) / self.omega0**2

    def ic_loss(self, t0, x0, v0):
        t_r = t0.detach().requires_grad_(True)
        x   = self.forward(t_r, x0, v0)
        xd  = torch.autograd.grad(x.sum(), t_r, create_graph=True)[0]
        return ((x - x0)**2).mean() + ((xd - v0)**2).mean() / self.omega0**2

    def total_loss(self, t_tr, x0_tr, v0_tr, x_tr,
                   t_c, x0_c, v0_c,
                   t_ic, x0_ic, v0_ic,
                   w_data=1.0, w_phys=1.0, w_ic=1.0):
        ld  = ((self.forward(t_tr, x0_tr, v0_tr) - x_tr)**2).mean()
        lp  = (self.residual(t_c, x0_c, v0_c)**2).mean()
        lic = self.ic_loss(t_ic, x0_ic, v0_ic)
        return w_data*ld + w_phys*lp + w_ic*lic, ld, lp, lic


class MultiICNN(nn.Module):
    """Plain NN: (t, x₀, ẋ₀) → x(t; x₀, ẋ₀), no physics."""

    def __init__(self, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.t_scale = t_scale
        self.net = _mlp(3, hidden_dim, 1, n_layers)

    def forward(self, t, x0, v0):
        return self.net(torch.cat([t / self.t_scale, x0, v0], dim=-1))


# ---------------------------------------------------------------------------
# Train PINN  (data warm-up → data + physics + IC)
# ---------------------------------------------------------------------------

EPOCHS_PINN, WARMUP = 25_000, 0
wd=1.0
wp=10.0
wi=0.0

pinn = MultiICPINN(mck.delta, mck.omega0, t_scale=T_EVAL[1]).to(DEVICE)
opt  = torch.optim.Adam(pinn.parameters(), lr=1e-3)
sch  = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[int(EPOCHS_PINN * .5), int(EPOCHS_PINN * .8)], gamma=0.5)

print("\nTraining PINN ...")
for ep in range(1, EPOCHS_PINN + 1):
    opt.zero_grad()
    if ep <= WARMUP:
        loss = ((pinn(t_tr, x0_tr, v0_tr) - x_tr)**2).mean()
    else:
        t_c, x0_c, v0_c = sample_colloc(2000)
        loss, *_ = pinn.total_loss(t_tr, x0_tr, v0_tr, x_tr,
                                   t_c, x0_c, v0_c,
                                    t_ic, x0_ic, v0_ic,
                                    w_data=wd,w_phys=wp,w_ic=wi)
    loss.backward(); opt.step(); sch.step()
    if ep % 5000 == 0:
        t_c, x0_c, v0_c = sample_colloc(2000)
        _, ld, lp, lic = pinn.total_loss(t_tr, x0_tr, v0_tr, x_tr,
                                         t_c, x0_c, v0_c,
                                         t_ic, x0_ic, v0_ic,
                                         w_data=wd,w_phys=wp,w_ic=wi)
        print(f"  ep {ep:5d}  data={ld.item():.3e}  "
              f"phys={lp.item():.3e}  ic={lic.item():.3e}")

# ---------------------------------------------------------------------------
# Train NN  (data only)
# ---------------------------------------------------------------------------

EPOCHS_NN = 5_000

nn_model = MultiICNN(t_scale=T_TRAIN[1]).to(DEVICE)
opt_nn   = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
sch_nn   = torch.optim.lr_scheduler.MultiStepLR(
    opt_nn, milestones=[int(EPOCHS_NN * .5), int(EPOCHS_NN * .8)], gamma=0.5)

print("\nTraining NN ...")
for ep in range(1, EPOCHS_NN + 1):
    opt_nn.zero_grad()
    loss_nn = ((nn_model(t_tr, x0_tr, v0_tr) - x_tr)**2).mean()
    loss_nn.backward(); opt_nn.step(); sch_nn.step()
print(f"  final loss = {loss_nn.item():.4e}")

# ---------------------------------------------------------------------------
# Evaluate on held-out ICs
# ---------------------------------------------------------------------------

t_eval_np = np.linspace(*T_EVAL, N_EVAL_PTS)
split = T_TRAIN[1]
in_m  = t_eval_np <= split
out_m = t_eval_np >  split

t_ev_t = torch.tensor(t_eval_np[:, None], dtype=torch.float32, device=DEVICE)

eval_results = []
for x0, v0 in zip(x0_eval, v0_eval):
    x_true = exact(t_eval_np, x0, v0)
    x0_ev  = torch.full_like(t_ev_t, x0)
    v0_ev  = torch.full_like(t_ev_t, v0)
    with torch.no_grad():
        x_pinn = pinn(t_ev_t, x0_ev, v0_ev).cpu().numpy().squeeze()
        x_nn   = nn_model(t_ev_t, x0_ev, v0_ev).cpu().numpy().squeeze()
    eval_results.append({
        "x0": x0, "v0": v0,
        "true": x_true, "pinn": x_pinn, "nn": x_nn,
        "pinn_in":  float(np.sqrt(np.mean((x_pinn[in_m]  - x_true[in_m])**2))),
        "pinn_out": float(np.sqrt(np.mean((x_pinn[out_m] - x_true[out_m])**2))),
        "nn_in":    float(np.sqrt(np.mean((x_nn[in_m]    - x_true[in_m])**2))),
        "nn_out":   float(np.sqrt(np.mean((x_nn[out_m]   - x_true[out_m])**2))),
    })

print(f"\n{'IC':>18} {'PINN interp':>12} {'PINN extrap':>12} "
      f"{'NN interp':>12} {'NN extrap':>12}")
print("-" * 70)
for r in eval_results:
    print(f"({r['x0']:+.3f},{r['v0']:+.4f})  "
          f"{r['pinn_in']:>12.4f} {r['pinn_out']:>12.4f} "
          f"{r['nn_in']:>12.4f} {r['nn_out']:>12.4f}")
avg_pinn = np.mean([r["pinn_out"] for r in eval_results])
avg_nn   = np.mean([r["nn_out"]   for r in eval_results])
print(f"\nAvg extrap RMSE — PINN: {avg_pinn:.4f}  NN: {avg_nn:.4f}  "
      f"(NN/PINN = {avg_nn / (avg_pinn + 1e-12):.1f}×)")

# ---------------------------------------------------------------------------
# Plot — 2×4 grid of held-out ICs
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
fig.suptitle(
    f"Multi-traj PINN vs NN — MCK  ({N_TRAIN_TRAJ} training trajectories, seed={SEED})\n"
    f"8 held-out initial conditions  |  shaded = extrapolation region",
    fontsize=11, fontweight="bold")

for ax, r in zip(axes.flat, eval_results):
    ax.plot(t_eval_np, r["true"], color=COLORS["True"],
            lw=1, ls="--", label="True", zorder=5)
    ax.plot(t_eval_np, r["pinn"], color=COLORS["PINN"], lw=1.8,
            label=f"PINN (ext={r['pinn_out']:.3f})")
    ax.plot(t_eval_np, r["nn"],   color=COLORS["NN"],   lw=1.8,
            label=f"NN   (ext={r['nn_out']:.3f})")
    ax.axvline(split, color="gray", ls=":", lw=1.2)
    ax.axvspan(split, t_eval_np[-1], alpha=0.07, color="gray")
    ax.set_title(f"x₀={r['x0']:+.3f}, ẋ₀={r['v0']:+.4f}", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylabel("x(t)", fontsize=8)

for ax in axes[1]:
    ax.set_xlabel("t (s)", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
            f"pinn_multi_traj_n{N_TRAIN_TRAJ}_s{SEED}.png"),
            dpi=130, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Animation — first held-out IC, 3-row spring-mass physical view
# ---------------------------------------------------------------------------

_r   = eval_results[0]
_s   = max(1, N_EVAL_PTS // 200)
_ta  = t_eval_np[::_s]
_nf  = len(_ta)
_xd  = {"True": _r["true"][::_s], "PINN": _r["pinn"][::_s], "NN": _r["nn"][::_s]}
_rows = {"True": 2.0, "PINN": 1.0, "NN": 0.0}
_WX, _R = -0.7, 0.04


def _sp1d(x0, x1, y, n=8, a=0.015):
    N = n*2+2
    xs = np.linspace(x0, x1, N)
    ys = np.full(N, y, dtype=float)
    ys[1:-1:2] += a; ys[2:-1:2] -= a
    return xs, ys


fig_a, (ax_ph, ax_tr) = plt.subplots(1, 2, figsize=(13, 5))
fig_a.suptitle(
    f"PINN vs NN — MCK  ({N_TRAIN_TRAJ} train traj., seed={SEED})  "
    f"eval IC: x₀={_r['x0']:+.3f}, ẋ₀={_r['v0']:+.4f}",
    fontsize=11)

ax_ph.set_xlim(-0.95, 0.65); ax_ph.set_ylim(-0.5, 2.6)
ax_ph.set_aspect("equal"); ax_ph.axis("off")
ax_ph.vlines(_WX, -0.3, 2.6, colors="k", lw=4)
for nm, y in _rows.items():
    ax_ph.text(_WX - 0.03, y, nm, ha="right", va="center",
               color=COLORS[nm], fontsize=9, fontweight="bold")

_spL, _bob = {}, {}
for nm, y in _rows.items():
    _spL[nm], = ax_ph.plot(*_sp1d(_WX, 0.0, y), color=COLORS[nm], lw=1.5)
    _bob[nm], = ax_ph.plot([], [], 'o', color=COLORS[nm], ms=14, zorder=5)
_ttx = ax_ph.text(0.98, 0.02, "", transform=ax_ph.transAxes,
                  ha="right", va="bottom", fontsize=9, color="gray")

_all_x = np.concatenate([_r["true"], _r["pinn"], _r["nn"]])
_ylo, _yhi = float(_all_x.min()) - 0.01, float(_all_x.max()) + 0.01
ax_tr.set_xlim(t_eval_np[0], t_eval_np[-1]); ax_tr.set_ylim(_ylo, _yhi)
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("x(t)"); ax_tr.grid(True, alpha=0.3)
ax_tr.axvline(split, color="gray", ls=":", lw=1.2)
ax_tr.plot(t_eval_np, _r["true"], color=COLORS["True"], lw=1, ls="--", label="True")
ax_tr.plot(t_eval_np, _r["pinn"], color=COLORS["PINN"], lw=1.5, label="PINN")
ax_tr.plot(t_eval_np, _r["nn"],   color=COLORS["NN"],   lw=1.5, label="NN")
ax_tr.legend(fontsize=8)
_cur, = ax_tr.plot([], [], color="k", lw=1.2, zorder=10)


def _upd(i):
    for nm, y in _rows.items():
        xv = float(np.clip(_xd[nm][i], -0.6, 0.55))
        _spL[nm].set_data(*_sp1d(_WX, xv - _R, y))
        _bob[nm].set_data([xv], [y])
    _cur.set_data([_ta[i], _ta[i]], [_ylo, _yhi])
    _ttx.set_text(f"t={_ta[i]:.3f} s")


_anim = FuncAnimation(fig_a, _upd, frames=_nf, interval=40, blit=False)
_ap   = os.path.join(RESULTS_DIR,
                     f"pinn_multi_traj_n{N_TRAIN_TRAJ}_s{SEED}_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
