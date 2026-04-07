### Dismech-JAX

JAX/Equinox port of the [Discrete Elastic Ribbon](https://github.com/shivamkumarpanda/discrete-elastic-ribbon) simulator for efficient differentiable physics simulation of elastic ribbons undergoing shear-induced buckling.

#### Setup

```bash
conda activate ribbon-jax
cd discrete-elastic-ribbon-jax
pip install -e .
```

**Dependencies:** jax, jaxlib, equinox, numpy, scipy, matplotlib

#### How to Run

**Shear-induced bifurcation (single run):**

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import dismech_jax as dm
from dismech_jax.models.kirchhoff import Kirchhoff

# Setup ribbon: N=45 nodes, L=0.1m, W/L=1/20
N, L, w_by_l, h = 45, 0.1, 1/20, 1e-3
w = w_by_l * L

geom = dm.Geometry(length=L, r0=h, axs=w*h, jxs=w*h**3/3,
                   ixs1=w*h**3/12, ixs2=h*w**3/12)
mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)

nodes = np.zeros((N, 3))
nodes[:, 0] = np.linspace(0, L, N)
rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)

# Fix both ends
start_nodes = np.where(nodes[:, 0] <= 0.01)[0]
end_nodes = np.where(nodes[:, 0] >= 0.09)[0]
rod, q0 = dm.fix_nodes(rod, q0, np.union1d(start_nodes, end_nodes))

# Energy model
dl = L / (N - 1)
model = Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

# Simulate
sp = dm.SimParams(dt=0.032, total_time=12.5, tol=0.001, ftol=0.0001,
                  dtol=0.01, max_iter=100, log_step=10, static_sim=False)
stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
stepper.adaptive_dt = True

# Define shear loading phases via before_step callback
# (see examples/shear_induced_bifurcation/simulate.py for full version)

result = stepper.simulate()
```

**Full benchmark sweep (all W/L ratios and energy models):**

```bash
# N=45, all models, W/L = [1/40, 1/20, 1/12, 1/6]
python benchmarks/run_wl_sweep.py --nodes 45 --wl 0.025 0.05 0.0833 0.1667

# N=63
python benchmarks/run_wl_sweep.py --nodes 63 --wl 0.025 0.05 0.0833 0.1667
```

**Run tests:**

```bash
cd discrete-elastic-ribbon-jax
pytest tests/ -v
```

#### Accomplishments

- Complete JAX/Equinox reimplementation of the Discrete Elastic Rod (DER) framework
- Five energy models: **Kirchhoff**, **Sano**, **Audoly**, **Sadowsky**, **Wunderlich**
- Implicit Euler time integration with Newton-Raphson solver
- Adaptive time-stepping with dt reduction/recovery
- Robust linear solver (Tikhonov regularization + SVD fallback)
- Analytical strain Jacobian + autodiff energy Hessian (hybrid approach)
- Block-diagonal Hessian assembly via `vmap` over per-triplet stencils (20x speedup over naive full Hessian)
- Shear-induced bifurcation simulation reproducing reference results across all W/L ratios
- Per-step metrics tracking: dt history, Newton iteration counts

#### Benchmark Results

Shear-induced bifurcation simulation on CPU. Backward shear phase only (t = 7.55 to 12.5s, sim duration = 4.95s), where the ribbon undergoes buckling and adaptive time-stepping is most active.

**N = 45 nodes (176 DOFs)**

| W/L  | Model     | x sim-time | Wall (s) | Steps | NR iters | NR/step |
|------|-----------|------------|----------|-------|----------|---------|
| 1/40 | Kirchhoff | 0.68x      | 3.4      | 283   | 1409     | 5.0     |
| 1/40 | Sano      | 0.67x      | 3.3      | 283   | 1410     | 5.0     |
| 1/40 | Audoly    | 0.45x      | 2.2      | 167   | 841      | 5.0     |
| 1/20 | Kirchhoff | 1.96x      | 9.7      | 778   | 4287     | 5.5     |
| 1/20 | Sano      | 1.80x      | 8.9      | 753   | 4136     | 5.5     |
| 1/20 | Audoly    | 2.75x      | 13.6     | 616   | 3389     | 5.5     |
| 1/12 | Kirchhoff | 2.04x      | 10.1     | 492   | 3009     | 6.1     |
| 1/12 | Sano      | 1.75x      | 8.7      | 502   | 2965     | 5.9     |
| 1/12 | Audoly    | 5.70x      | 28.2     | 1359  | 9047     | 6.7     |
| 1/6  | Kirchhoff | 9.07x      | 44.9     | 1418  | 8977     | 6.3     |
| 1/6  | Sano      | 8.02x      | 39.7     | 1440  | 9236     | 6.4     |
| 1/6  | Audoly    | 7.36x      | 36.4     | 1027  | 6830     | 6.7     |

**N = 63 nodes (248 DOFs)**

| W/L  | Model     | x sim-time | Wall (s) | Steps | NR iters | NR/step |
|------|-----------|------------|----------|-------|----------|---------|
| 1/40 | Kirchhoff | 1.39x      | 6.9      | 321   | 1798     | 5.6     |
| 1/40 | Sano      | 1.31x      | 6.5      | 286   | 1648     | 5.8     |
| 1/40 | Audoly    | 1.29x      | 6.4      | 257   | 1483     | 5.8     |
| 1/20 | Kirchhoff | 3.14x      | 15.6     | 735   | 4467     | 6.1     |
| 1/20 | Sano      | 2.90x      | 14.3     | 645   | 4102     | 6.4     |
| 1/20 | Audoly    | 6.81x      | 33.7     | 703   | 4407     | 6.3     |
| 1/12 | Kirchhoff | 164x       | 812.4    | 1227  | 7952     | 6.5     |
| 1/12 | Sano      | 13.4x      | 66.1     | 1498  | 9468     | 6.3     |
| 1/12 | Audoly    | 17.3x      | 85.4     | 2353  | 15092    | 6.4     |
| 1/6  | Kirchhoff | 48.2x      | 238.6    | 2524  | 18593    | 7.4     |
| 1/6  | Sano      | 54.8x      | 271.4    | 2500  | 17368    | 6.9     |
| 1/6  | Audoly    | 54.6x      | 270.3    | 2465  | 16167    | 6.6     |

*"x sim-time"* = wall-clock / simulated duration. Values < 1x mean faster than real-time.

#### Architecture

```
TimeStepper → Rod (System) → Triplet (Stencil) + Energy Model
                           ↓
                jax.grad/hessian for F, H
```

All components are `eqx.Module` (pytree-compatible), enabling end-to-end differentiation through the implicit solver via the Implicit Function Theorem.
