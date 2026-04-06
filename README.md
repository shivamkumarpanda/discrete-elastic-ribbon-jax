### Dismech-JAX

JAX/Equinox port of the Discrete Elastic Ribbon simulator for efficient differentiable physics simulation.

#### Goals

1. **Reproduce** the shear-induced bifurcation results from the NumPy/PyTorch reference
2. **Maximize efficiency** via JIT compilation, vmap, and GPU acceleration
3. **Enable differentiable physics** for HoDEL and neural network training

#### Setup

```bash
conda activate ribbon-jax
cd discrete-elastic-ribbon-jax
pip install -e .
```

#### Status

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed progress.

| Milestone | Description | Status |
|-----------|-------------|--------|
| 0 | Foundation fixes & float64 | **Complete** |
| 1 | Energy models (Kirchhoff, Sano, Sadowsky, Audoly, Wunderlich) | **Complete** (58 tests pass) |
| 2 | Dynamic time integration (implicit Euler) | **Complete** |
| 3 | Adaptive time-stepping | **Complete** |
| 4 | Robust solver (regularization + SVD) | **Complete** |
| 5 | Boundary conditions & geometry I/O | **Complete** |
| 6 | Tracking & logging | **Complete** |
| 7 | Shear-induced bifurcation simulation | **Complete** (runs end-to-end) |
| 8 | Homotopy support | Pending |
| 9 | Performance optimization | **Complete** (20x Newton speedup) |
| 10 | HoDEL / NN training integration | Pending |

#### Architecture

```
simulate.py → TimeStepper → Rod (System) → Triplet (Stencil) + Energy Model
                                         ↓
                              jax.grad/hessian for F, H
```

All components are `eqx.Module` (pytree-compatible), enabling end-to-end differentiation through the implicit solver via the Implicit Function Theorem.

#### Performance (CPU, fully optimized)

**Without callbacks (pure physics, no BC updates):**

| N (nodes) | DOFs | ms/step | Steps/sec |
|-----------|------|---------|-----------|
| 11 | 43 | 3.0 | 336 |
| 21 | 83 | 3.8 | 262 |
| 41 | 163 | 6.7 | 150 |
| 63 | 251 | 10.6 | 94 |
| 105 | 419 | 25.6 | 39 |

**With callbacks (BC updates via move_nodes each step):**

| N (nodes) | DOFs | ms/step | Steps/sec |
|-----------|------|---------|-----------|
| 11 | 43 | 3.0 | 330 |
| 21 | 83 | 12.4 | 80 |
| 41 | 163 | 35.6 | 28 |

Key optimizations:
- **Block-diagonal Hessian**: per-triplet `vmap(jax.hessian)` on 11-DOF stencils, scatter to global (20x vs naive)
- **JIT'd Newton loop**: `lax.scan` with fast path (5 iters) + full fallback
- **Cached pytrees**: triplets/model cached to prevent JIT recompilation from BC changes
- **~1.4s** one-time JIT compilation, then consistent per-step times
