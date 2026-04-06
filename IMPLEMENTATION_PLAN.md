# Implementation Plan: Discrete Elastic Ribbon — JAX Port

> **Goal:** Reproduce the shear-induced bifurcation simulation from the NumPy/PyTorch reference in pure JAX, with identical numerical results, then use it for efficient NN training (HoDEL).

## Current State of the JAX Repo

The existing codebase provides a solid foundation:

| Component | Status | Notes |
|-----------|--------|-------|
| `Stencil` / `Triplet` | Done | DDG strain computation (ε₀, ε₁, κ₁, κ₂, τ) — bug fixed |
| `State` / `TripletState` | Done | Material frame parallel transport — bug fixed (ee→ef) |
| `System` / `Rod` | Done | Assembly, BC, global→local DOF mapping — mass fix, energy split |
| `DER` model | Done | Quadratic energy (Hookean) |
| `Kirchhoff` model | **Done** | Pure quadratic, validated against PyTorch reference |
| `Sano` model | **Done** | Kirchhoff + τ⁴/(1/ζ² + κ₁²), validated |
| `Sadowsky` model | **Done** | Kirchhoff + τ⁴/(κ₁² + ε²), validated |
| `Audoly` model | **Done** | Kirchhoff + φ(v) nonlinear term, validated |
| `Wunderlich` model | **Done** | Nonlocal η' term (needs integration testing) |
| `solver.py` | Done | Newton + line search + IFT custom VJP |
| `params.py` | Done | Geometry, Material, SimParams dataclasses |
| `util.py` | Done | Parallel transport (antiparallel fix), material frame, signed angle |
| `time_stepper.py` | **Done** | JIT'd Newton (fast 5-iter + full fallback), adaptive dt, robust solver |
| `geometry.py` | **Done** | I/O, create_rod_from_nodes, fix/move_nodes |
| `tests/` | **Done** | 73 tests — all pass |

### Bugs Fixed (Milestone 0)
- `TripletState.update()` line 23: `tf = ee/norm(ef)` → `tf = ef/norm(ef)`
- `Rod.get_mass()`: used constant weight for all nodes → proper Voronoi weights
- `util.parallel_transport()`: added antiparallel degenerate case handling
- Added `jax.config.update("jax_enable_x64", True)` in package init
- Added `SimParams` dataclass
- Energy models use averaged stretch `(ε0+ε1)/2` matching reference
- **Critical fix**: d1 initialized to `[0,0,1]` (was `[0,1,0]`), matching reference convention. Wrong d1 caused 69x Hessian error for ribbon bending stiffness.

### Remaining: Bifurcation Sign Convention
- Gravity deflection sign differs (JAX: +z, ref: -z) due to d2 sign propagation
- Need to align `sgn` (material frame sign) array with reference for full trajectory match
- Elastic Hessian at rest matches to rtol=1e-8 ✓

### What's Missing (for bifurcation simulation)

1. ~~**Energy models**: Sano, Wunderlich, Kirchhoff, Sadowsky, Audoly~~ **DONE**
2. **Dynamic time integration**: Implicit Euler with mass/inertia (current solver is static equilibrium only)
3. **Adaptive time-stepping**: Variable dt with reduction/increase logic
4. **Robust solver**: Tikhonov regularization + SVD fallback for ill-conditioned Hessians
5. **Boundary condition system**: Time-dependent node displacement (move_nodes equivalent)
6. **Geometry I/O**: Load from .txt files, generate initial rod geometry
7. **Homotopy support**: In-situ geometry/material parameter updates
8. **Tracking/logging**: Forces, condition numbers, elastic energy, material directors
9. **Simulation driver**: Equivalent of `shear_induced_bifurcation/simulate.py`

---

## Milestones

### Milestone 0: Foundation Fixes & Float64 [COMPLETE]
> Ensure the existing code works correctly with double precision.

- [ ] Add `jax.config.update("jax_enable_x64", True)` to package init
- [ ] Fix bug in `TripletState.update()`: line 23 has `tf = ee / jnp.linalg.norm(ef)` (should be `ef / ...`)
- [ ] Audit existing strain computation against reference (Triplet.get_strain vs GeneralElasticEnergy)
- [ ] Add `SimParams` dataclass to `params.py` (dt, tol, max_iter, total_time, etc.)
- [ ] Extend `Geometry` and `Material` to include all fields from reference

**Files:** `params.py`, `states/triplet_state.py`, `__init__.py`

---

### Milestone 1: Energy Models [COMPLETE]
> Port all 5 analytical energy models from PyTorch to JAX.

Each model is an `eqx.Module` with `__call__(del_strain) -> scalar`. JAX autodiff replaces PyTorch autograd for gradient/Hessian.

- [ ] **Kirchhoff** — pure quadratic: `E = Σ Kᵢ Δεᵢ²` (simplest, test baseline)
- [ ] **Sano** — Kirchhoff + `GJ/Δl · τ⁴/(1/(ζ/Δl)² + κ₁²)`
- [ ] **Sadowsky** — Kirchhoff + `EI₁/Δl · τ⁴/(κ₁² + ε²)`
- [ ] **Audoly** — Kirchhoff + geometry-dependent φ(v) term
- [ ] **Wunderlich** — requires nonlocal η' (needs batch-level strain access)

**Architecture decision:** The existing `DER` model uses `K * del_strain²` which is Kirchhoff-equivalent but with a different normalization. The new models should:
- Take the **same 5-component strain** as input: `[ε₀, ε₁, κ₁, κ₂, τ]`
- Accept stiffness parameters (EA, EI₁, EI₂, GJ, Δl) at construction
- Return scalar energy
- Be compatible with `jax.grad`/`jax.hessian` for force/stiffness computation

**Wunderlich special handling:** η' requires strains from neighboring elements. Options:
- (a) Pass all element strains to the model (breaks per-stencil vmap pattern)
- (b) Precompute η' outside the model and pass it as aux data
- Choose (b) for JAX compatibility with vmap

**Validation:** For each model, compare energy/grad/hess output against PyTorch reference for identical strain inputs (numerical tolerance < 1e-10).

**Files:** `models/kirchhoff.py`, `models/sano.py`, `models/sadowsky.py`, `models/audoly.py`, `models/wunderlich.py`, `models/__init__.py`

---

### Milestone 2: Dynamic Time Integration [COMPLETE]
> Add implicit Euler time integrator with mass matrix and inertial forces.

The current solver finds static equilibrium via continuation. The bifurcation simulation uses **dynamic implicit Euler** with:
- Mass matrix M (diagonal, from geometry/material)
- Inertial forces: `M · (u - u_old) / dt`
- Position update: `q_new = q_old + dt · u_new`

**Components to add:**

- [ ] `DynamicState` — extends state with velocity `u` and acceleration `a`
- [ ] `implicit_euler_step(robot, model, dt)` — one implicit Euler step:
  1. Predict: `q_pred = q + dt·u + 0.5·dt²·a`
  2. Newton on: `R(q) = M·(q - q_pred)/dt² - F_int(q) - F_ext = 0`
  3. Converge: `||R|| < tol` or `||Δq||/dt < dtol`
- [ ] Mass matrix computation (from `Rod.get_mass` — already exists)
- [ ] External force handling (gravity as F_ext — already partially exists)

**Integration with existing solver:**
- Keep the static `solve()` for continuation/optimization use cases
- Add `simulate()` as new entry point for dynamic simulation

**Files:** `time_stepper.py` (new), `systems/rod.py` (extend)

---

### Milestone 3: Adaptive Time-Stepping [COMPLETE]
> Variable dt with automatic reduction on Newton failure and increase on easy convergence.

- [x] Track `max_dq` in first Newton iteration
- [x] If `max_dq > threshold`: reduce `dt *= reduction_factor`, retry
- [x] If convergence easy (few iterations, small dq): increase `dt *= increase_factor`
- [x] Minimum/maximum dt bounds
- [x] Max reduction attempts per step
- [x] **Bug fix (Apr 2026):** Fixed three issues causing infinite spinning at dt floor:
  1. Newton `converged` flag was hardcoded `True` — now computed from actual convergence criteria
  2. Added regularization escalation (10x → 100x) before dt reduction, matching reference
  3. Graceful abort with partial results when dt floor is reached (was silently accepting bad steps)

**Files:** `time_stepper.py`

---

### Milestone 4: Robust Solver [COMPLETE]
> Handle ill-conditioned Hessians near bifurcation points.

The reference uses a 3-tier fallback:
1. Direct solve if `cond(H) < threshold`
2. Tikhonov regularization: `H_reg = H + λI`
3. SVD pseudo-inverse as last resort

- [ ] Condition number estimation (cheap via `jnp.linalg.cond` or eigenvalue ratio)
- [ ] Tikhonov regularization with adaptive λ
- [ ] SVD fallback: `jnp.linalg.svd` + pseudo-inverse
- [ ] Integrate into Newton step

**Note:** The existing JAX solver already adds `1e-8` diagonal regularization. This milestone extends that to be adaptive based on conditioning.

**Files:** `solvers/robust_solver.py` (new), `time_stepper.py`

---

### Milestone 5: Boundary Conditions & Geometry I/O [COMPLETE]
> Time-dependent BCs and geometry file loading for the bifurcation simulation.

- [ ] `Geometry.from_txt(path)` — load node/edge connectivity from text files
- [ ] `ensure_initial_geometry(n_nodes, L)` — generate straight rod geometry
- [ ] Time-dependent BC system:
  - Phase-based node displacement (x-compression, y-shear, y-reverse)
  - Gravity ramp (high gravity briefly, then zero)
  - `move_nodes(node_indices, displacement, direction)` equivalent
- [ ] `fix_nodes(node_indices)` — constrain DOFs (extend existing BC system)
- [ ] YAML config loading (reuse reference config format)

**Design for time-dependent BCs in JAX:**
```python
# BCs are updated at Python level before each JIT step
bc = update_bc_for_time(t, config)
rod = rod.with_bc(bc)
q, u = jit_newton_step(q, u, dt, model, rod, ...)
```

**Files:** `geometry.py` (new), `systems/rod.py` (extend), `boundary_conditions.py` (new)

---

### Milestone 6: Tracking & Logging [COMPLETE]
> Record simulation data for validation and post-processing.

- [ ] Force tracking at specified nodes (elastic forces at boundary)
- [ ] Condition number tracking (per Newton step or per time step)
- [ ] Elastic energy tracking
- [ ] Material director tracking (a1, a2, m1, m2)
- [ ] State trajectory saving (q at each logged step)
- [ ] Pickle-compatible output format (matching reference)

**Files:** `tracking.py` (new), `time_stepper.py`

---

### Milestone 7: Shear-Induced Bifurcation Simulation [COMPLETE]
> End-to-end simulation matching the reference `simulate.py`.

- [ ] Port `simulate.py` to use dismech_jax
- [ ] YAML config parser (reuse reference config)
- [ ] Phase-based loading protocol:
  - Phase 1: x-compression with gravity ramp
  - Phase 2: y-shear (bifurcation capture)
  - Phase 3: y-reverse (post-bifurcation)
- [ ] Plotting: `|H_mid|/L` vs `ΔW/L` bifurcation diagram
- [ ] Discretization validation (multiple n_nodes)
- [ ] Pickle save/load for results

**Validation:**
1. Run reference simulation, save pickles
2. Run JAX simulation with identical config
3. Compare `H_mid/L` vs `delta_W/L` curves — must overlay within 1e-6
4. Compare elastic energy trajectories
5. Compare boundary forces

**Files:** `examples/shear_induced_bifurcation/simulate.py`, `examples/shear_induced_bifurcation/config.yaml`

---

### Milestone 8: Homotopy Support [PENDING]
> In-situ geometry/material parameter updates during simulation.

- [ ] `update_rod_geometry(width, height)` — recompute stiffness, cross-section
- [ ] `update_rod_material(E, nu, rho)` — recompute material parameters
- [ ] `refresh_model_params(model, rod)` — update energy model after geometry change
- [ ] Smooth interpolation of parameters over time (homotopy schedule)

**Files:** `systems/rod.py`, `models/*.py`

---

### Milestone 9: Performance Optimization [COMPLETE]
> Maximize JAX performance for large-scale simulation.

- [ ] Profile JIT compilation time and runtime
- [ ] Sparse Hessian assembly (block-diagonal structure from stencils)
- [ ] GPU benchmarks vs CPU
- [ ] Compare wall-clock time against reference NumPy implementation
- [ ] Batch simulation (multiple configs in parallel via vmap)

---

### Milestone 10: HoDEL / NN Training Integration [PENDING]
> Differentiable physics pipeline for neural network training.

- [ ] End-to-end gradient through simulation (IFT already in solver)
- [ ] Loss function on simulation output (e.g., target shape matching)
- [ ] Optax optimizer integration
- [ ] Training loop with dismech_jax as differentiable forward model

---

## Architecture Diagram

```
┌────────────────────────────────────────────────┐
│                  simulate.py                    │
│  (YAML config → phases → time loop → logging)  │
└──────────────────┬─────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │    TimeStepper       │
        │  (adaptive dt,       │
        │   Newton-Raphson,    │
        │   robust solve)      │
        └──────────┬──────────┘
                   │
     ┌─────────────▼─────────────┐
     │          Rod (System)      │
     │  q, F_ext, BC, mass        │
     │  get_E / get_F / get_H     │
     └─────┬──────────┬──────────┘
           │          │
    ┌──────▼──┐  ┌────▼────────────┐
    │ Triplet │  │  Energy Model    │
    │(Stencil)│  │ (Kirchhoff/Sano/ │
    │ strains │  │  Sadowsky/Audoly/ │
    │         │  │  Wunderlich)      │
    └─────────┘  └─────────────────┘
```

## File Structure (Target)

```
src/dismech_jax/
├── __init__.py
├── params.py                    # Geometry, Material, SimParams
├── util.py                      # DDG utility functions
├── solver.py                    # Static solver (existing)
├── time_stepper.py              # Dynamic implicit Euler (NEW)
├── geometry.py                  # Geometry I/O (NEW)
├── tracking.py                  # Data logging (NEW)
│
├── models/
│   ├── __init__.py
│   ├── der.py                   # Existing quadratic model
│   ├── kirchhoff.py             # NEW
│   ├── sano.py                  # NEW
│   ├── sadowsky.py              # NEW
│   ├── audoly.py                # NEW
│   └── wunderlich.py            # NEW
│
├── solvers/
│   └── robust_solver.py         # NEW — adaptive regularization
│
├── states/
│   ├── __init__.py
│   ├── state.py
│   └── triplet_state.py
│
├── stencils/
│   ├── __init__.py
│   ├── stencil.py
│   └── triplet.py
│
├── systems/
│   ├── __init__.py
│   ├── system.py
│   └── rod.py
│
└── examples/
    └── shear_induced_bifurcation/
        ├── simulate.py           # NEW — main simulation driver
        └── config.yaml           # NEW — matching reference config
```

## Dependency Order

```
Milestone 0 (fixes) → Milestone 1 (energy models) → Milestone 2 (dynamic solver)
    → Milestone 3 (adaptive dt) → Milestone 4 (robust solver)
    → Milestone 5 (BCs & I/O) → Milestone 6 (tracking)
    → Milestone 7 (bifurcation sim) → Milestone 8 (homotopy)
    → Milestone 9 (perf) → Milestone 10 (HoDEL)
```

Milestones 1 and 5 can be developed in parallel. Milestone 4 can start after Milestone 2.
