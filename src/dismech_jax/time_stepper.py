"""Dynamic implicit Euler time stepper with adaptive dt and robust solving.

Implements the Newton-Raphson solver for the implicit Euler system:
    R(q) = -dE/dq + (M/dt²)(q - q_n) - (M/dt)u_n + F_ext = 0

Two modes:
  1. simulate() — Python outer loop with callbacks, adaptive dt, logging
  2. simulate_jit() — fully JIT'd with pre-computed BC schedule (fastest)
"""
import dataclasses
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from .systems.rod import Rod, BC
from .states import TripletState
from .params import SimParams


# ─── Scatter index precomputation ─────────────────────────────────────────────

def _build_scatter_indices(n_dof, N_triplets):
    """Precompute scatter indices for local 11×11 blocks → global matrix."""
    starts = jnp.arange(N_triplets) * 4
    offsets = jnp.arange(11)
    local_to_global = starts[:, None] + offsets[None, :]
    return local_to_global


def _build_hessian_scatter(local_to_global, N_triplets):
    """Precompute flattened row/col indices for Hessian scatter."""
    row_idx = jnp.repeat(local_to_global, 11, axis=1).reshape(-1)
    col_idx = jnp.tile(local_to_global, (1, 11)).reshape(N_triplets, 11, 11).reshape(-1)
    return row_idx, col_idx


# ─── Fully JIT'd single Newton step ──────────────────────────────────────────

def _robust_solve(H, R, reg, cond_threshold, reg_factor, max_reg_factor):
    """JAX-compatible robust linear solve matching reference RobustSolver.

    Strategy:
      1. Check condition number of H
      2. If ill-conditioned: adaptive Tikhonov regularization
      3. Direct solve
      4. If NaN: fallback to SVD pseudo-inverse
    """
    n = H.shape[0]

    # Condition number check + adaptive regularization
    cond = jnp.linalg.cond(H)
    needs_extra_reg = (cond > cond_threshold) | ~jnp.isfinite(cond)
    lambda_reg = jnp.where(
        needs_extra_reg,
        jnp.where(
            jnp.isfinite(cond),
            jnp.minimum(reg_factor * jnp.sqrt(cond), max_reg_factor),
            max_reg_factor,
        ),
        reg,
    )
    H_reg = H + lambda_reg * jnp.eye(n)

    # Direct solve
    dq = jnp.linalg.solve(H_reg, R)

    # SVD fallback if direct solve produced NaN/Inf
    def svd_fallback(_):
        # Try max regularization first
        H_max = H + max_reg_factor * jnp.eye(n)
        dq2 = jnp.linalg.solve(H_max, R)
        ok2 = jnp.all(jnp.isfinite(dq2))

        # SVD pseudo-inverse of REGULARIZED H as last resort
        # Using regularized H avoids amplifying indefinite (negative) eigenvalues
        U, s, Vt = jnp.linalg.svd(H_max, full_matrices=False)
        s_tol = jnp.max(jnp.abs(s)) * jnp.finfo(s.dtype).eps * n
        s_inv = jnp.where(jnp.abs(s) > s_tol, 1.0 / s, 0.0)
        dq_pinv = Vt.T @ (s_inv * (U.T @ R))

        return jnp.where(ok2, dq2, dq_pinv)

    solve_ok = jnp.all(jnp.isfinite(dq))
    return jax.lax.cond(solve_ok, lambda _: dq, svd_fallback, None)


def _make_jit_step(triplets, model, max_iter, n_dof, N_triplets, tol, ftol, dtol):
    """Build a fully JIT-compiled function for one time step.

    Captures triplets and model as static pytree structure.
    Returns a function: (q, u, aux, F_ext, free_dof, mass, dt, reg) -> StepResult
    """
    local_to_global = _build_scatter_indices(n_dof, N_triplets)
    hess_row, hess_col = _build_hessian_scatter(local_to_global, N_triplets)

    # Robust solver parameters (matching reference RobustSolver defaults)
    cond_threshold = 1e12
    reg_factor = 1e-8
    max_reg_factor_base = 1e-4

    @eqx.filter_jit
    def jit_step(q_n, u_n, aux, F_ext, free_dof, mass, dt, reg, static_sim, eta):
        """One full implicit Euler step with Newton-Raphson.

        Returns: (q_new, u_new, aux_new, converged, n_iters, max_dq, initial_max_dq)
        """
        M_dt2 = mass / dt**2
        M_dt = mass / dt
        # Viscous damping: F_damp = -eta * u. The Jacobian contribution is -eta/dt.
        # Cap eta/dt to avoid blow-up at tiny dt during adaptive retries.
        eta_dt = jnp.minimum(eta / dt, eta / jnp.float64(0.001))
        n_free = free_dof.shape[0]
        # Scale max_reg_factor with caller's reg (allows escalation from outer loop)
        max_reg = jnp.maximum(max_reg_factor_base, reg)

        # ── Per-triplet energy, gradient, Hessian via chain rule ──
        # E→strain via autodiff, strain→q via autodiff with fixed bishop frame.
        # This matches the reference's approach and ensures correct Hessian.
        from .analytical_grad_hess import compute_local_energy_grad_hess

        def compute_local(t, q_loc, a):
            return compute_local_energy_grad_hess(t, q_loc, a, model, t.bar_strain)

        def compute_and_assemble(q, aux_inner):
            """Compute local E/g/H, assemble global grad+Hessian."""
            batch_q = _global_q_to_batch_q(q, N_triplets)
            _, local_g, local_H = jax.vmap(compute_local)(triplets, batch_q, aux_inner)

            # Scatter gradient
            F_g = jnp.zeros(n_dof).at[local_to_global.ravel()].add(local_g.ravel())
            # Scatter Hessian
            H_g = jnp.zeros((n_dof, n_dof)).at[hess_row, hess_col].add(local_H.reshape(-1))
            return F_g, H_g

        # ── Newton loop via lax.while_loop ──
        # while_loop traces body ONCE (O(1) compile time vs O(max_iter) for scan)
        def newton_cond(carry):
            q, iteration, err0, err, max_dq, init_mdq, damping, running_max_dq = carry
            not_max_iter = iteration < max_iter
            disp_not_conv = max_dq / dt >= dtol
            force_not_conv = err >= tol
            rel_not_conv = (err >= err0 * ftol) | (err0 <= 0)
            not_converged = disp_not_conv & force_not_conv & rel_not_conv
            return not_max_iter & not_converged

        def newton_body(carry):
            q, iteration, err0, _err_prev, _max_dq_prev, init_mdq, damping, running_max_dq = carry

            # Update material frame for current iterate (matching reference:
            # re-computes a1,a2,ref_twist at each Newton iteration via
            # compute_time_parallel).
            batch_q_iter = _global_q_to_batch_q(q, N_triplets)
            aux_iter = jax.vmap(lambda a, lq: a.update(lq))(aux, batch_q_iter)

            grad_E, hess_E = compute_and_assemble(q, aux_iter)
            # Matching reference sign convention exactly:
            #   forces  = +grad_E - gravity + inertia [+ damping]
            #   jacobian = +hess_E + diag(M/dt²) [+ diag(eta/dt)]
            #   dq = solve(jacobian, forces);  q -= dq
            #
            # Reference: forces -= F_elastic (=-grad_E) → +(grad_E)
            #            forces -= gravity_forces (=M*g)  → -(M*g)
            #            forces += inertial_force (=M/dt²(q-q_n) - M/dt*u_n)
            #            forces -= damping_force (=-eta*u) → +(eta*u)
            F_inertia = M_dt2 * (q - q_n) - M_dt * u_n
            R = jnp.where(static_sim,
                          grad_E - F_ext,
                          grad_E - F_ext + F_inertia + eta * u_n)
            H = jnp.where(static_sim,
                          hess_E,
                          hess_E + jnp.diag(M_dt2 + eta_dt))

            R_free = R[free_dof]
            H_free = H[jnp.ix_(free_dof, free_dof)]

            # Robust linear solve (condition check + SVD fallback)
            dq_free = _robust_solve(
                H_free, R_free, reg, cond_threshold, reg_factor, max_reg)

            # Guard: skip solve when residual is near-zero to avoid amplifying
            # near-zero Hessian eigenvalues (matches reference _min_force=1e-8)
            dq_free = jnp.where(
                jnp.linalg.norm(R_free) < 1e-8,
                jnp.zeros_like(dq_free),
                dq_free)

            # Constant damping factor after iteration 10 (matching reference:
            # _adaptive_damping returns max(alpha * 0.9, 0.1) with alpha=1.0 constant)
            d = jnp.where(iteration >= 10, jnp.float64(0.9), jnp.float64(1.0))
            dq_damped = dq_free * d
            q_new = q.at[free_dof].add(-dq_damped)

            step_max_dq = jnp.max(jnp.abs(dq_damped))
            err = jnp.linalg.norm(R_free)
            err0_new = jnp.where(iteration == 0, err, err0)
            init_mdq_new = jnp.where(iteration == 0, jnp.max(jnp.abs(dq_free)), init_mdq)
            # Track running maximum displacement across all iterations (for dt reduction check)
            new_running_max = jnp.maximum(running_max_dq, step_max_dq)

            return (q_new, iteration + 1, err0_new, err, step_max_dq, init_mdq_new, d, new_running_max)

        init = (q_n, jnp.int32(0), jnp.float64(0.0), jnp.float64(jnp.inf),
                jnp.float64(jnp.inf), jnp.float64(0.0), jnp.float64(1.0), jnp.float64(0.0))

        q_final, n_iters, err0_final, err_final, final_max_dq, init_max_dq, _, running_max_dq = jax.lax.while_loop(
            newton_cond, newton_body, init
        )
        # Check if we actually converged (any one criterion met) vs hit max_iter
        converged = (
            (final_max_dq / dt < dtol) |
            (err_final < tol) |
            ((err_final < err0_final * ftol) & (err0_final > 0))
        )

        u_new = (q_final - q_n) / dt

        # Update aux
        batch_q = _global_q_to_batch_q(q_final, N_triplets)
        new_aux = jax.vmap(lambda a, lq: a.update(lq))(aux, batch_q)

        return q_final, u_new, new_aux, converged, n_iters, running_max_dq, init_max_dq

    return jit_step


def _global_q_to_batch_q(q, N_triplets):
    """Extract overlapping 11-DOF windows for each triplet."""
    starts = jnp.arange(N_triplets) * 4
    return jax.vmap(lambda s: jax.lax.dynamic_slice(q, (s,), (11,)))(starts)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class StepResult:
    q: jax.Array
    u: jax.Array
    aux: TripletState
    converged: bool
    iterations: int
    max_dq: float
    initial_max_dq: float


@dataclasses.dataclass
class SimulationResult:
    qs: list
    us: list
    times: list
    elastic_energies: list
    condition_numbers: list
    forces: list
    force_times: list


# ─── TimeStepper ──────────────────────────────────────────────────────────────

class TimeStepper:
    """Implicit Euler time stepper with adaptive dt and robust solving.

    Two simulation modes:
      simulate()     — Python outer loop, supports callbacks, adaptive dt, logging
      simulate_jit() — fully JIT'd via lax.scan, pre-computed BCs (fastest)
    """

    def __init__(self, rod, model, sim_params, mass, q0, aux):
        self.rod = rod
        self.model = model
        self.sim_params = sim_params
        self.mass = mass
        self.q = q0.copy()
        self.u = jnp.zeros_like(q0)
        self.aux = aux

        n_dof = int(q0.shape[0])
        N_triplets = (n_dof + 1) // 4 - 2
        self._n_dof = n_dof
        self._N_triplets = N_triplets
        self._local_to_global = _build_scatter_indices(n_dof, N_triplets)

        self._update_free_dof()

        # Build JIT'd step function using while_loop (O(1) compile time)
        self._jit_step = _make_jit_step(
            rod.triplets, model,
            max_iter=sim_params.max_iter,
            n_dof=n_dof,
            N_triplets=N_triplets,
            tol=sim_params.tol,
            ftol=sim_params.ftol,
            dtol=sim_params.dtol,
        )

        # Adaptive dt
        self.adaptive_dt = False
        self.max_dq_threshold = 0.1
        self.dt_reduction_factor = 0.5
        self.dt_increase_factor = 1.2
        self.min_dt = sim_params.dt / 1e6
        self.max_dt = sim_params.dt * 2.0
        self.max_dt_reductions = 40

        # Robust solver
        self.base_reg = 1e-8
        self.current_reg = 1e-8

        # Viscous damping (F = -eta * u). Set to 0 for no damping.
        self.eta = 0.0

        self.before_step: Callable | None = None
        self.track_forces_nodes: np.ndarray | None = None
        self._track_condition_number = False
        self._track_elastic_energy = False
        self._track_material_directors = False

    def _update_free_dof(self):
        bc_idx = self.rod.bc.idx_b
        free_mask = jnp.ones(self._n_dof, dtype=bool)
        if bc_idx.shape[0] > 0:
            free_mask = free_mask.at[bc_idx].set(False)
        self._free_dof = jnp.where(free_mask)[0]

    def set_nodes_to_track_forces(self, node_indices: np.ndarray):
        self.track_forces_nodes = node_indices

    def enable_condition_number_tracking(self):
        self._track_condition_number = True

    def enable_elastic_energy_tracking(self):
        self._track_elastic_energy = True

    def enable_material_director_tracking(self):
        self._track_material_directors = True

    # ── Fully JIT'd simulation (fastest, no Python per-step overhead) ─────────

    def simulate_jit(
        self,
        n_steps: int,
        dt: float | None = None,
        bc_displacements: jax.Array | None = None,
        bc_node_dofs: jax.Array | None = None,
        F_ext_schedule: jax.Array | None = None,
        log_every: int = 1,
    ) -> SimulationResult:
        """Run simulation with pre-computed BCs via lax.scan (no Python overhead).

        Args:
            n_steps: number of time steps
            dt: time step (default: sim_params.dt)
            bc_displacements: (n_steps,) displacement per step for BC nodes
            bc_node_dofs: (n_bc_dofs,) DOF indices to apply displacement to
            F_ext_schedule: (n_steps, n_dof) or None — per-step external forces
            log_every: log state every N steps

        Returns:
            SimulationResult with qs logged every log_every steps
        """
        if dt is None:
            dt = self.sim_params.dt
        dt = jnp.float64(dt)
        reg = jnp.float64(self.current_reg)
        static_sim = jnp.bool_(self.sim_params.static_sim)

        free_dof = self._free_dof
        F_ext_base = self.rod.F_ext

        # Build per-step scan function
        @jax.jit
        def scan_body(carry, step_data):
            q, u, aux = carry
            bc_disp, F_ext_step = step_data

            # Apply BC displacement: q[bc_dofs] += disp
            q_bc = jnp.where(
                bc_node_dofs is not None,
                q.at[bc_node_dofs].add(bc_disp),
                q,
            ) if bc_node_dofs is not None else q

            F_ext = jnp.where(F_ext_step is not None, F_ext_step, F_ext_base) \
                if F_ext_schedule is not None else F_ext_base

            q_new, u_new, aux_new, conv, n_it, mdq, imdq = self._jit_step(
                q_bc, u, aux, F_ext, free_dof, self.mass, dt, reg, static_sim,
                jnp.float64(self.eta)
            )

            # Update BC positions in q_new
            q_new = jnp.where(
                bc_node_dofs is not None,
                q_new.at[bc_node_dofs].set(q_bc[bc_node_dofs]),
                q_new,
            ) if bc_node_dofs is not None else q_new

            return (q_new, u_new, aux_new), q_new

        # Prepare step data
        if bc_displacements is None:
            bc_displacements = jnp.zeros(n_steps)
        if F_ext_schedule is None:
            F_ext_per_step = jnp.broadcast_to(F_ext_base, (n_steps, self._n_dof))
        else:
            F_ext_per_step = F_ext_schedule

        step_data = (bc_displacements, F_ext_per_step)

        # Run scan
        (q_final, u_final, aux_final), all_qs = jax.lax.scan(
            scan_body, (self.q, self.u, self.aux), step_data
        )

        # Update state
        self.q = q_final
        self.u = u_final
        self.aux = aux_final

        # Build result (subsample for logging)
        log_indices = np.arange(0, n_steps, log_every)
        times = [(i + 1) * float(dt) for i in log_indices]

        result = SimulationResult(
            qs=[np.array(all_qs[i]) for i in log_indices],
            us=[], times=times,
            elastic_energies=[], condition_numbers=[],
            forces=[], force_times=[],
        )
        return result

    # ── Python outer loop simulation (supports callbacks, adaptive dt) ────────

    def simulate(self) -> SimulationResult:
        """Run simulation with Python outer loop. Supports callbacks and adaptive dt."""
        sp = self.sim_params
        dt = sp.dt
        current_time = 0.0
        total_time = sp.total_time

        result = SimulationResult(
            qs=[], us=[], times=[],
            elastic_energies=[], condition_numbers=[],
            forces=[], force_times=[],
        )

        result.qs.append(np.array(self.q))
        result.us.append(np.array(self.u))
        result.times.append(current_time)

        import sys as _sys
        _print_every = max(1, int(total_time / dt / 20))  # ~20 progress prints
        _step_count = 0
        _t_wall_start = __import__('time').perf_counter()

        log_counter = 0
        free_dof = self._free_dof
        # Pre-create JAX scalars to avoid per-step allocation
        _reg = jnp.float64(self.current_reg)
        _static_sim = jnp.bool_(sp.static_sim)
        _dt = jnp.float64(dt)
        _mass = self.mass
        _F_ext = self.rod.F_ext
        _step = self._jit_step
        _adaptive = self.adaptive_dt
        _threshold = self.max_dq_threshold
        _has_callback = self.before_step is not None
        _log_step = sp.log_step
        _eta = jnp.float64(self.eta)
        _track_E = self._track_elastic_energy
        _track_F = self.track_forces_nodes is not None

        while current_time < total_time - 1e-14:
            # Save state BEFORE applying BCs so retries can restore cleanly
            q_backup = self.q
            u_backup = self.u
            aux_backup = self.aux
            rod_backup = self.rod

            # Apply BCs BEFORE time advance (matching reference convention)
            if _has_callback:
                bc_before = self.rod.bc.idx_b
                self.rod, self.q, self.u, self.aux = self.before_step(
                    self.rod, self.q, self.u, self.aux, current_time, dt
                )
                # Update cached values if rod changed
                _F_ext = self.rod.F_ext
                if self.rod.bc.idx_b is not bc_before:
                    self._update_free_dof()
                    free_dof = self._free_dof

            # ── Per-step retry loop with regularization escalation ──
            dt_reductions_this_step = 0
            reg_level = 0  # 0=base, 1=10x, 2=100x
            step_successful = False

            while not step_successful and dt_reductions_this_step < self.max_dt_reductions:
                _reg = jnp.float64(self.current_reg)
                q_new, u_new, aux_new, conv, n_it, mdq, imdq = _step(
                    self.q, self.u, self.aux, _F_ext,
                    free_dof, _mass, _dt, _reg, _static_sim, _eta,
                )
                conv_val = bool(conv)
                imdq_val = float(imdq)
                mdq_val = float(mdq)  # max displacement across all Newton iters

                # Detect NaN/divergence in Newton output — treat as convergence failure
                has_nan = ((not np.isfinite(mdq_val)) or (not np.isfinite(imdq_val))
                           or bool(jnp.any(~jnp.isfinite(q_new)))
                           or mdq_val > 1e6)  # absurdly large step = diverged
                if has_nan:
                    conv_val = False
                    mdq_val = float('inf')
                    imdq_val = float('inf')

                def _rollback_and_reduce_dt():
                    """Restore pre-BC state, reduce dt, re-apply BCs with new dt."""
                    nonlocal dt, _dt, current_time, free_dof, _F_ext
                    dt = max(self.min_dt, dt * self.dt_reduction_factor)
                    _dt = jnp.float64(dt)
                    self.current_reg = self.base_reg
                    # Restore pre-BC state (before move_nodes was applied)
                    self.q, self.u, self.aux = q_backup, u_backup, aux_backup
                    self.rod = rod_backup
                    # Re-apply BCs with the new smaller dt
                    if _has_callback:
                        bc_before = self.rod.bc.idx_b
                        self.rod, self.q, self.u, self.aux = self.before_step(
                            self.rod, self.q, self.u, self.aux, current_time, dt
                        )
                        _F_ext = self.rod.F_ext
                        if self.rod.bc.idx_b is not bc_before:
                            self._update_free_dof()
                            free_dof = self._free_dof

                def _rollback_reg_only():
                    """Restore post-BC state for regularization retry (same dt)."""
                    # BCs already applied with correct dt, just restore q/u/aux
                    self.q = q_after_bc
                    self.u = u_after_bc
                    self.aux = aux_after_bc

                # Save post-BC state for reg-only retries (same dt, same BCs)
                q_after_bc = self.q
                u_after_bc = self.u
                aux_after_bc = self.aux

                if not conv_val:
                    # Newton failed — try regularization escalation before dt reduction
                    if reg_level == 0:
                        self.current_reg = self.base_reg * 10.0
                        reg_level = 1
                        _rollback_reg_only()
                        dt_reductions_this_step += 1
                        continue
                    elif reg_level == 1:
                        self.current_reg = self.base_reg * 100.0
                        reg_level = 2
                        _rollback_reg_only()
                        dt_reductions_this_step += 1
                        continue
                    # reg_level == 2: regularization exhausted, fall through to dt reduction

                # Check displacement (for both converged and reg-exhausted cases)
                if _adaptive and mdq_val > _threshold:
                    new_dt = max(self.min_dt, dt * self.dt_reduction_factor)
                    if new_dt < dt * 0.99:
                        reg_level = 0
                        _rollback_and_reduce_dt()
                        dt_reductions_this_step += 1
                        continue
                    elif not has_nan:
                        # dt at floor — accept step anyway (matching reference)
                        # But NEVER accept NaN steps
                        step_successful = True
                        self.current_reg = self.base_reg
                        break
                    else:
                        # NaN at dt floor — can't accept, keep trying with more reg
                        dt_reductions_this_step += 1
                        continue

                if not conv_val and not step_successful:
                    new_dt = max(self.min_dt, dt * self.dt_reduction_factor)
                    if new_dt < dt * 0.99:
                        reg_level = 0
                        _rollback_and_reduce_dt()
                        dt_reductions_this_step += 1
                        continue
                    elif not has_nan:
                        step_successful = True
                        self.current_reg = self.base_reg
                        break
                    else:
                        dt_reductions_this_step += 1
                        continue

                # Step accepted
                step_successful = True
                self.current_reg = self.base_reg

            if not step_successful:
                # Exhausted max_dt_reductions retries
                print(f"\n{'='*60}")
                print(f"SIMULATION STOPPED at t={current_time:.4f}s / {total_time:.1f}s")
                print(f"  Exhausted {self.max_dt_reductions} retry attempts")
                print(f"  Returning {len(result.qs)} frames computed so far.")
                print(f"{'='*60}")
                self.current_reg = self.base_reg
                return result

            # Advance time by the dt that was actually used for this step
            dt_used = dt
            current_time += dt_used

            # dt increase for FUTURE steps (after time advance, so it doesn't
            # affect the current step's time accounting)
            if _adaptive:
                if imdq_val < 0.1 * _threshold:
                    new_dt = min(self.max_dt, dt * self.dt_increase_factor)
                    if new_dt > dt * 1.01:
                        dt = new_dt
                        _dt = jnp.float64(dt)

            self.q = q_new
            self.u = u_new
            self.aux = aux_new

            _step_count += 1
            if _step_count % _print_every == 0:
                _elapsed = __import__('time').perf_counter() - _t_wall_start
                _pct = current_time / total_time * 100
                print(f"  t={current_time:.3f}/{total_time:.1f}s ({_pct:.0f}%) | step {_step_count} | dt={dt:.2e} | wall={_elapsed:.1f}s", flush=True)

            log_counter += 1
            if log_counter >= _log_step:
                log_counter = 0
                result.qs.append(np.array(self.q))
                result.us.append(np.array(self.u))
                result.times.append(current_time)

                if _track_E:
                    E = float(self.rod._internal_energy(self.q, self.model, self.aux))
                    result.elastic_energies.append(E)

                if _track_F:
                    F_int = np.array(jax.grad(
                        lambda _q: self.rod._internal_energy(_q, self.model, self.aux)
                    )(self.q))
                    node_forces = {}
                    for n in self.track_forces_nodes:
                        dofs = [4 * n, 4 * n + 1, 4 * n + 2]
                        node_forces[int(n)] = F_int[dofs]
                    result.forces.append(node_forces)
                    result.force_times.append(current_time)

        return result
