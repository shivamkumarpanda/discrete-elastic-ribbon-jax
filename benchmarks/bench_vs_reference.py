#!/usr/bin/env python3
"""Benchmark JAX vs reference NumPy implementation on identical configs.

Runs a short bifurcation-like simulation (compression phase only) and compares wall time.
"""
import time
import numpy as np

# ─── JAX version ──────────────────────────────────────────────────────────────

def run_jax(N=21, total_time=0.5, dt=0.001):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import dismech_jax as dm

    L = 0.1; w = L / 12; h = 1e-3
    geom = dm.Geometry(length=L, r0=h, axs=w*h, jxs=w*h**3/3,
                       ixs1=w*h**3/12, ixs2=h*w**3/12)
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)

    nodes = np.zeros((N, 3)); nodes[:, 0] = np.linspace(0, L, N)
    rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)
    rod, q0 = dm.fix_nodes(rod, q0, np.array([0, N - 1]))

    dl = L / (N - 1)
    model = dm.Sano.from_geometry(jnp.float64(dl), geom, mat)

    sp = dm.SimParams(dt=dt, total_time=total_time, tol=1e-3, ftol=1e-4,
                      dtol=0.01, max_iter=100, log_step=10, static_sim=False)

    stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
    stepper.adaptive_dt = True
    stepper.max_dq_threshold = 0.1
    stepper.min_dt = dt / 1e6
    stepper.max_dt = dt * 2.0

    # Compress: move node 0 in x
    start_nodes = np.array([0])
    u0 = 0.01

    def before_step(rod, q, u, aux, t, dt_step):
        if t > dt_step + 1e-14:
            disp = u0 * dt_step
            q, rod = dm.move_nodes(q, rod, start_nodes, disp, 0)
        return rod, q, u, aux

    stepper.before_step = before_step

    # Warmup JIT
    print(f"  JAX N={N}: warming up JIT...")
    t0 = time.perf_counter()
    stepper._newton_step(dt)
    jit_time = time.perf_counter() - t0
    print(f"  JIT warmup: {jit_time:.2f}s")

    # Reset
    stepper.q = q0.copy()
    stepper.u = jnp.zeros_like(q0)

    print(f"  JAX N={N}: running simulation ({total_time}s, dt={dt})...")
    t0 = time.perf_counter()
    result = stepper.simulate()
    elapsed = time.perf_counter() - t0
    n_steps = len(result.qs)
    print(f"  JAX: {elapsed:.3f}s wall, {n_steps} logged steps, "
          f"{n_steps/elapsed:.1f} steps/s (excl JIT)")

    return elapsed, n_steps, result


if __name__ == "__main__":
    print("=" * 60)
    print("JAX vs Reference Benchmark")
    print("=" * 60)

    for N in [11, 21, 41]:
        for total_time in [0.5, 1.0]:
            print(f"\n--- N={N}, total_time={total_time}s ---")
            try:
                jax_time, jax_steps, _ = run_jax(N, total_time)
                print(f"  Summary: JAX took {jax_time:.2f}s for {jax_steps} logged steps")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback; traceback.print_exc()
