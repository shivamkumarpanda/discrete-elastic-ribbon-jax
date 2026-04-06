#!/usr/bin/env python3
"""Profile the JAX simulation pipeline — find bottlenecks.

Measures: JIT compile time, per-step time, Hessian time, solve time, etc.
"""
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm


def setup_rod(N=21):
    """Create standard ribbon rod for benchmarking."""
    L = 0.1
    w_by_l = 1.0 / 12.0
    w = w_by_l * L
    h = 1e-3
    geom = dm.Geometry(
        length=L, r0=h,
        axs=w * h, jxs=w * h**3 / 3.0,
        ixs1=w * h**3 / 12.0, ixs2=h * w**3 / 12.0,
    )
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)

    nodes = np.zeros((N, 3))
    nodes[:, 0] = np.linspace(0, L, N)
    rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)

    fixed = np.array([0, N - 1])
    rod, q0 = dm.fix_nodes(rod, q0, fixed)

    dl = L / (N - 1)
    model = dm.Sano.from_geometry(jnp.float64(dl), geom, mat)
    return rod, model, q0, aux, mass, geom, mat


def time_fn(fn, name, warmup=1, repeats=5):
    """Time a function with warmup."""
    for _ in range(warmup):
        fn()
    # Force JAX to finish
    jax.block_until_ready(fn())

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  {name}: {avg*1000:.2f} ± {std*1000:.2f} ms")
    return avg


def bench_components(N=21):
    """Benchmark individual components."""
    print(f"\n{'='*60}")
    print(f"COMPONENT BENCHMARKS — N={N} nodes ({4*N-1} DOFs)")
    print(f"{'='*60}")

    rod, model, q0, aux, mass, geom, mat = setup_rod(N)
    n_dof = q0.shape[0]
    print(f"  DOFs: {n_dof}, Triplets: {N-2}")

    # Perturb slightly so strains are nonzero
    key = jax.random.PRNGKey(42)
    q = q0 + jax.random.normal(key, q0.shape) * 1e-5

    # 1. Internal energy computation
    E_fn = lambda: rod._internal_energy(q, model, aux)
    time_fn(E_fn, "Energy (scalar)")

    # 2. Gradient (force)
    grad_fn = lambda: jax.grad(rod._internal_energy)(q, model, aux)
    time_fn(grad_fn, "Gradient (force)")

    # 3. Full Hessian
    hess_fn = lambda: jax.hessian(rod._internal_energy)(q, model, aux)
    time_fn(hess_fn, "Hessian (full dense)")

    # 4. Linear solve (representative size)
    H = jax.hessian(rod._internal_energy)(q, model, aux)
    R = jax.grad(rod._internal_energy)(q, model, aux)
    H_reg = H + 1e-8 * jnp.eye(n_dof)
    solve_fn = lambda: jnp.linalg.solve(H_reg, R)
    time_fn(solve_fn, "Linear solve")

    # 5. Aux state update (parallel transport)
    batch_q = dm.Rod.global_q_to_batch_q(q)
    aux_fn = lambda: jax.vmap(lambda a, lq: a.update(lq))(aux, batch_q)
    time_fn(aux_fn, "Aux state update")


def bench_newton_step(N=21, n_steps=5):
    """Benchmark full Newton steps."""
    print(f"\n{'='*60}")
    print(f"NEWTON STEP BENCHMARKS — N={N} nodes, {n_steps} steps")
    print(f"{'='*60}")

    rod, model, q0, aux, mass, geom, mat = setup_rod(N)

    sp = dm.SimParams(dt=0.001, total_time=n_steps * 0.001, tol=1e-6, ftol=1e-4,
                      dtol=1e-4, max_iter=50, log_step=1, static_sim=False)

    stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)

    # Warmup: one step
    print("  Warming up (first step includes JIT compilation)...")
    t0 = time.perf_counter()
    result = stepper._newton_step(sp.dt)
    t_first = time.perf_counter() - t0
    print(f"  First step (with JIT): {t_first*1000:.1f} ms, {result.iterations} iters")

    stepper.q = result.q
    stepper.u = result.u
    stepper.aux = result.aux

    # Subsequent steps (JIT cached)
    times = []
    iters = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        result = stepper._newton_step(sp.dt)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        iters.append(result.iterations)
        stepper.q = result.q
        stepper.u = result.u
        stepper.aux = result.aux

    avg = np.mean(times)
    avg_iters = np.mean(iters)
    print(f"  Subsequent steps: {avg*1000:.1f} ± {np.std(times)*1000:.1f} ms, avg {avg_iters:.1f} iters")
    print(f"  Per-iteration: {avg/avg_iters*1000:.2f} ms")


def bench_full_simulation(N=21, total_time=0.1):
    """Benchmark full simulation wall time."""
    print(f"\n{'='*60}")
    print(f"FULL SIMULATION — N={N}, total_time={total_time}s")
    print(f"{'='*60}")

    rod, model, q0, aux, mass, geom, mat = setup_rod(N)

    sp = dm.SimParams(dt=0.001, total_time=total_time, tol=1e-6, ftol=1e-4,
                      dtol=1e-4, max_iter=100, log_step=10, static_sim=False)

    stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
    stepper.adaptive_dt = True
    stepper.max_dq_threshold = 0.1

    t0 = time.perf_counter()
    result = stepper.simulate()
    elapsed = time.perf_counter() - t0

    n_logged = len(result.qs)
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Logged steps: {n_logged}")
    print(f"  Effective steps/sec: {n_logged / elapsed:.1f}")
    return elapsed


if __name__ == "__main__":
    for N in [11, 21, 41]:
        bench_components(N)
        bench_newton_step(N)

    print("\n" + "="*60)
    print("FULL SIMULATIONS")
    print("="*60)
    for N in [11, 21]:
        bench_full_simulation(N, total_time=0.05)
