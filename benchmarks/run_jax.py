#!/usr/bin/env python3
"""Run the JAX bifurcation simulation and save results.

Must be run with: conda run -n ribbon-jax python benchmarks/run_jax.py
"""
import time
import pickle
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm


def run_jax(N=21, w_by_l=1.0/12.0, total_time=12.5, dt=0.001):
    L = 0.1
    w = w_by_l * L
    h = 1e-3

    geom = dm.Geometry(
        length=L, r0=h, axs=w * h, jxs=w * h**3 / 3,
        ixs1=w * h**3 / 12, ixs2=h * w**3 / 12,
    )
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)

    nodes = np.zeros((N, 3))
    nodes[:, 0] = np.linspace(0, L, N)
    rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)

    # Fix boundary nodes
    node_pos = nodes
    start_nodes = np.where(node_pos[:, 0] <= 0.01)[0]
    end_nodes = np.where(node_pos[:, 0] >= 0.09)[0]
    fixed = np.union1d(start_nodes, end_nodes)
    rod, q0 = dm.fix_nodes(rod, q0, fixed)

    dl = L / (N - 1)
    model = dm.Sano.from_geometry(jnp.float64(dl), geom, mat)

    sp = dm.SimParams(
        dt=dt, total_time=total_time, tol=0.001, ftol=0.0001,
        dtol=0.01, max_iter=100, log_step=1, static_sim=False,
    )

    stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
    stepper.adaptive_dt = True
    stepper.max_dq_threshold = 0.1
    stepper.dt_reduction_factor = 0.5
    stepper.min_dt = dt / 1e6
    stepper.max_dt = dt * 2.0
    stepper.max_dt_reductions = 40
    stepper.enable_elastic_energy_tracking()

    # BC callback: gravity ramp + compression + shear + reverse
    u0 = 0.01
    ramp_end = 0.05
    g_ramp_z = -1000.0
    g_after_z = 0.0  # Zero gravity after ramp
    # Extend gravity ramp slightly into compression (0.05→0.10) to maintain
    # the z-perturbation until axial load exceeds the Euler buckling threshold.
    # The reference achieves this naturally via a ~3% gradient approximation error
    # that acts as an implicit symmetry-breaking perturbation. With JAX's exact
    # autodiff gradients, the perturbation decays before buckling can develop,
    # so we sustain gravity briefly to seed the instability. The post-buckle
    # trajectory is independent of the exact ramp duration (validated for 0.10–1.3s).
    ramp_off = 0.10
    phases = [
        (0.05, 2.55, 0, 1),     # compression in x
        (2.55, 7.55, 1, 1),     # shear in y
        (7.55, 12.5, 1, -1),    # reverse shear
    ]

    # Pre-extract mass z-components for gravity force updates
    mass_z = mass[2::4]

    def before_step(rod, q, u, aux, t, dt_step):
        gz = g_ramp_z if t <= ramp_off else g_after_z
        F_ext = jnp.zeros_like(q)
        F_ext = F_ext.at[2::4].set(mass_z * gz)
        rod = rod.with_F_ext(F_ext)

        # Phase-based node displacement
        for (t_start, t_end, direction, sign) in phases:
            if t_start <= t < t_end:
                q, rod = dm.move_nodes(q, rod, start_nodes, u0 * dt_step * sign, direction)
                break

        return rod, q, u, aux

    stepper.before_step = before_step

    print(f"Running JAX: N={N}, W/L={w_by_l:.4f}, total_time={total_time}s")
    t0 = time.perf_counter()
    result = stepper.simulate()
    elapsed = time.perf_counter() - t0

    n_steps = len(result.qs)
    print(f"  Done: {elapsed:.1f}s, {n_steps} logged steps, {n_steps/elapsed:.1f} steps/s")

    save_path = f'/data/shivam/Ribbon/bench_results/jax_result_n{N}_wl{int(1/w_by_l)}.pkl'
    save_data = {
        'qs': np.stack(result.qs),
        'times': np.array(result.times),
        'elapsed': elapsed,
        'n_steps': n_steps,
        'fixed_nodes': fixed,
        'start_nodes': start_nodes,
        'elastic_energies': result.elastic_energies,
        'N': N,
        'w_by_l': w_by_l,
        'total_time': total_time,
        'dt': dt,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Saved: {save_path}")
    return save_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=21)
    parser.add_argument('--total-time', type=float, default=12.5)
    args = parser.parse_args()
    run_jax(N=args.nodes, total_time=args.total_time)
