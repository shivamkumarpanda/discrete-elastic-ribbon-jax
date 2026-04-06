#!/usr/bin/env python3
"""Run JAX bifurcation simulation for multiple node counts and compare."""
import time
import pickle
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm


def run_single(N, total_time=12.5, dt=0.001, eta=0.0):
    """Run one simulation and return results + metrics."""
    L = 0.1; w = L / 12; h = 1e-3

    geom = dm.Geometry(
        length=L, r0=h, axs=w * h, jxs=w * h**3 / 3,
        ixs1=w * h**3 / 12, ixs2=h * w**3 / 12,
    )
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)

    nodes = np.zeros((N, 3))
    nodes[:, 0] = np.linspace(0, L, N)
    rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)

    start_nodes = np.where(nodes[:, 0] <= 0.01)[0]
    end_nodes = np.where(nodes[:, 0] >= 0.09)[0]
    fixed = np.union1d(start_nodes, end_nodes)
    rod, q0 = dm.fix_nodes(rod, q0, fixed)

    dl = L / (N - 1)
    model = dm.Sano.from_geometry(jnp.float64(dl), geom, mat)

    sp = dm.SimParams(
        dt=dt, total_time=total_time, tol=0.001, ftol=0.0001,
        dtol=0.01, max_iter=100, log_step=10, static_sim=False,
    )

    stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
    stepper.adaptive_dt = True
    stepper.max_dq_threshold = 0.1
    stepper.dt_reduction_factor = 0.5
    stepper.min_dt = dt / 1e6
    stepper.max_dt = dt * 2.0
    stepper.max_dt_reductions = 60
    stepper.enable_elastic_energy_tracking()
    stepper.eta = eta

    mass_z = mass[2::4]
    ramp_off = 0.10  # extended gravity ramp

    phases = [
        (0.05, 2.55, 0, 1),     # compression in x
        (2.55, 7.55, 1, 1),     # shear in y
        (7.55, 12.5, 1, -1),    # reverse shear
    ]

    def before_step(rod, q, u, aux, t, dt_step):
        gz = -1000.0 if t <= ramp_off else 0.0
        F_ext = jnp.zeros_like(q)
        F_ext = F_ext.at[2::4].set(mass_z * gz)
        rod = rod.with_F_ext(F_ext)
        for (t_start, t_end, direction, sign) in phases:
            if t_start <= t < t_end:
                q, rod = dm.move_nodes(q, rod, start_nodes, 0.01 * dt_step * sign, direction)
                break
        return rod, q, u, aux

    stepper.before_step = before_step

    print(f"\n{'='*60}")
    print(f"Running N={N}, DOFs={q0.shape[0]}, dt={dt}, max_dt={stepper.max_dt}, eta={eta}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = stepper.simulate()
    elapsed = time.perf_counter() - t0

    n_steps = len(result.qs)
    qs = np.stack(result.qs)
    times = np.array(result.times)
    final_t = times[-1]

    mid = N // 2
    print(f"  Done: {elapsed:.1f}s wall, {n_steps} logged steps, final_t={final_t:.3f}s")
    print(f"  Steps/sec: {n_steps / elapsed:.1f}")

    save_data = {
        'qs': qs, 'times': times, 'elapsed': elapsed, 'n_steps': n_steps,
        'fixed_nodes': fixed, 'start_nodes': start_nodes,
        'elastic_energies': result.elastic_energies,
        'N': N, 'w_by_l': 1.0/12.0, 'total_time': total_time, 'dt': dt,
        'n_dof': int(q0.shape[0]),
    }

    save_path = f'/data/shivam/Ribbon/bench_results/jax_result_n{N}_wl12.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Saved: {save_path}")

    return save_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', type=float, default=0.0, help='Viscous damping coefficient')
    parser.add_argument('--nodes', type=int, nargs='+', default=[21, 45, 63, 85])
    args = parser.parse_args()

    node_counts = args.nodes
    results = {}

    for N in node_counts:
        results[N] = run_single(N, eta=args.eta)

    # Print efficiency summary
    print(f"\n{'='*70}")
    print(f"{'N':>5} {'DOFs':>6} {'Steps':>7} {'Wall(s)':>8} {'Final t':>8} {'Steps/s':>8} {'ms/step':>8}")
    print(f"{'-'*70}")
    for N in node_counts:
        r = results[N]
        ms_per_step = r['elapsed'] / r['n_steps'] * 1000
        print(f"{N:5d} {r['n_dof']:6d} {r['n_steps']:7d} {r['elapsed']:8.1f} {r['times'][-1]:8.3f} "
              f"{r['n_steps']/r['elapsed']:8.1f} {ms_per_step:8.1f}")


if __name__ == '__main__':
    main()
