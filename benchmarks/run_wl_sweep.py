#!/usr/bin/env python3
"""Run bifurcation simulations across W/L ratios and energy models.

Tracks efficiency metrics: wall time, step count, dt statistics, Newton iterations.
"""
import time
import pickle
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm
from dismech_jax.models.kirchhoff import Kirchhoff, Sano, Audoly


ENERGY_MODELS = {
    'kirchhoff': Kirchhoff,
    'sano': Sano,
    'audoly': Audoly,
}


def run_single(N, w_by_l, energy_model_name, total_time=12.5, dt=0.032):
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

    start_nodes = np.where(nodes[:, 0] <= 0.01)[0]
    end_nodes = np.where(nodes[:, 0] >= 0.09)[0]
    fixed = np.union1d(start_nodes, end_nodes)
    rod, q0 = dm.fix_nodes(rod, q0, fixed)

    dl = L / (N - 1)
    model_cls = ENERGY_MODELS[energy_model_name]
    model = model_cls.from_geometry(jnp.float64(dl), geom, mat)

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
    stepper.max_dt_reductions = 80

    mass_z = mass[2::4]
    ramp_off = 0.05  # match reference

    phases = [
        (0.05, 2.55, 0, 1),
        (2.55, 7.55, 1, 1),
        (7.55, 12.5, 1, -1),
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

    wl_str = f"1b{int(1/w_by_l)}" if w_by_l < 1 else f"{w_by_l:.1f}"
    print(f"  {energy_model_name:>10} W/L={wl_str}: N={N}", end="", flush=True)

    t0 = time.perf_counter()
    result = stepper.simulate()
    elapsed = time.perf_counter() - t0

    n_steps = len(result.qs)
    qs = np.stack(result.qs)
    times = np.array(result.times)
    final_t = times[-1] if len(times) > 0 else 0
    print(f" → {elapsed:.1f}s, {n_steps} steps, t_final={final_t:.2f}")

    save_data = {
        'qs': qs, 'times': times, 'elapsed': elapsed, 'n_steps': n_steps,
        'fixed_nodes': fixed, 'start_nodes': start_nodes,
        'elastic_energies': result.elastic_energies,
        'N': N, 'w_by_l': w_by_l, 'total_time': total_time, 'dt': dt,
        'n_dof': int(q0.shape[0]), 'energy_model': energy_model_name,
    }

    save_path = f'/data/shivam/Ribbon/bench_results/jax_{energy_model_name}_n{N}_{wl_str}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    return save_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=45)
    parser.add_argument('--dt', type=float, default=0.032)
    parser.add_argument('--models', nargs='+', default=['kirchhoff', 'sano', 'audoly'])
    parser.add_argument('--wl', nargs='+', type=float,
                        default=[1/40, 1/20, 1/12, 1/6, 1/2])
    args = parser.parse_args()

    results = {}
    for w_by_l in args.wl:
        for model_name in args.models:
            key = (model_name, w_by_l)
            try:
                results[key] = run_single(
                    args.nodes, w_by_l, model_name, dt=args.dt)
            except Exception as e:
                print(f"  FAILED: {model_name} W/L={w_by_l}: {e}")
                import traceback; traceback.print_exc()
                results[key] = None

    # Efficiency summary
    print(f"\n{'='*90}")
    print(f"{'Model':>12} {'W/L':>8} {'Steps':>7} {'Wall(s)':>8} {'t_final':>8} "
          f"{'Steps/s':>8} {'ms/step':>8}")
    print(f"{'-'*90}")
    for (model_name, w_by_l), r in results.items():
        wl_str = f"1/{int(1/w_by_l)}" if w_by_l < 1 else f"{w_by_l}"
        if r is None:
            print(f"{model_name:>12} {wl_str:>8} {'FAIL':>7}")
        else:
            ms = r['elapsed'] / max(r['n_steps'], 1) * 1000
            sps = r['n_steps'] / max(r['elapsed'], 0.001)
            print(f"{model_name:>12} {wl_str:>8} {r['n_steps']:>7} "
                  f"{r['elapsed']:>8.1f} {r['times'][-1]:>8.2f} "
                  f"{sps:>8.1f} {ms:>8.1f}")


if __name__ == '__main__':
    main()
