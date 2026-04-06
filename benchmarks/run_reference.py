#!/usr/bin/env python3
"""Run the reference (NumPy/PyTorch) bifurcation simulation and save results.

Must be run with: conda run -n dismech python benchmarks/run_reference.py
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import time
import pickle
import numpy as np

sys.path.insert(0, '/data/shivam/Ribbon/discrete-elastic-ribbon/src')
import dismech

def run_reference(N=21, w_by_l=1.0/12.0, total_time=12.5, dt=0.001):
    L = 0.1
    w = w_by_l * L
    h = 1e-3

    geom = dismech.GeomParams(
        rod_r0=h, shell_h=0,
        axs=w * h, jxs=w * h**3 / 3,
        ixs1=w * h**3 / 12, ixs2=h * w**3 / 12,
    )
    material = dismech.Material(
        density=1000.0, youngs_rod=10e9, youngs_shell=0,
        poisson_rod=0.5, poisson_shell=0,
    )
    sim_params = dismech.SimParams(
        static_sim=False, two_d_sim=False, use_mid_edge=False,
        use_line_search=False, show_floor=False,
        log_data=True, log_step=1, dt=dt, max_iter=100,
        total_time=total_time, plot_step=1,
        tol=0.001, ftol=0.0001, dtol=0.01,
    )
    env = dismech.Environment()
    env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))

    # Generate geometry
    from pathlib import Path
    geo_dir = Path('/data/shivam/Ribbon/bench_results/ref_geo')
    geo_dir.mkdir(parents=True, exist_ok=True)
    geo_path = geo_dir / f'rod_n{N}.txt'
    if not geo_path.exists():
        dl_val = L / (N - 1)
        lines = ['*Nodes']
        for i in range(N):
            lines.append(f'{i * dl_val},0,0')
        lines.append('*Edges')
        for i in range(1, N):
            lines.append(f'{i},{i + 1}')
        geo_path.write_text('\n'.join(lines) + '\n')

    geo = dismech.Geometry.from_txt(str(geo_path))
    robot = dismech.SoftRobot(geom, material, geo, sim_params, env)

    # Fix boundary nodes
    node_pos = robot.state.q[robot.node_dof_indices].reshape(-1, 3)
    start = np.where(node_pos[:, 0] <= 0.01)[0]
    end = np.where(node_pos[:, 0] >= 0.09)[0]
    fixed = np.union1d(start, end)
    robot = robot.fix_nodes(fixed)

    stepper = dismech.ImplicitEulerTimeStepper(
        robot, energy_model='sano', sano_zeta=None,
    )

    # BC: gravity ramp + compression + shear + reverse
    u0 = 0.01
    ramp_end = 0.05
    g_ramp = np.array([0.0, 0.0, -1000.0])
    g_after = np.array([0.0, 0.0, 0.0])
    phases = [
        {'start_time': 0.05, 'end_time': 2.55, 'direction': 0, 'reverse': False},
        {'start_time': 2.55, 'end_time': 7.55, 'direction': 1, 'reverse': False},
        {'start_time': 7.55, 'end_time': 12.5, 'direction': 1, 'reverse': True},
    ]

    def move_and_twist(robot, t):
        if t == robot.sim_params.dt:
            return robot
        if t <= ramp_end:
            robot.env.g = g_ramp.copy()
        else:
            robot.env.g = g_after.copy()
        for ph in phases:
            if ph['start_time'] <= t < ph['end_time']:
                sign = -1 if ph.get('reverse', False) else 1
                robot = robot.move_nodes(start, u0 * robot.sim_params.dt * sign, ph['direction'])
                break
        return robot

    stepper.before_step = move_and_twist
    stepper.adaptive_dt = True
    stepper.max_dq_threshold = 0.1
    stepper.dt_reduction_factor = 0.5
    base_dt = sim_params.dt
    stepper.min_dt = base_dt / 1e6
    stepper.max_dt = base_dt * 2.0
    stepper.max_dt_reductions = 40

    stepper.enable_elastic_energy_tracking()

    print(f"Running reference: N={N}, W/L={w_by_l:.4f}, total_time={total_time}s")
    t0 = time.perf_counter()
    result = stepper.simulate()
    elapsed = time.perf_counter() - t0

    robots, tracked_forces_list, tracked_forces_times, \
        condition_numbers, condition_number_times, \
        elastic_energies, elastic_energy_times, \
        material_directors_list, material_directors_times = result

    qs = np.stack([r.state.q for r in robots])
    n_steps = len(robots)
    print(f"  Done: {elapsed:.1f}s, {n_steps} logged steps, {n_steps/elapsed:.1f} steps/s")

    save_path = f'/data/shivam/Ribbon/bench_results/ref_result_n{N}_wl{int(1/w_by_l)}.pkl'
    save_data = {
        'qs': qs,
        'elapsed': elapsed,
        'n_steps': n_steps,
        'fixed_nodes': fixed,
        'start_nodes': start,
        'elastic_energies': elastic_energies,
        'elastic_energy_times': elastic_energy_times,
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
    run_reference(N=args.nodes, total_time=args.total_time)
