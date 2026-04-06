#!/usr/bin/env python3
"""Generate reference fixtures for cross-validation tests.

Run ONCE with: conda activate dismech && python tests/generate_ref_fixtures.py
Saves compressed-state gradient, Hessian, and strains from reference implementation.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "discrete-elastic-ribbon" / "src"))
import dismech

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)

N = 21
L = 0.1
W = L / 12.0
H = 1e-3
DL = L / (N - 1)


def run():
    geom = dismech.GeomParams(rod_r0=H, shell_h=0, axs=W*H, jxs=W*H**3/3,
                              ixs1=W*H**3/12, ixs2=H*W**3/12)
    material = dismech.Material(density=1000, youngs_rod=10e9, youngs_shell=0,
                                poisson_rod=0.5, poisson_shell=0)
    sp = dismech.SimParams(static_sim=True, two_d_sim=False, use_mid_edge=False,
                           use_line_search=False, show_floor=False, log_data=False,
                           log_step=1, dt=0.001, max_iter=1, total_time=0.001,
                           plot_step=1, tol=1e-3, ftol=1e-4, dtol=0.01)
    env = dismech.Environment()

    geo_dir = FIXTURE_DIR / "geo"
    geo_dir.mkdir(exist_ok=True)
    geo_path = geo_dir / f"rod_n{N}.txt"
    if not geo_path.exists():
        lines = ["*Nodes"]
        for i in range(N):
            lines.append(f"{i * DL},0,0")
        lines.append("*Edges")
        for i in range(1, N):
            lines.append(f"{i},{i + 1}")
        geo_path.write_text("\n".join(lines) + "\n")

    geo = dismech.Geometry.from_txt(str(geo_path))
    robot = dismech.SoftRobot(geom, material, geo, sp, env)
    node_pos = robot.state.q[robot.node_dof_indices].reshape(-1, 3)
    start = np.where(node_pos[:, 0] <= 0.01)[0]
    end_n = np.where(node_pos[:, 0] >= 0.09)[0]
    fixed = np.union1d(start, end_n)
    robot = robot.fix_nodes(fixed)

    stepper = dismech.ImplicitEulerTimeStepper(robot, energy_model='sano')

    # ── Fixture 1: Hessian at rest ──
    state_rest = robot.state
    for ee in stepper._TimeStepper__elastic_energies:
        F_rest, J_rest = ee.grad_hess_energy_linear_elastic(state_rest, sparse=False)
        break
    H_rest = -J_rest

    # ── Fixture 2: Hessian at compressed config ──
    # Compress node 0 by 0.5mm in x
    import dataclasses
    q_comp = state_rest.q.copy()
    q_comp[0] += 0.0005  # dx = 0.5mm
    state_comp = dataclasses.replace(state_rest, q=q_comp)
    # Recompute material directors for compressed state
    for ee in stepper._TimeStepper__elastic_energies:
        strains_comp = ee.get_strain(state_comp)
        F_comp, J_comp = ee.grad_hess_energy_linear_elastic(state_comp, sparse=False)
        break
    H_comp = -J_comp

    # ── Fixture 3: Hessian at z-perturbed + compressed config ──
    q_zpert = q_comp.copy()
    mid = N // 2
    q_zpert[3 * mid + 2] = 1e-4  # z perturbation at mid node
    state_zpert = dataclasses.replace(state_rest, q=q_zpert)
    for ee in stepper._TimeStepper__elastic_energies:
        strains_zpert = ee.get_strain(state_zpert)
        F_zpert, J_zpert = ee.grad_hess_energy_linear_elastic(state_zpert, sparse=False)
        break
    H_zpert = -J_zpert

    # ── Save ──
    np.savez(FIXTURE_DIR / "ref_n21_sano.npz",
             # Rest
             q_rest=state_rest.q, H_rest=H_rest, F_rest=F_rest,
             # Compressed
             q_comp=q_comp, H_comp=H_comp, F_comp=F_comp, strains_comp=strains_comp,
             # Z-perturbed + compressed
             q_zpert=q_zpert, H_zpert=H_zpert, F_zpert=F_zpert, strains_zpert=strains_zpert,
             # Metadata
             fixed_nodes=fixed, start_nodes=start,
             m1_rest=state_rest.m1, m2_rest=state_rest.m2,
             mass=robot.mass_matrix,
             N=N, L=L, W=W, H_thickness=H, DL=DL)

    print(f"Saved fixtures to {FIXTURE_DIR / 'ref_n21_sano.npz'}")
    print(f"  H_rest norm: {np.linalg.norm(H_rest):.4e}")
    print(f"  H_comp norm: {np.linalg.norm(H_comp):.4e}")
    print(f"  H_zpert norm: {np.linalg.norm(H_zpert):.4e}")
    print(f"  F_comp norm: {np.linalg.norm(F_comp):.4e}")
    print(f"  F_zpert norm: {np.linalg.norm(F_zpert):.4e}")


if __name__ == "__main__":
    run()
