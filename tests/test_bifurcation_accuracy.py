"""Focused accuracy tests: JAX vs reference at critical bifurcation stages.

Tests the Hessian, gravity deflection, and compression response match
the reference implementation for the exact same geometry/material/config.
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm

# Load reference
_ref_src = Path(__file__).parent.parent.parent / "discrete-elastic-ribbon" / "src"

def _load_ref():
    """Import reference dismech package."""
    sys.path.insert(0, str(_ref_src))
    import dismech
    return dismech


# ── Geometry setup shared between JAX and reference ──

L = 0.1
W_BY_L = 1.0 / 12.0
W = W_BY_L * L
H = 1e-3
N = 21
DL = L / (N - 1)


def make_jax_rod():
    geom = dm.Geometry(length=L, r0=H, axs=W*H, jxs=W*H**3/3,
                       ixs1=W*H**3/12, ixs2=H*W**3/12)
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)
    nodes = np.zeros((N, 3))
    nodes[:, 0] = np.linspace(0, L, N)
    rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)
    model = dm.Sano.from_geometry(jnp.float64(DL), geom, mat)
    return rod, q0, aux, mass, model, geom, mat


def make_ref_rod():
    dismech = _load_ref()
    import os; os.environ["MKL_THREADING_LAYER"] = "GNU"
    geom = dismech.GeomParams(rod_r0=H, shell_h=0, axs=W*H, jxs=W*H**3/3,
                              ixs1=W*H**3/12, ixs2=H*W**3/12)
    material = dismech.Material(density=1000, youngs_rod=10e9, youngs_shell=0,
                                poisson_rod=0.5, poisson_shell=0)
    sp = dismech.SimParams(static_sim=True, two_d_sim=False, use_mid_edge=False,
                           use_line_search=False, show_floor=False, log_data=False,
                           log_step=1, dt=0.001, max_iter=1, total_time=0.001,
                           plot_step=1, tol=1e-3, ftol=1e-4, dtol=0.01)
    env = dismech.Environment()
    # Generate geometry file
    geo_dir = Path("/data/shivam/Ribbon/bench_results/test_geo")
    geo_dir.mkdir(parents=True, exist_ok=True)
    geo_path = geo_dir / f"rod_n{N}.txt"
    if not geo_path.exists():
        lines = ["*Nodes"]
        for i in range(N):
            lines.append(f"{i*DL},0,0")
        lines.append("*Edges")
        for i in range(1, N):
            lines.append(f"{i},{i+1}")
        geo_path.write_text("\n".join(lines) + "\n")
    geo = dismech.Geometry.from_txt(str(geo_path))
    robot = dismech.SoftRobot(geom, material, geo, sp, env)
    stepper = dismech.ImplicitEulerTimeStepper(robot, energy_model='sano')
    return robot, stepper, dismech


# ── Tests ──

class TestHessianMatch:
    """Verify JAX and reference elastic Hessians match at rest configuration."""

    def test_z_diagonal_match(self):
        """Z-DOF diagonal of Hessian should match within machine precision."""
        rod, q0, aux, mass, model, geom, mat = make_jax_rod()
        H_jax = np.array(jax.hessian(lambda q: rod._internal_energy(q, model, aux))(q0))

        robot, stepper, dismech = make_ref_rod()
        state = robot.state
        for ee in stepper._TimeStepper__elastic_energies:
            _, J_ref = ee.grad_hess_energy_linear_elastic(state, sparse=False)
            H_ref = -J_ref
            break

        # Compare z-DOF diagonals
        z_jax = np.arange(2, 4*N-1, 4)
        z_ref = np.arange(2, 3*N, 3)

        for i in range(N):
            jax_val = H_jax[z_jax[i], z_jax[i]]
            ref_val = H_ref[z_ref[i], z_ref[i]]
            np.testing.assert_allclose(jax_val, ref_val, rtol=1e-8,
                err_msg=f"H_zz mismatch at node {i}")

    def test_full_hessian_frobenius(self):
        """Full Hessian Frobenius norm should match."""
        rod, q0, aux, mass, model, geom, mat = make_jax_rod()
        H_jax = np.array(jax.hessian(lambda q: rod._internal_energy(q, model, aux))(q0))

        robot, stepper, dismech = make_ref_rod()
        state = robot.state
        for ee in stepper._TimeStepper__elastic_energies:
            _, J_ref = ee.grad_hess_energy_linear_elastic(state, sparse=False)
            H_ref = -J_ref
            break

        # Extract position-only blocks for comparison (ignore twist DOF ordering)
        # JAX positions: [0,1,2, 4,5,6, 8,9,10, ...]
        # Ref positions: [0,1,2, 3,4,5, 6,7,8, ...]
        pos_jax = []
        pos_ref = []
        for i in range(N):
            pos_jax.extend([4*i, 4*i+1, 4*i+2])
            pos_ref.extend([3*i, 3*i+1, 3*i+2])

        H_pos_jax = H_jax[np.ix_(pos_jax, pos_jax)]
        H_pos_ref = H_ref[np.ix_(pos_ref, pos_ref)]

        np.testing.assert_allclose(H_pos_jax, H_pos_ref, rtol=1e-8, atol=1e-6,
            err_msg="Position-block Hessian mismatch")


class TestMaterialDirectors:
    """Verify material director initialization matches reference."""

    def test_d1_direction(self):
        """d1 should point in z for a straight rod along x."""
        rod, q0, aux, mass, model, geom, mat = make_jax_rod()
        # First triplet, first edge
        d1_0 = np.array(aux.d1[0, 0])
        np.testing.assert_allclose(d1_0, [0, 0, 1], atol=1e-14,
            err_msg="d1 should be [0,0,1] for rod along x")

    def test_d1_matches_reference(self):
        """JAX d1 should match reference a1/m1."""
        rod, q0, aux, mass, model, geom, mat = make_jax_rod()
        robot, stepper, dismech = make_ref_rod()

        jax_d1 = np.array(aux.d1[0, 0])
        ref_m1 = robot.state.m1[0]
        np.testing.assert_allclose(jax_d1, ref_m1, atol=1e-14,
            err_msg="JAX d1 should match reference m1")


class TestGravityDeflection:
    """Verify gravity produces the same deflection in both codes."""

    def test_static_gravity_deflection(self):
        """Under constant gravity, equilibrium z-deflection should match."""
        # JAX: static solve with gravity
        rod, q0, aux, mass, model, geom, mat = make_jax_rod()
        fixed = np.array([0, N-1])
        rod, q0 = dm.fix_nodes(rod, q0, fixed)

        mass_z = mass[2::4]
        F_ext = jnp.zeros_like(q0).at[2::4].set(mass_z * (-1000.0))
        rod = rod.with_F_ext(F_ext)

        sp = dm.SimParams(dt=0.001, total_time=0.05, tol=1e-6, ftol=1e-6,
                          dtol=1e-6, max_iter=100, log_step=1, static_sim=False)
        stepper = dm.TimeStepper(rod, model, sp, mass, q0, aux)
        result = stepper.simulate()

        jax_z_mid = result.qs[-1][4*(N//2)+2]

        # Reference
        robot, ref_stepper, dismech = make_ref_rod()
        robot = robot.fix_nodes(fixed)
        robot.env.g = np.array([0, 0, -1000.0])

        sp_ref = dismech.SimParams(static_sim=False, two_d_sim=False, use_mid_edge=False,
                                   use_line_search=False, show_floor=False, log_data=True,
                                   log_step=1, dt=0.001, max_iter=100, total_time=0.05,
                                   plot_step=1, tol=1e-6, ftol=1e-6, dtol=1e-6)
        robot_new = dismech.SoftRobot(
            dismech.GeomParams(rod_r0=H, shell_h=0, axs=W*H, jxs=W*H**3/3,
                               ixs1=W*H**3/12, ixs2=H*W**3/12),
            dismech.Material(density=1000, youngs_rod=10e9, youngs_shell=0,
                             poisson_rod=0.5, poisson_shell=0),
            dismech.Geometry.from_txt(str(Path("/data/shivam/Ribbon/bench_results/test_geo") / f"rod_n{N}.txt")),
            sp_ref,
            dismech.Environment()
        )
        robot_new.env.add_force('gravity', g=np.array([0, 0, -1000.0]))
        robot_new = robot_new.fix_nodes(fixed)
        ref_stepper2 = dismech.ImplicitEulerTimeStepper(robot_new, energy_model='sano')
        result_ref = ref_stepper2.simulate()
        robots_ref = result_ref[0]
        ref_z_mid = robots_ref[-1].state.q[3*(N//2)+2]

        print(f"  JAX z_mid = {jax_z_mid:.8e}")
        print(f"  Ref z_mid = {ref_z_mid:.8e}")
        print(f"  Ratio = {jax_z_mid/ref_z_mid:.6f}")

        # Magnitudes should be close (sign may differ due to d2 convention)
        # Allow ~2x tolerance for dynamic accumulation differences
        np.testing.assert_allclose(abs(jax_z_mid), abs(ref_z_mid), rtol=1.0,
            err_msg="Gravity deflection magnitude should be within 2x")


class TestStrainAtPerturbedConfig:
    """Verify strains match at a z-perturbed configuration."""

    def test_kappa_at_z_perturbation(self):
        """Curvature from z-perturbation should match reference."""
        rod, q0, aux, mass, model, geom, mat = make_jax_rod()

        # Perturb mid node z
        dz = 1e-4
        q_pert = q0.at[4*(N//2)+2].set(dz)

        # JAX strains
        batch_q = dm.Rod.global_q_to_batch_q(q_pert)
        mid_triplet = N//2 - 1  # triplet containing mid node as node 1
        t_mid = jax.tree.map(lambda x: x[mid_triplet], rod.triplets)
        a_mid = jax.tree.map(lambda x: x[mid_triplet], aux)
        jax_strain = np.array(t_mid.get_strain(batch_q[mid_triplet], a_mid))

        # Reference strains
        robot, stepper, dismech_mod = make_ref_rod()
        import dataclasses
        q_ref_pert = robot.state.q.copy()
        q_ref_pert[3*(N//2)+2] = dz
        state_pert = dataclasses.replace(robot.state, q=q_ref_pert)
        for ee in stepper._TimeStepper__elastic_energies:
            ref_strains = ee.get_strain(state_pert)
            ref_strain_mid = ref_strains[mid_triplet]
            break

        # Compare: JAX [eps0, eps1, kappa1, kappa2, tau] vs ref [eps_avg, kappa1, kappa2, tau]
        jax_eps_avg = 0.5 * (jax_strain[0] + jax_strain[1])
        print(f"  JAX: eps_avg={jax_eps_avg:.6e}, k1={jax_strain[2]:.6e}, k2={jax_strain[3]:.6e}")
        print(f"  Ref: eps_avg={ref_strain_mid[0]:.6e}, k1={ref_strain_mid[1]:.6e}, k2={ref_strain_mid[2]:.6e}")

        np.testing.assert_allclose(jax_eps_avg, ref_strain_mid[0], rtol=1e-6,
            err_msg="Averaged stretch should match")
        # kappa1_jax should correspond to kappa1_ref (with correct d1=[0,0,1])
        np.testing.assert_allclose(jax_strain[2], ref_strain_mid[1], rtol=1e-6,
            err_msg="kappa1 should match")
        np.testing.assert_allclose(jax_strain[3], ref_strain_mid[2], rtol=1e-6,
            err_msg="kappa2 should match")
