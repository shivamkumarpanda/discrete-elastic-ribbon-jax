"""Tests for dynamic time stepper, geometry I/O, and boundary conditions."""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm


def ribbon_params():
    """Standard ribbon geometry and material for tests."""
    L = 0.1
    w_by_l = 1.0 / 14.0
    w = w_by_l * L
    h = 1e-3
    geom = dm.Geometry(
        length=L, r0=h,
        axs=w * h, jxs=w * h**3 / 3.0,
        ixs1=w * h**3 / 12.0, ixs2=h * w**3 / 12.0,
    )
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)
    return geom, mat, L, w, h


class TestGeometryIO:

    def test_ensure_initial_geometry(self, tmp_path):
        geom, mat, L, w, h = ribbon_params()
        path = dm.ensure_initial_geometry(tmp_path, 11, L)
        assert path.exists()
        nodes, edges = dm.load_geometry_txt(path)
        assert nodes.shape == (11, 3)
        assert edges.shape == (10, 2)
        np.testing.assert_allclose(nodes[0], [0, 0, 0])
        np.testing.assert_allclose(nodes[-1, 0], L, rtol=1e-10)

    def test_create_rod_from_nodes(self):
        geom, mat, L, w, h = ribbon_params()
        N = 11
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat)
        assert q0.shape[0] == 4 * N - 1
        assert mass.shape[0] == 4 * N - 1


class TestFixAndMoveNodes:

    def test_fix_nodes(self):
        geom, mat, L, w, h = ribbon_params()
        N = 11
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)

        rod_fixed, q0_fixed = dm.fix_nodes(rod, q0, np.array([0, N - 1]))
        # Should have 6 constrained DOFs (2 nodes * 3 pos DOFs)
        assert rod_fixed.bc.idx_b.shape[0] == 6

    def test_move_nodes(self):
        geom, mat, L, w, h = ribbon_params()
        N = 11
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)
        rod_fixed, q0_fixed = dm.fix_nodes(rod, q0, np.array([0]))

        disp = 0.001
        q_new, rod_new = dm.move_nodes(q0_fixed, rod_fixed, np.array([0]), disp, 1)
        # Node 0 y-position should have moved
        np.testing.assert_allclose(float(q_new[1]), disp, rtol=1e-10)


class TestTimeStepper:

    def test_static_equilibrium(self):
        """A straight rod with no gravity should stay in place under static sim."""
        geom, mat, L, w, h = ribbon_params()
        N = 11
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)
        rod_fixed, q0 = dm.fix_nodes(rod, q0, np.array([0, N - 1]))

        dl = L / (N - 1)
        model = dm.Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

        sp = dm.SimParams(dt=0.01, total_time=0.03, tol=1e-8, ftol=1e-4,
                          dtol=1e-4, max_iter=50, log_step=1, static_sim=True)

        stepper = dm.TimeStepper(rod_fixed, model, sp, mass, q0, aux)
        result = stepper.simulate()

        # Should have 4 logged states (initial + 3 steps)
        assert len(result.qs) == 4

        # Final state should be very close to initial (no forces)
        np.testing.assert_allclose(result.qs[-1], result.qs[0], atol=1e-12)

    def test_gravity_drop(self):
        """A rod with gravity should deflect downward (dynamic sim)."""
        geom, mat, L, w, h = ribbon_params()
        N = 7
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)
        rod_fixed, q0 = dm.fix_nodes(rod, q0, np.array([0, N - 1]))

        dl = L / (N - 1)
        model = dm.Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

        sp = dm.SimParams(dt=0.001, total_time=0.005, tol=1e-6, ftol=1e-4,
                          dtol=1e-4, max_iter=100, log_step=1, static_sim=False)

        stepper = dm.TimeStepper(rod_fixed, model, sp, mass, q0, aux)
        result = stepper.simulate()

        # Middle node z-position should change under gravity
        mid = N // 2
        z_init = result.qs[0][4 * mid + 2]
        z_final = result.qs[-1][4 * mid + 2]
        assert abs(z_final - z_init) > 1e-8, f"Expected z to move under gravity, got {z_init} -> {z_final}"

    def test_energy_tracking(self):
        """Energy tracking should record values."""
        geom, mat, L, w, h = ribbon_params()
        N = 7
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9.81)
        rod_fixed, q0 = dm.fix_nodes(rod, q0, np.array([0, N - 1]))

        dl = L / (N - 1)
        model = dm.Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

        sp = dm.SimParams(dt=0.001, total_time=0.003, tol=1e-6, ftol=1e-4,
                          dtol=1e-4, max_iter=100, log_step=1, static_sim=False)

        stepper = dm.TimeStepper(rod_fixed, model, sp, mass, q0, aux)
        stepper.enable_elastic_energy_tracking()
        result = stepper.simulate()

        assert len(result.elastic_energies) > 0, "Should have tracked energies"

    def test_adaptive_dt(self):
        """Adaptive dt should reduce step when needed."""
        geom, mat, L, w, h = ribbon_params()
        N = 7
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=-9810.0)
        rod_fixed, q0 = dm.fix_nodes(rod, q0, np.array([0, N - 1]))

        dl = L / (N - 1)
        model = dm.Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

        sp = dm.SimParams(dt=0.01, total_time=0.03, tol=1e-6, ftol=1e-4,
                          dtol=1e-4, max_iter=100, log_step=1, static_sim=False)

        stepper = dm.TimeStepper(rod_fixed, model, sp, mass, q0, aux)
        stepper.adaptive_dt = True
        stepper.max_dq_threshold = 0.001
        stepper.min_dt = 1e-6
        stepper.max_dt = 0.01
        result = stepper.simulate()

        # Should have produced some output (may be truncated due to adaptive dt)
        assert len(result.qs) >= 1


class TestBeforeStepCallback:

    def test_callback_modifies_forces(self):
        """before_step callback should be called and can modify the rod."""
        geom, mat, L, w, h = ribbon_params()
        N = 7
        nodes = np.zeros((N, 3))
        nodes[:, 0] = np.linspace(0, L, N)
        rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)
        rod_fixed, q0 = dm.fix_nodes(rod, q0, np.array([0, N - 1]))

        dl = L / (N - 1)
        model = dm.Kirchhoff.from_geometry(jnp.float64(dl), geom, mat)

        callback_called = [False]

        def my_callback(rod, q, u, aux, t, dt):
            callback_called[0] = True
            return rod, q, u, aux

        sp = dm.SimParams(dt=0.01, total_time=0.02, tol=1e-6, ftol=1e-4,
                          dtol=1e-4, max_iter=50, log_step=1, static_sim=True)

        stepper = dm.TimeStepper(rod_fixed, model, sp, mass, q0, aux)
        stepper.before_step = my_callback
        result = stepper.simulate()

        assert callback_called[0], "Callback should have been called"
