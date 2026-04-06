"""Cross-validation tests: JAX energy models vs PyTorch reference implementations.

Tests energy values, gradients, and Hessians for identical physical inputs.
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import torch

# Direct import of reference energy modules (bypass the full dismech package to avoid numba)
_ref_dir = Path(__file__).parent.parent.parent / "discrete-elastic-ribbon" / "src" / "dismech" / "elastics"

def _load_ref_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _ref_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_kirchhoff_mod = _load_ref_module("ref_kirchhoff", "analytical_kirchhoff_elastic_energy.py")
_sano_mod = _load_ref_module("ref_sano", "analytical_sanos_elastic_energy.py")
_sadowsky_mod = _load_ref_module("ref_sadowsky", "analytical_sadowsky_elastic_energy.py")
_audoly_mod = _load_ref_module("ref_audoly", "analytical_audoly_elastic_energy.py")

AnalyticalKirchhoffElasticEnergy = _kirchhoff_mod.AnalyticalKirchhoffElasticEnergy
AnalyticalSanosElasticEnergy = _sano_mod.AnalyticalSanosElasticEnergy
AnalyticalSadowskyElasticEnergy = _sadowsky_mod.AnalyticalSadowskyElasticEnergy
AnalyticalAudolyElasticEnergy = _audoly_mod.AnalyticalAudolyElasticEnergy

# JAX models
from dismech_jax.models.kirchhoff import Kirchhoff, Sano, Sadowsky, Audoly, _compute_stiffness
from dismech_jax.params import Geometry, Material


# --- Shared test geometry ---

def ribbon_geometry():
    """Standard ribbon test geometry matching the bifurcation simulation."""
    L = 0.1
    w_by_l = 1.0 / 14.0
    w = w_by_l * L
    h = 1e-3
    return Geometry(
        length=L,
        r0=h,
        axs=w * h,
        jxs=w * h**3 / 3.0,
        ixs1=w * h**3 / 12.0,
        ixs2=h * w**3 / 12.0,
    ), Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)


def get_stiffness_params():
    """Get physical stiffness parameters for tests."""
    geom, mat = ribbon_geometry()
    EA, EI1, EI2, GJ = _compute_stiffness(geom, mat)
    N = 21
    dl = geom.length / (N - 1)
    h = geom.r0
    w = geom.axs / h
    zeta = float(np.sqrt(w**4 / (120.0 * h**2)))
    return float(EA), float(EI1), float(EI2), float(GJ), dl, h, w, zeta


# --- Helper: convert 5-component physical strain to 4-component normalized ---

def physical_to_normalized(del_strain_5, h, dl):
    """Convert [eps0, eps1, k1, k2, tau] to [eps_avg, k1*h/dl, k2*h/dl, tau*h/dl]."""
    eps_avg = 0.5 * (del_strain_5[0] + del_strain_5[1])
    return np.array([eps_avg, del_strain_5[2] * h / dl,
                     del_strain_5[3] * h / dl, del_strain_5[4] * h / dl])


# --- Test Data ---

TEST_STRAINS = [
    np.array([0.01, 0.01, 0.5, -0.3, 0.2]),       # moderate strains
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),           # zero strain
    np.array([0.001, 0.002, 5.0, 1.0, 3.0]),        # large bending/twist
    np.array([-0.005, -0.003, 0.1, 0.1, -0.1]),     # compression + small bend
    np.array([0.0, 0.0, 10.0, 0.0, 0.01]),          # pure bending + tiny twist
]


# ==================== KIRCHHOFF TESTS ====================

class TestKirchhoff:

    def setup_method(self):
        EA, EI1, EI2, GJ, self.dl, self.h, self.w, _ = get_stiffness_params()
        self.ref = AnalyticalKirchhoffElasticEnergy(EA, EI1, EI2, GJ, self.dl, self.h)
        geom, mat = ribbon_geometry()
        self.jax_model = Kirchhoff.from_geometry(jnp.float64(self.dl), geom, mat)

    @pytest.mark.parametrize("strain", TEST_STRAINS)
    def test_energy(self, strain):
        """JAX energy matches PyTorch reference energy."""
        x_nn = physical_to_normalized(strain, self.h, self.dl)
        ref_E = self.ref.forward(torch.tensor(x_nn.reshape(1, 4), dtype=torch.float64)).item()
        ref_E_phys = ref_E * 0.5 * self.ref.EA * self.dl  # un-normalize

        jax_E = float(self.jax_model(jnp.array(strain)))
        np.testing.assert_allclose(jax_E, ref_E_phys, rtol=1e-12,
                                   err_msg=f"Kirchhoff energy mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])  # skip zero for grad
    def test_gradient(self, strain):
        """JAX grad matches reference gradient (via chain rule back to 5-component)."""
        jax_grad = jax.grad(self.jax_model)(jnp.array(strain))
        jax_grad = np.array(jax_grad)

        # Reference: compute grad w.r.t. normalized 4-component, then chain-rule
        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, ref_grad_nn, _ = self.ref.compute_energy_grad_hess(x_nn)
        # un-normalize gradient and map back to 5-component
        ref_grad_phys = _map_grad_nn_to_5(ref_grad_nn, self.h, self.dl,
                                           self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_grad, ref_grad_phys, rtol=1e-10, atol=1e-15,
                                   err_msg=f"Kirchhoff gradient mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])
    def test_hessian(self, strain):
        """JAX hessian matches reference hessian."""
        jax_hess = np.array(jax.hessian(self.jax_model)(jnp.array(strain)))

        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, _, ref_hess_nn = self.ref.compute_energy_grad_hess(x_nn)
        ref_hess_phys = _map_hess_nn_to_5(ref_hess_nn, self.h, self.dl,
                                            self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_hess, ref_hess_phys, rtol=1e-10, atol=1e-15,
                                   err_msg=f"Kirchhoff Hessian mismatch for strain={strain}")


# ==================== SANO TESTS ====================

class TestSano:

    def setup_method(self):
        EA, EI1, EI2, GJ, self.dl, self.h, self.w, self.zeta = get_stiffness_params()
        self.ref = AnalyticalSanosElasticEnergy(EA, EI1, EI2, GJ, self.dl, self.zeta, self.h)
        geom, mat = ribbon_geometry()
        self.jax_model = Sano.from_geometry(jnp.float64(self.dl), geom, mat, zeta=self.zeta)

    @pytest.mark.parametrize("strain", TEST_STRAINS)
    def test_energy(self, strain):
        x_nn = physical_to_normalized(strain, self.h, self.dl)
        ref_E = self.ref.forward(torch.tensor(x_nn.reshape(1, 4), dtype=torch.float64)).item()
        ref_E_phys = ref_E * 0.5 * self.ref.EA * self.dl

        jax_E = float(self.jax_model(jnp.array(strain)))
        np.testing.assert_allclose(jax_E, ref_E_phys, rtol=1e-12,
                                   err_msg=f"Sano energy mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])
    def test_gradient(self, strain):
        jax_grad = np.array(jax.grad(self.jax_model)(jnp.array(strain)))

        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, ref_grad_nn, _ = self.ref.compute_energy_grad_hess(x_nn)
        ref_grad_phys = _map_grad_nn_to_5(ref_grad_nn, self.h, self.dl,
                                           self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_grad, ref_grad_phys, rtol=1e-10, atol=1e-15,
                                   err_msg=f"Sano gradient mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])
    def test_hessian(self, strain):
        jax_hess = np.array(jax.hessian(self.jax_model)(jnp.array(strain)))

        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, _, ref_hess_nn = self.ref.compute_energy_grad_hess(x_nn)
        ref_hess_phys = _map_hess_nn_to_5(ref_hess_nn, self.h, self.dl,
                                            self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_hess, ref_hess_phys, rtol=1e-10, atol=1e-15,
                                   err_msg=f"Sano Hessian mismatch for strain={strain}")


# ==================== SADOWSKY TESTS ====================

class TestSadowsky:

    def setup_method(self):
        EA, EI1, EI2, GJ, self.dl, self.h, self.w, _ = get_stiffness_params()
        self.ref = AnalyticalSadowskyElasticEnergy(EA, EI1, EI2, GJ, self.dl, self.h)
        geom, mat = ribbon_geometry()
        self.jax_model = Sadowsky.from_geometry(jnp.float64(self.dl), geom, mat)

    @pytest.mark.parametrize("strain", TEST_STRAINS)
    def test_energy(self, strain):
        x_nn = physical_to_normalized(strain, self.h, self.dl)
        ref_E = self.ref.forward(torch.tensor(x_nn.reshape(1, 4), dtype=torch.float64)).item()
        ref_E_phys = ref_E * 0.5 * self.ref.EA * self.dl

        jax_E = float(self.jax_model(jnp.array(strain)))
        np.testing.assert_allclose(jax_E, ref_E_phys, rtol=1e-12,
                                   err_msg=f"Sadowsky energy mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])
    def test_gradient(self, strain):
        jax_grad = np.array(jax.grad(self.jax_model)(jnp.array(strain)))

        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, ref_grad_nn, _ = self.ref.compute_energy_grad_hess(x_nn)
        ref_grad_phys = _map_grad_nn_to_5(ref_grad_nn, self.h, self.dl,
                                           self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_grad, ref_grad_phys, rtol=1e-10, atol=1e-15,
                                   err_msg=f"Sadowsky gradient mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])
    def test_hessian(self, strain):
        jax_hess = np.array(jax.hessian(self.jax_model)(jnp.array(strain)))

        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, _, ref_hess_nn = self.ref.compute_energy_grad_hess(x_nn)
        ref_hess_phys = _map_hess_nn_to_5(ref_hess_nn, self.h, self.dl,
                                            self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_hess, ref_hess_phys, rtol=1e-10, atol=1e-15,
                                   err_msg=f"Sadowsky Hessian mismatch for strain={strain}")


# ==================== AUDOLY TESTS ====================

class TestAudoly:

    def setup_method(self):
        EA, EI1, EI2, GJ, self.dl, self.h, self.w, _ = get_stiffness_params()
        self.ref = AnalyticalAudolyElasticEnergy(EA, EI1, EI2, GJ, self.dl, self.w, self.h)
        geom, mat = ribbon_geometry()
        self.jax_model = Audoly.from_geometry(jnp.float64(self.dl), geom, mat)

    @pytest.mark.parametrize("strain", TEST_STRAINS)
    def test_energy(self, strain):
        x_nn = physical_to_normalized(strain, self.h, self.dl)
        ref_E = self.ref.forward(torch.tensor(x_nn.reshape(1, 4), dtype=torch.float64)).item()
        ref_E_phys = ref_E * 0.5 * self.ref.EA * self.dl

        jax_E = float(self.jax_model(jnp.array(strain)))
        np.testing.assert_allclose(jax_E, ref_E_phys, rtol=1e-10,
                                   err_msg=f"Audoly energy mismatch for strain={strain}")

    @pytest.mark.parametrize("strain", TEST_STRAINS[1:])
    def test_gradient(self, strain):
        jax_grad = np.array(jax.grad(self.jax_model)(jnp.array(strain)))

        x_nn = physical_to_normalized(strain, self.h, self.dl)
        _, ref_grad_nn, _ = self.ref.compute_energy_grad_hess(x_nn)
        ref_grad_phys = _map_grad_nn_to_5(ref_grad_nn, self.h, self.dl,
                                           self.ref.EA, self.dl)
        np.testing.assert_allclose(jax_grad, ref_grad_phys, rtol=1e-8, atol=1e-14,
                                   err_msg=f"Audoly gradient mismatch for strain={strain}")


# ==================== STRAIN COMPUTATION TESTS ====================

class TestStrainComputation:
    """Test that JAX Triplet strain computation matches reference."""

    def test_straight_rod_zero_strain(self):
        """A straight rod should have zero bending/twist strain."""
        from dismech_jax.stencils import Triplet
        from dismech_jax.states import TripletState

        dl = 0.005
        q = jnp.array([0.0, 0.0, 0.0,  # n0
                        0.0,              # theta_e
                        dl, 0.0, 0.0,     # n1
                        0.0,              # theta_f
                        2*dl, 0.0, 0.0])  # n2

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        aux = TripletState(t_pair, d1_pair, jnp.array(0.0))

        triplet = Triplet.init(q, aux, l_k=jnp.array([dl, dl]))
        strain = triplet.get_strain(q, aux)

        # bar_strain should equal current strain, so del_strain = 0
        del_strain = strain - triplet.bar_strain
        np.testing.assert_allclose(np.array(del_strain), 0.0, atol=1e-15)

    def test_stretched_rod(self):
        """Stretching a rod should give nonzero eps0, eps1."""
        from dismech_jax.stencils import Triplet
        from dismech_jax.states import TripletState

        dl = 0.005
        stretch = 0.01
        q_init = jnp.array([0.0, 0.0, 0.0, 0.0,
                            dl, 0.0, 0.0, 0.0,
                            2*dl, 0.0, 0.0])
        q_stretched = jnp.array([0.0, 0.0, 0.0, 0.0,
                                 dl*(1+stretch), 0.0, 0.0, 0.0,
                                 2*dl*(1+stretch), 0.0, 0.0])

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        aux = TripletState(t_pair, d1_pair, jnp.array(0.0))

        triplet = Triplet.init(q_init, aux, l_k=jnp.array([dl, dl]))
        strain = triplet.get_strain(q_stretched, aux)
        del_strain = strain - triplet.bar_strain

        np.testing.assert_allclose(float(del_strain[0]), stretch, rtol=1e-10)
        np.testing.assert_allclose(float(del_strain[1]), stretch, rtol=1e-10)
        np.testing.assert_allclose(float(del_strain[2]), 0.0, atol=1e-14)
        np.testing.assert_allclose(float(del_strain[3]), 0.0, atol=1e-14)
        np.testing.assert_allclose(float(del_strain[4]), 0.0, atol=1e-14)

    def test_twisted_rod(self):
        """Twisting should give nonzero tau."""
        from dismech_jax.stencils import Triplet
        from dismech_jax.states import TripletState

        dl = 0.005
        q_init = jnp.array([0.0, 0.0, 0.0, 0.0,
                            dl, 0.0, 0.0, 0.0,
                            2*dl, 0.0, 0.0])
        twist_angle = 0.1
        q_twisted = jnp.array([0.0, 0.0, 0.0, 0.0,
                               dl, 0.0, 0.0, twist_angle,
                               2*dl, 0.0, 0.0])

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        aux = TripletState(t_pair, d1_pair, jnp.array(0.0))

        triplet = Triplet.init(q_init, aux, l_k=jnp.array([dl, dl]))
        strain = triplet.get_strain(q_twisted, aux)
        del_strain = strain - triplet.bar_strain

        # tau = theta_f - theta_e + beta
        # initial: theta_e=0, theta_f=0, beta=0 -> tau_bar=0
        # twisted: theta_e=0, theta_f=twist_angle, beta=0 -> tau=twist_angle
        np.testing.assert_allclose(float(del_strain[4]), twist_angle, rtol=1e-10)

    def test_bent_rod(self):
        """Bending in z should give nonzero kappa."""
        from dismech_jax.stencils import Triplet
        from dismech_jax.states import TripletState

        dl = 0.005
        q_init = jnp.array([0.0, 0.0, 0.0, 0.0,
                            dl, 0.0, 0.0, 0.0,
                            2*dl, 0.0, 0.0])
        bend_z = 0.001
        q_bent = jnp.array([0.0, 0.0, 0.0, 0.0,
                            dl, 0.0, 0.0, 0.0,
                            2*dl, 0.0, bend_z])

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        aux = TripletState(t_pair, d1_pair, jnp.array(0.0))

        triplet = Triplet.init(q_init, aux, l_k=jnp.array([dl, dl]))
        strain = triplet.get_strain(q_bent, aux)
        del_strain = strain - triplet.bar_strain

        # Bending in z should produce nonzero kappa
        assert abs(float(del_strain[2])) > 0 or abs(float(del_strain[3])) > 0, \
            "Bending should produce nonzero curvature"


# ==================== ROD ASSEMBLY TESTS ====================

class TestRodAssembly:
    """Test Rod system energy/force/Hessian computation."""

    def test_rod_creation(self):
        """Basic rod creation and DOF counting."""
        from dismech_jax.systems import Rod
        geom, mat = ribbon_geometry()
        rod, q0, aux = Rod.from_geometry(geom, mat, N=5)
        assert q0.shape[0] == 4 * 5 - 1  # 5 nodes * 4 - 1

    def test_zero_strain_zero_force(self):
        """At initial config (zero strain), internal elastic forces should be zero."""
        from dismech_jax.systems import Rod
        from dismech_jax.models.kirchhoff import Kirchhoff

        geom, mat = ribbon_geometry()
        rod_no_grav, q0, aux = Rod.from_geometry(geom, mat, N=5, gravity=0.0)
        model = Kirchhoff.from_geometry(jnp.float64(geom.length / 4), geom, mat)

        F_int = jax.grad(lambda q: rod_no_grav._internal_energy(q, model, aux))(q0)
        np.testing.assert_allclose(np.array(F_int), 0.0, atol=1e-14)

    def test_force_gradient_consistency(self):
        """Force should be negative gradient of internal energy."""
        from dismech_jax.systems import Rod
        from dismech_jax.models.kirchhoff import Kirchhoff

        geom, mat = ribbon_geometry()
        rod, q0, aux = Rod.from_geometry(geom, mat, N=5, gravity=0.0)
        model = Kirchhoff.from_geometry(jnp.float64(geom.length / 4), geom, mat)

        # Perturb
        key = jax.random.PRNGKey(42)
        dq = jax.random.normal(key, q0.shape) * 1e-4
        q = q0 + dq

        E = rod._internal_energy(q, model, aux)
        grad_E = jax.grad(lambda _q: rod._internal_energy(_q, model, aux))(q)

        # Finite difference check
        eps_fd = 1e-7
        for i in range(min(5, len(q))):
            q_plus = q.at[i].add(eps_fd)
            q_minus = q.at[i].add(-eps_fd)
            E_plus = rod._internal_energy(q_plus, model, aux)
            E_minus = rod._internal_energy(q_minus, model, aux)
            fd_grad = (E_plus - E_minus) / (2 * eps_fd)
            np.testing.assert_allclose(float(grad_E[i]), float(fd_grad),
                                       rtol=1e-5, atol=1e-12,
                                       err_msg=f"Gradient mismatch at DOF {i}")


# ==================== UTILITY TESTS ====================

class TestTripletStateUpdate:
    """Test the TripletState.update bug fix."""

    def test_update_straight_rod(self):
        """Update with same config should preserve state."""
        from dismech_jax.states import TripletState

        dl = 0.005
        q = jnp.array([0.0, 0.0, 0.0, 0.0,
                        dl, 0.0, 0.0, 0.0,
                        2*dl, 0.0, 0.0])

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        aux = TripletState(t_pair, d1_pair, jnp.array(0.0))

        new_aux = aux.update(q)
        np.testing.assert_allclose(np.array(new_aux.t), np.array(aux.t), atol=1e-14)
        np.testing.assert_allclose(np.array(new_aux.d1), np.array(aux.d1), atol=1e-14)


class TestParallelTransport:
    """Test util functions."""

    def test_identity_transport(self):
        """Transport along same direction should preserve vector."""
        from dismech_jax.util import parallel_transport
        u = jnp.array([0.0, 1.0, 0.0])
        t = jnp.array([1.0, 0.0, 0.0])
        result = parallel_transport(u, t, t)
        np.testing.assert_allclose(np.array(result), np.array(u), atol=1e-14)

    def test_90_degree_rotation(self):
        """Transport d1=[0,1,0] from t0=[1,0,0] to t1=[0,1,0]."""
        from dismech_jax.util import parallel_transport
        u = jnp.array([0.0, 1.0, 0.0])
        t0 = jnp.array([1.0, 0.0, 0.0])
        t1 = jnp.array([0.0, 1.0, 0.0])
        result = parallel_transport(u, t0, t1)
        # After transporting, u should rotate
        assert abs(float(jnp.linalg.norm(result)) - 1.0) < 1e-14
        # u should be perpendicular to t1
        assert abs(float(jnp.dot(result, t1))) < 1e-14


# ==================== HELPER FUNCTIONS ====================

def _map_grad_nn_to_5(grad_nn_4, h, dl, EA, delta_l):
    """Map 4-component normalized gradient to 5-component physical gradient.

    Reference: grad_nn is ∂E_nn/∂x_nn where x_nn = [eps_avg, k1*h/dl, k2*h/dl, tau*h/dl]
    and E_nn = E_phys / (0.5 * EA * dl).

    JAX: grad is ∂E_phys/∂[eps0, eps1, k1, k2, tau]

    Chain rule:
        ∂E/∂eps0 = ∂E/∂eps_avg * ∂eps_avg/∂eps0 = (0.5*EA*dl)*grad_nn[0] * 0.5
        ∂E/∂eps1 = same
        ∂E/∂k1 = (0.5*EA*dl)*grad_nn[1] * (h/dl)
        ∂E/∂k2 = (0.5*EA*dl)*grad_nn[2] * (h/dl)
        ∂E/∂tau = (0.5*EA*dl)*grad_nn[3] * (h/dl)
    """
    scale = 0.5 * EA * delta_l
    h_dl = h / dl

    grad_phys = np.zeros(5)
    grad_phys[0] = scale * grad_nn_4[0] * 0.5   # ∂E/∂eps0
    grad_phys[1] = scale * grad_nn_4[0] * 0.5   # ∂E/∂eps1
    grad_phys[2] = scale * grad_nn_4[1] * h_dl   # ∂E/∂k1
    grad_phys[3] = scale * grad_nn_4[2] * h_dl   # ∂E/∂k2
    grad_phys[4] = scale * grad_nn_4[3] * h_dl   # ∂E/∂tau
    return grad_phys


def _map_hess_nn_to_5(hess_nn_4, h, dl, EA, delta_l):
    """Map 4x4 normalized Hessian to 5x5 physical Hessian.

    Uses the Jacobian of the coordinate transform:
        J[i,a] = ∂(5-comp)_i / ∂(4-comp_nn)_a

    H_5x5 = J^T @ (scale * H_4x4) @ J
    """
    scale = 0.5 * EA * delta_l
    h_dl = h / dl

    # Jacobian mapping 4-comp nn -> 5-comp physical
    # x_nn[0] = eps_avg -> depends on eps0, eps1 via eps_avg = 0.5*(eps0+eps1)
    # x_nn[1] = k1 * h/dl
    # x_nn[2] = k2 * h/dl
    # x_nn[3] = tau * h/dl

    # d(x_nn)/d(x_5): 4x5 matrix
    # ∂x_nn[0]/∂eps0 = 0.5, ∂x_nn[0]/∂eps1 = 0.5
    # ∂x_nn[1]/∂k1 = h/dl
    # ∂x_nn[2]/∂k2 = h/dl
    # ∂x_nn[3]/∂tau = h/dl
    J = np.zeros((4, 5))
    J[0, 0] = 0.5
    J[0, 1] = 0.5
    J[1, 2] = h_dl
    J[2, 3] = h_dl
    J[3, 4] = h_dl

    # H_5x5 = J^T @ (scale * H_4x4) @ J
    return J.T @ (scale * hess_nn_4) @ J
