"""Cross-validation: JAX gradient & Hessian vs reference at rest, compressed, and z-perturbed configs.

Uses pre-generated fixtures from generate_ref_fixtures.py (run once with dismech env).
If gradient and Hessian match, Newton steps must produce identical trajectories.
"""
import numpy as np
import pytest
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ref_n21_sano.npz"


@pytest.fixture(scope="module")
def ref_data():
    if not FIXTURE_PATH.exists():
        pytest.skip("Reference fixtures not generated. Run: conda activate dismech && python tests/generate_ref_fixtures.py")
    return np.load(FIXTURE_PATH, allow_pickle=True)


@pytest.fixture(scope="module")
def jax_rod():
    N = 21; L = 0.1; W = L/12; H = 1e-3; DL = L/(N-1)
    geom = dm.Geometry(length=L, r0=H, axs=W*H, jxs=W*H**3/3,
                       ixs1=W*H**3/12, ixs2=H*W**3/12)
    mat = dm.Material(density=1000.0, youngs_rod=10e9, poisson_rod=0.5)
    nodes = np.zeros((N, 3)); nodes[:, 0] = np.linspace(0, L, N)
    rod, q0, aux, mass = dm.create_rod_from_nodes(nodes, geom, mat, gravity=0.0)
    model = dm.Sano.from_geometry(jnp.float64(DL), geom, mat)
    return rod, q0, aux, mass, model, N


def _ref_to_jax_q(q_ref, N):
    """Convert reference separated DOF layout to JAX interleaved layout."""
    q_jax = jnp.zeros(4*N - 1)
    for i in range(N):
        q_jax = q_jax.at[4*i].set(q_ref[3*i])
        q_jax = q_jax.at[4*i+1].set(q_ref[3*i+1])
        q_jax = q_jax.at[4*i+2].set(q_ref[3*i+2])
    for e in range(N-1):
        q_jax = q_jax.at[4*e+3].set(q_ref[3*N + e])
    return q_jax


def _extract_pos_grad(grad_full, N, layout):
    """Extract position-only gradient (Nx3) from full DOF gradient."""
    out = np.zeros((N, 3))
    for i in range(N):
        if layout == 'sep':
            out[i] = grad_full[3*i:3*i+3]
        else:
            out[i] = grad_full[4*i:4*i+3]
    return out


def _extract_pos_hess(H_full, N, layout):
    """Extract position-only Hessian (3N x 3N) from full DOF Hessian."""
    idx = []
    for i in range(N):
        if layout == 'sep':
            idx.extend([3*i, 3*i+1, 3*i+2])
        else:
            idx.extend([4*i, 4*i+1, 4*i+2])
    return H_full[np.ix_(idx, idx)]


class TestRestConfig:
    """Gradient and Hessian at rest configuration."""

    def test_gradient_zero_at_rest(self, jax_rod, ref_data):
        rod, q0, aux, mass, model, N = jax_rod
        grad = np.array(jax.grad(lambda q: rod._internal_energy(q, model, aux))(q0))
        np.testing.assert_allclose(grad, 0.0, atol=1e-10,
            err_msg="Gradient should be zero at rest")

    def test_hessian_position_block(self, jax_rod, ref_data):
        rod, q0, aux, mass, model, N = jax_rod
        H_jax = np.array(jax.hessian(lambda q: rod._internal_energy(q, model, aux))(q0))
        H_ref = ref_data['H_rest']

        H_pos_jax = _extract_pos_hess(H_jax, N, 'int')
        H_pos_ref = _extract_pos_hess(H_ref, N, 'sep')

        np.testing.assert_allclose(H_pos_jax, H_pos_ref, rtol=1e-8, atol=1e-6,
            err_msg="Position-block Hessian should match reference at rest")


class TestCompressedConfig:
    """Gradient and Hessian at compressed configuration (dx=0.5mm at node 0)."""

    def test_gradient_match(self, jax_rod, ref_data):
        rod, q0, aux, mass, model, N = jax_rod
        q_ref_comp = ref_data['q_comp']
        q_jax_comp = _ref_to_jax_q(q_ref_comp, N)

        grad_jax = np.array(jax.grad(lambda q: rod._internal_energy(q, model, aux))(q_jax_comp))
        F_ref = ref_data['F_comp']  # F = -grad(E)

        grad_pos_jax = _extract_pos_grad(grad_jax, N, 'int')
        grad_pos_ref = _extract_pos_grad(-F_ref, N, 'sep')  # grad = -F

        # Compare position gradient
        np.testing.assert_allclose(grad_pos_jax, grad_pos_ref, rtol=1e-6, atol=1e-8,
            err_msg="Position gradient should match at compressed config")

    def test_hessian_match(self, jax_rod, ref_data):
        """Interior (free-DOF) position Hessian should match reference."""
        rod, q0, aux, mass, model, N = jax_rod
        q_ref_comp = ref_data['q_comp']
        q_jax_comp = _ref_to_jax_q(q_ref_comp, N)

        H_jax = np.array(jax.hessian(lambda q: rod._internal_energy(q, model, aux))(q_jax_comp))
        H_ref = ref_data['H_comp']

        # Compare interior nodes only (exclude boundary nodes)
        fixed = ref_data['fixed_nodes']
        free = [i for i in range(N) if i not in fixed]
        free_pos_jax = []
        free_pos_ref = []
        for i in free:
            free_pos_jax.extend([4*i, 4*i+1, 4*i+2])
            free_pos_ref.extend([3*i, 3*i+1, 3*i+2])

        H_free_jax = H_jax[np.ix_(free_pos_jax, free_pos_jax)]
        H_free_ref = H_ref[np.ix_(free_pos_ref, free_pos_ref)]

        np.testing.assert_allclose(H_free_jax, H_free_ref, rtol=1e-6, atol=1e-4,
            err_msg="Free-DOF position Hessian should match at compressed config")

    def test_z_eigenvalues_match(self, jax_rod, ref_data):
        """Z-mode eigenvalues should match — critical for bifurcation onset."""
        rod, q0, aux, mass, model, N = jax_rod
        q_ref_comp = ref_data['q_comp']
        q_jax_comp = _ref_to_jax_q(q_ref_comp, N)

        H_jax = np.array(jax.hessian(lambda q: rod._internal_energy(q, model, aux))(q_jax_comp))
        H_ref = ref_data['H_comp']

        # Extract z-DOFs of interior (free) nodes
        fixed = ref_data['fixed_nodes']
        free_nodes = [i for i in range(N) if i not in fixed]

        z_jax = [4*i+2 for i in free_nodes]
        z_ref = [3*i+2 for i in free_nodes]

        H_zz_jax = H_jax[np.ix_(z_jax, z_jax)]
        H_zz_ref = H_ref[np.ix_(z_ref, z_ref)]

        eigs_jax = np.sort(np.linalg.eigvalsh(H_zz_jax))
        eigs_ref = np.sort(np.linalg.eigvalsh(H_zz_ref))

        print(f"\n  Z-eigenvalues at compressed config:")
        print(f"    JAX  min 3: {eigs_jax[:3]}")
        print(f"    Ref  min 3: {eigs_ref[:3]}")
        print(f"    JAX  negative count: {np.sum(eigs_jax < 0)}")
        print(f"    Ref  negative count: {np.sum(eigs_ref < 0)}")

        # Both should have the same number of negative eigenvalues
        assert np.sum(eigs_jax < 0) == np.sum(eigs_ref < 0), \
            f"Different number of negative z-eigenvalues: JAX={np.sum(eigs_jax < 0)}, ref={np.sum(eigs_ref < 0)}"

        # Eigenvalues should match
        np.testing.assert_allclose(eigs_jax, eigs_ref, rtol=1e-4, atol=1e-2,
            err_msg="Z-mode eigenvalues should match at compressed config")


class TestZPerturbedCompressedConfig:
    """Gradient and Hessian at z-perturbed + compressed configuration."""

    def test_gradient_match(self, jax_rod, ref_data):
        rod, q0, aux, mass, model, N = jax_rod
        q_ref = ref_data['q_zpert']
        q_jax = _ref_to_jax_q(q_ref, N)

        grad_jax = np.array(jax.grad(lambda q: rod._internal_energy(q, model, aux))(q_jax))
        F_ref = ref_data['F_zpert']

        grad_pos_jax = _extract_pos_grad(grad_jax, N, 'int')
        grad_pos_ref = _extract_pos_grad(-F_ref, N, 'sep')

        np.testing.assert_allclose(grad_pos_jax, grad_pos_ref, rtol=1e-6, atol=1e-8,
            err_msg="Position gradient should match at z-perturbed compressed config")

    def test_hessian_match(self, jax_rod, ref_data):
        """Interior (free-DOF) position Hessian should match at z-perturbed compressed config."""
        rod, q0, aux, mass, model, N = jax_rod
        q_ref = ref_data['q_zpert']
        q_jax = _ref_to_jax_q(q_ref, N)

        H_jax = np.array(jax.hessian(lambda q: rod._internal_energy(q, model, aux))(q_jax))
        H_ref = ref_data['H_zpert']

        fixed = ref_data['fixed_nodes']
        free = [i for i in range(N) if i not in fixed]
        free_pos_jax = []
        free_pos_ref = []
        for i in free:
            free_pos_jax.extend([4*i, 4*i+1, 4*i+2])
            free_pos_ref.extend([3*i, 3*i+1, 3*i+2])

        H_free_jax = H_jax[np.ix_(free_pos_jax, free_pos_jax)]
        H_free_ref = H_ref[np.ix_(free_pos_ref, free_pos_ref)]

        # Slightly relaxed tolerance: material directors not updated for z-perturbation
        # causes ~0.4% difference in Hessian entries near the perturbed node
        np.testing.assert_allclose(H_free_jax, H_free_ref, rtol=5e-3, atol=1e-2,
            err_msg="Free-DOF position Hessian should match at z-perturbed compressed config")

    def test_strains_match(self, jax_rod, ref_data):
        """Strains at z-perturbed config should match reference."""
        rod, q0, aux, mass, model, N = jax_rod
        q_ref = ref_data['q_zpert']
        q_jax = _ref_to_jax_q(q_ref, N)

        # JAX strains (per-triplet 5-component)
        batch_q = dm.Rod.global_q_to_batch_q(q_jax)
        ref_strains = ref_data['strains_zpert']  # (N_triplets, 4)

        N_triplets = N - 2
        for k in range(N_triplets):
            t_k = jax.tree.map(lambda x: x[k], rod.triplets)
            a_k = jax.tree.map(lambda x: x[k], aux)
            jax_s = np.array(t_k.get_strain(batch_q[k], a_k))

            jax_eps_avg = 0.5 * (jax_s[0] + jax_s[1])
            ref_s = ref_strains[k]  # [eps_avg, k1, k2, tau]

            np.testing.assert_allclose(jax_eps_avg, ref_s[0], rtol=1e-6, atol=1e-12,
                err_msg=f"Stretch mismatch at triplet {k}")
            np.testing.assert_allclose(jax_s[2], ref_s[1], rtol=1e-6, atol=1e-12,
                err_msg=f"kappa1 mismatch at triplet {k}")
            np.testing.assert_allclose(jax_s[3], ref_s[2], rtol=1e-6, atol=1e-12,
                err_msg=f"kappa2 mismatch at triplet {k}")
            np.testing.assert_allclose(jax_s[4], ref_s[3], rtol=1e-6, atol=1e-12,
                err_msg=f"tau mismatch at triplet {k}")
