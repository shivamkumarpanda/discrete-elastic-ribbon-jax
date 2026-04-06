import jax
import jax.numpy as jnp

from .stencil import Stencil
from ..states import TripletState
from ..util import material_frame, parallel_transport, signed_angle


class Triplet(Stencil):
    """DDG triplet stencil: computes 5-component strain [ε0, ε1, κ1, κ2, τ]."""

    l_k: jax.Array  # [l_ke, l_kf] — reference edge lengths

    def get_strain(self, q: jax.Array, aux: TripletState | None = None) -> jax.Array:
        """Compute strain vector from 11-DOF local q and material frame state.

        DOF layout: [n0x,n0y,n0z, θe, n1x,n1y,n1z, θf, n2x,n2y,n2z]
        Returns: [ε0, ε1, κ1, κ2, τ]
        """
        te_old, tf_old = aux.t
        d1e, d1f = aux.d1

        l_ke, l_kf = self.l_k

        n0 = q[0:3]
        n1 = q[4:7]
        n2 = q[8:11]
        theta_e = q[3]
        theta_f = q[7]

        # Compute current tangents
        ee = n1 - n0
        ef = n2 - n1
        te = ee / jnp.linalg.norm(ee)
        tf = ef / jnp.linalg.norm(ef)

        # Material directors via parallel transport + twist rotation
        m1e, m2e = material_frame(d1e, te_old, te, theta_e)
        m1f, m2f = material_frame(d1f, tf_old, tf, theta_f)

        # Stretch strains (per edge)
        eps0 = self.get_epsilon(n0, n1, l_ke)
        eps1 = self.get_epsilon(n1, n2, l_kf)

        # Curvature strains
        kappa1, kappa2 = self.get_kappa(n0, n1, n2, m1e, m2e, m1f, m2f)

        # Twist strain — computed geometrically (matching reference) so that
        # autodiff captures dτ/d(node_positions) through the parallel transport
        # and signed angle. The algebraic form τ = θf - θe + β gives the same
        # VALUE but has dτ/d(positions) = 0 when β is a constant from aux.
        tau = self._get_twist_geometric(m1e, m1f, te, tf)

        return jnp.array([eps0, eps1, kappa1, kappa2, tau])

    @staticmethod
    def _get_twist_geometric(
        m1e: jax.Array, m1f: jax.Array, te: jax.Array, tf: jax.Array
    ) -> jax.Array:
        """Compute twist strain geometrically: angle between PT(m1e) and m1f.

        This matches the reference implementation which computes twist as
        signed_angle(parallel_transport(m1e, te, tf), m1f, tf).
        """
        m1e_transported = parallel_transport(m1e, te, tf)
        return signed_angle(m1e_transported, m1f, tf)
