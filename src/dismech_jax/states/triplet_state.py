from __future__ import annotations
import jax
import jax.numpy as jnp

from .state import State
from ..util import parallel_transport, get_ref_twist


class TripletState(State):
    """Material frame state for a triplet of nodes (two edges)."""
    t: jax.Array   # [[te], [tf]] — unit tangent vectors, shape (2, 3)
    d1: jax.Array  # [[d1e], [d1f]] — reference directors, shape (2, 3)
    beta: jax.Array  # reference twist scalar

    def update(self, q: jax.Array) -> TripletState:
        """Update material frame after DOF change. q is local 11-DOF vector."""
        te_old, tf_old = self.t
        d1e_old, d1f_old = self.d1
        beta_old = self.beta

        n0 = q[0:3]
        n1 = q[4:7]
        n2 = q[8:11]

        ee = n1 - n0
        ef = n2 - n1
        te = ee / jnp.linalg.norm(ee)
        tf = ef / jnp.linalg.norm(ef)  # BUG FIX: was ee / norm(ef)

        d1e = parallel_transport(d1e_old, te_old, te)
        d1f = parallel_transport(d1f_old, tf_old, tf)

        t_new = jnp.array([te, tf])
        d1_new = jnp.array([d1e, d1f])
        beta_new = get_ref_twist(d1e, d1f, te, tf, beta_old)

        return TripletState(t_new, d1_new, beta_new)
