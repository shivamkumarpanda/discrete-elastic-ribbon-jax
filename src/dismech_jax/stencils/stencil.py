from abc import abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import equinox as eqx

from ..states import State


class Stencil(eqx.Module):
    """Base class for DDG stencils."""

    bar_strain: jax.Array

    @classmethod
    def init(cls, q: jax.Array, aux: State | None = None, **kwargs) -> Self:
        """Create a Stencil with bar_strain computed from initial configuration."""
        temp_instance = cls(bar_strain=jnp.empty(0), **kwargs)
        bar_strain = temp_instance.get_strain(q, aux)
        return cls(bar_strain=bar_strain, **kwargs)

    def get_energy(
        self, q: jax.Array, model: eqx.Module, aux: State | None = None
    ) -> jax.Array:
        """Compute scalar elastic energy: E = model(strain - bar_strain)."""
        del_strain = self.get_strain(q, aux) - self.bar_strain
        return model(del_strain)

    @abstractmethod
    def get_strain(self, q: jax.Array, aux: State | None = None) -> jax.Array:
        """Compute strain vector from local DOFs and auxiliary state."""

    @staticmethod
    def get_epsilon(n0: jax.Array, n1: jax.Array, l_k: jax.Array) -> jax.Array:
        """Stretch strain: ε = |n1 - n0| / l_k - 1."""
        edge = n1 - n0
        edge_len = jnp.linalg.norm(edge)
        return edge_len / l_k - 1.0

    @staticmethod
    def get_kappa(
        n0: jax.Array,
        n1: jax.Array,
        n2: jax.Array,
        m1e: jax.Array,
        m2e: jax.Array,
        m1f: jax.Array,
        m2f: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Discrete curvature strains κ1, κ2 from binormal curvature vector."""
        ee = n1 - n0
        ef = n2 - n1
        norm_e = jnp.linalg.norm(ee)
        norm_f = jnp.linalg.norm(ef)
        te = ee / norm_e
        tf = ef / norm_f
        chi = 1.0 + jnp.dot(te, tf)
        kb = 2.0 * jnp.cross(te, tf) / chi
        kappa1 = 0.5 * jnp.dot(kb, m2e + m2f)
        kappa2 = -0.5 * jnp.dot(kb, m1e + m1f)
        return kappa1, kappa2

    @staticmethod
    def get_tau(theta_e: jax.Array, theta_f: jax.Array, beta: jax.Array) -> jax.Array:
        """Discrete twist strain: τ = θ_f - θ_e + β."""
        return theta_f - theta_e + beta
