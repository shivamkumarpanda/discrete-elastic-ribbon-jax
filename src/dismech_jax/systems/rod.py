from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from ..models import DER
from ..stencils import Triplet
from ..states import TripletState
from ..params import Geometry, Material
from .system import System
from ..solver import solve


class BC(eqx.Module):
    """Linear boundary condition: q[idx_b] = xb_m * lambda + xb_c."""

    idx_b: jax.Array
    xb_m: jax.Array
    xb_c: jax.Array


class Rod(System[TripletState]):
    triplets: Triplet
    F_ext: jax.Array
    bc: BC

    @classmethod
    def from_geometry(
        cls,
        geom: Geometry,
        material: Material,
        N: int = 30,
        bc: BC = BC(jnp.empty(0, dtype=jnp.int32), jnp.empty(0), jnp.empty(0)),
        origin: jax.Array = jnp.array([0.0, 0.0, 0.0]),
        gravity: float = -9.81,
    ) -> tuple[Rod, jax.Array, TripletState]:
        if N < 3:
            raise ValueError("Cannot create a rod with less than 3 nodes.")
        if geom.length < 1e-6:
            raise ValueError("Cannot create a rod less than 1 um.")

        # DOF layout: [x0,y0,z0, θ0, x1,y1,z1, θ1, ..., xN-1,yN-1,zN-1]
        # Total DOFs: 4*N - 1 (N nodes × 3 pos + (N-1) edge twists)
        q0 = jnp.zeros(4 * N - 1)
        xs = jnp.linspace(0, geom.length, N) + origin[0]
        q0 = q0.at[0::4].set(xs)
        q0 = q0.at[1::4].set(origin[1])
        q0 = q0.at[2::4].set(origin[2])
        batch_q = Rod.global_q_to_batch_q(q0)

        l_ks = jnp.diff(xs)
        mass = Rod.get_mass(geom, material, l_ks)

        N_triplets = batch_q.shape[0]
        batch_l_ks = jax.vmap(lambda i: jax.lax.dynamic_slice(l_ks, (i,), (2,)))(
            jnp.arange(N_triplets)
        )

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        ts = jnp.broadcast_to(t_pair, (N_triplets, 2, 3))
        d1s = jnp.broadcast_to(d1_pair, (N_triplets, 2, 3))
        betas = jnp.zeros(N_triplets)

        batch_aux = jax.vmap(TripletState)(ts, d1s, betas)
        triplets = jax.vmap(lambda q, a, l_k: Triplet.init(q, a, l_k=l_k))(
            batch_q, batch_aux, batch_l_ks
        )

        F_ext = jnp.zeros_like(q0).at[2::4].set(mass[2::4] * gravity)
        rod = Rod(
            triplets=triplets,
            F_ext=F_ext,
            bc=bc,
        )
        return rod, q0, batch_aux

    def with_bc(self, bc: BC) -> Rod:
        return eqx.tree_at(lambda r: r.bc, self, bc)

    def with_F_ext(self, F_ext: jax.Array) -> Rod:
        return eqx.tree_at(lambda r: r.F_ext, self, F_ext)

    def get_DER(self, geom: Geometry, material: Material) -> DER:
        return DER.from_legacy(self.triplets.l_k[0, 0], geom, material)

    def get_E(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        batch_qs = self.global_q_to_batch_q(q)
        return jnp.sum(
            jax.vmap(lambda t, q_loc, _aux: t.get_energy(q_loc, model, _aux))(
                self.triplets, batch_qs, aux
            )
        ) - jnp.dot(self.F_ext, q)  # Include external force potential

    def get_q(self, _lambda: jax.Array, q0: jax.Array) -> jax.Array:
        has_bc = self.bc.idx_b.shape[0] > 0
        return jnp.where(has_bc, q0.at[self.bc.idx_b].set(self.bc.xb_m * _lambda + self.bc.xb_c), q0)

    def get_F(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        mask = jnp.ones_like(q).at[self.bc.idx_b].set(0.0)
        F_int = jax.grad(lambda _q: self._internal_energy(_q, model, aux))(q)
        return mask * (self.F_ext - F_int)

    def get_H(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        mask = jnp.ones_like(q).at[self.bc.idx_b].set(0.0)
        H = jax.hessian(lambda _q: self._internal_energy(_q, model, aux))(q)
        H = H * mask[:, None] * mask[None, :]
        diag_idx = jnp.arange(H.shape[0])
        return H.at[diag_idx, diag_idx].add(1.0 - mask)

    def _internal_energy(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        """Internal elastic energy only (no external forces)."""
        batch_qs = self.global_q_to_batch_q(q)
        return jnp.sum(
            jax.vmap(lambda t, q_loc, _aux: t.get_energy(q_loc, model, _aux))(
                self.triplets, batch_qs, aux
            )
        )

    @staticmethod
    def global_q_to_batch_q(q: jax.Array) -> jax.Array:
        """Extract overlapping 11-DOF windows for each triplet.
        q layout: [x0,y0,z0,θ0, x1,y1,z1,θ1, ..., xN-1,yN-1,zN-1]
        Each triplet: [n_i, θ_i, n_{i+1}, θ_{i+1}, n_{i+2}] = 11 DOFs
        """
        N_triplets = (q.shape[0] + 1) // 4 - 2
        starts = jnp.arange(N_triplets) * 4
        return jax.vmap(lambda s: jax.lax.dynamic_slice(q, (s,), (11,)))(starts)

    @staticmethod
    def get_mass(geom: Geometry, material: Material, l_ks: jax.Array) -> jax.Array:
        """Compute diagonal mass vector with proper Voronoi weights."""
        N = l_ks.shape[0] + 1
        n_dof = N * 4 - 1
        mass = jnp.zeros(n_dof)
        A = geom.axs if geom.axs else jnp.pi * geom.r0**2

        # Voronoi lengths per node: half of adjacent edges
        # Interior nodes: 0.5*(l_{i-1} + l_i)
        # Boundary nodes: 0.5*l_edge
        voronoi = jnp.zeros(N)
        voronoi = voronoi.at[0].add(0.5 * l_ks[0])
        voronoi = voronoi.at[-1].add(0.5 * l_ks[-1])
        # Interior contributions
        voronoi = voronoi.at[:-1].add(0.5 * l_ks)
        voronoi = voronoi.at[1:].add(0.5 * l_ks)

        dm_nodes = voronoi * A * material.density
        # Set mass for x, y, z DOFs of each node
        for d in range(3):
            mass = mass.at[d::4].set(dm_nodes)

        # Edge twist DOF mass (rotational inertia)
        factor = geom.jxs / geom.axs if (geom.jxs and geom.axs) else geom.r0**2 / 2.0
        edge_mass = l_ks * A * material.density * factor
        # Edge DOFs are at positions 3, 7, 11, ... in the global q vector
        edge_dof_indices = jnp.arange(N - 1) * 4 + 3
        mass = mass.at[edge_dof_indices].set(edge_mass)

        return mass

    @staticmethod
    def _get_batched_axes() -> Rod:
        bc_spec = BC(idx_b=None, xb_c=None, xb_m=0)  # type: ignore
        return Rod(triplets=None, F_ext=None, bc=bc_spec)  # type: ignore

    @eqx.filter_jit
    def solve(
        self,
        model: eqx.Module,
        lambdas: jax.Array,
        q0: jax.Array,
        aux: TripletState,
        iters: int = 10,
        ls_steps: int = 10,
        c1: float = 1e-4,
        max_dt: float = 1e-1,
    ) -> jax.Array:
        return solve(model, lambdas, q0, aux, self, iters, ls_steps, c1, max_dt)

    @eqx.filter_jit
    def batch_solve(
        self,
        model: eqx.Module,
        lambdas: jax.Array,
        q0: jax.Array,
        aux: TripletState,
        iters: int = 10,
        ls_steps: int = 10,
        c1: float = 1e-4,
        max_dt: float = 1e-1,
    ) -> jax.Array:
        rod_axes = self._get_batched_axes()
        v_solve = eqx.filter_vmap(
            solve, in_axes=(None, None, None, None, rod_axes, None, None, None, None)
        )
        return v_solve(model, lambdas, q0, aux, self, iters, ls_steps, c1, max_dt)

    @eqx.filter_jit
    def batch_F(
        self,
        qs: jax.Array,
        model: eqx.Module,
        aux: TripletState,
    ) -> jax.Array:
        rod_axes = self._get_batched_axes()

        def fn(r, qs_per_rod):
            return jax.vmap(lambda _q: r.get_F(_q, model, aux))(qs_per_rod)

        v_F = eqx.filter_vmap(
            lambda r, _q: fn(r, _q),
            in_axes=(rod_axes, 0),
        )
        return v_F(self, qs)
