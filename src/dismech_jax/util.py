import jax
import jax.numpy as jnp


def parallel_transport(u: jax.Array, t0: jax.Array, t1: jax.Array) -> jax.Array:
    """Transport vector u from tangent frame t0 to t1 using rotation formula."""
    b = jnp.cross(t0, t1)
    b_norm_sq = jnp.dot(b, b)
    d = jnp.dot(t0, t1)
    denom = 1.0 + d

    # When t0 ≈ t1 (denom ≈ 2), no rotation needed.
    # When t0 ≈ -t1 (denom ≈ 0), degenerate — return u as-is.
    safe_denom = jnp.where(denom > 1e-10, denom, 1.0)
    b_cross_u = jnp.cross(b, u)
    result = u + b_cross_u + jnp.cross(b, b_cross_u) / safe_denom

    # Use original u when nearly antiparallel
    return jnp.where(denom > 1e-10, result, u)


def signed_angle(u: jax.Array, v: jax.Array, n: jax.Array) -> jax.Array:
    """Signed angle from u to v around axis n."""
    w = jnp.cross(u, v)
    dot_uv = jnp.dot(u, v)
    signed_sin = jnp.dot(w, n)
    return jnp.arctan2(signed_sin, dot_uv)


def rotate_axis_angle(u: jax.Array, v: jax.Array, theta: jax.Array) -> jax.Array:
    """Rotate u around axis v by angle theta (Rodrigues formula)."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return c * u + s * jnp.cross(v, u) + jnp.dot(v, u) * (1 - c) * v


def material_frame(
    d1_old: jax.Array, t_old: jax.Array, t_new: jax.Array, theta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute material directors (m1, m2) by transporting d1 and rotating by theta.

    The bishop frame (d1, d2) is parallel-transported from the old tangent to
    the new tangent, then rotated by the twist angle theta to give material
    directors m1, m2. All operations are fully differentiable so that
    jax.grad and jax.hessian capture the complete dependence of strains on q.
    """
    d1_new = parallel_transport(d1_old, t_old, t_new)
    d2_new = jnp.cross(t_new, d1_new)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    m1 = c * d1_new + s * d2_new
    m2 = -s * d1_new + c * d2_new
    return m1, m2


def get_ref_twist(
    d1e: jax.Array, d1f: jax.Array, te: jax.Array, tf: jax.Array, r: jax.Array
) -> jax.Array:
    """Compute reference twist between edges e and f."""
    ut = parallel_transport(d1e, te, tf)
    ut = rotate_axis_angle(ut, tf, r)
    return r + signed_angle(ut, d1f, tf)
