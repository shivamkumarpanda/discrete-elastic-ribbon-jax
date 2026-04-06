import jax
import jax.numpy as jnp
import equinox as eqx


def _compute_stiffness(geom, material):
    """Compute EA, EI1, EI2, GJ from geometry and material.

    Args:
        geom: Geometry dataclass with r0, axs, ixs1, ixs2, jxs
        material: Material dataclass with youngs_rod, poisson_rod

    Returns:
        (EA, EI1, EI2, GJ) tuple of floats
    """
    A = geom.axs if geom.axs else jnp.pi * geom.r0**2
    EA = material.youngs_rod * A

    if geom.ixs1 and geom.ixs2:
        EI1 = material.youngs_rod * geom.ixs1
        EI2 = material.youngs_rod * geom.ixs2
    else:
        I_circ = jnp.pi * geom.r0**4 / 4.0
        EI1 = material.youngs_rod * I_circ
        EI2 = material.youngs_rod * I_circ

    J = geom.jxs if geom.jxs else jnp.pi * geom.r0**4 / 2.0
    G = material.youngs_rod / (2.0 * (1.0 + material.poisson_rod))
    GJ = G * J

    return EA, EI1, EI2, GJ


class Kirchhoff(eqx.Module):
    """Kirchhoff elastic energy (pure quadratic).

    E = 0.5*EA*dl*eps_avg^2 + 0.5*(EI1/dl)*k1^2 + 0.5*(EI2/dl)*k2^2 + 0.5*(GJ/dl)*tau^2

    Input del_strain = [eps0, eps1, k1, k2, tau] (5-component).
    eps_avg = (eps0 + eps1) / 2 is used for stretch.
    """
    EA_dl: jax.Array   # EA * delta_l
    EI1_dl: jax.Array  # EI1 / delta_l
    EI2_dl: jax.Array  # EI2 / delta_l
    GJ_dl: jax.Array   # GJ / delta_l

    @classmethod
    def from_geometry(cls, l_k: jax.Array, geom, material) -> "Kirchhoff":
        EA, EI1, EI2, GJ = _compute_stiffness(geom, material)
        return cls(
            EA_dl=jnp.asarray(EA * l_k, dtype=jnp.float64),
            EI1_dl=jnp.asarray(EI1 / l_k, dtype=jnp.float64),
            EI2_dl=jnp.asarray(EI2 / l_k, dtype=jnp.float64),
            GJ_dl=jnp.asarray(GJ / l_k, dtype=jnp.float64),
        )

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        eps_avg = 0.5 * (del_strain[0] + del_strain[1])
        k1 = del_strain[2]
        k2 = del_strain[3]
        tau = del_strain[4]

        return (0.5 * self.EA_dl * eps_avg**2
                + 0.5 * self.EI1_dl * k1**2
                + 0.5 * self.EI2_dl * k2**2
                + 0.5 * self.GJ_dl * tau**2)


class Sano(eqx.Module):
    """Sano elastic energy: Kirchhoff + nonlinear twist-curvature coupling.

    E = E_kirchhoff + 0.5*(EI1/dl)*tau^4 / (1/(zeta/dl)^2 + k1^2)

    Input del_strain = [eps0, eps1, k1, k2, tau] (5-component).
    """
    EA_dl: jax.Array
    EI1_dl: jax.Array
    EI2_dl: jax.Array
    GJ_dl: jax.Array
    inv_zeta_dl_sq: jax.Array  # 1 / (zeta / delta_l)^2

    @classmethod
    def from_geometry(cls, l_k: jax.Array, geom, material,
                      zeta: float | None = None) -> "Sano":
        EA, EI1, EI2, GJ = _compute_stiffness(geom, material)

        if zeta is None:
            h = geom.r0
            A = geom.axs if geom.axs else jnp.pi * h**2
            w = A / h
            zeta = float(jnp.sqrt(w**4 / (120.0 * h**2)))

        zeta_dl = zeta / float(l_k)
        inv_zeta_dl_sq = 1.0 / (zeta_dl**2)

        return cls(
            EA_dl=jnp.asarray(EA * l_k, dtype=jnp.float64),
            EI1_dl=jnp.asarray(EI1 / l_k, dtype=jnp.float64),
            EI2_dl=jnp.asarray(EI2 / l_k, dtype=jnp.float64),
            GJ_dl=jnp.asarray(GJ / l_k, dtype=jnp.float64),
            inv_zeta_dl_sq=jnp.asarray(inv_zeta_dl_sq, dtype=jnp.float64),
        )

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        eps_avg = 0.5 * (del_strain[0] + del_strain[1])
        k1 = del_strain[2]
        tau = del_strain[4]

        E_kirchhoff = (0.5 * self.EA_dl * eps_avg**2
                       + 0.5 * self.EI1_dl * k1**2
                       + 0.5 * self.EI2_dl * del_strain[3]**2
                       + 0.5 * self.GJ_dl * tau**2)

        denom = self.inv_zeta_dl_sq + k1**2
        E_sano = 0.5 * self.EI1_dl * tau**4 / denom

        return E_kirchhoff + E_sano


class Sadowsky(eqx.Module):
    """Sadowsky elastic energy: Kirchhoff + regularized twist-curvature coupling.

    E = E_kirchhoff + 0.5*(EI1/dl)*tau^4 / (k1^2 + eps^2)

    Input del_strain = [eps0, eps1, k1, k2, tau] (5-component).
    """
    EA_dl: jax.Array
    EI1_dl: jax.Array
    EI2_dl: jax.Array
    GJ_dl: jax.Array
    eps_sq: jax.Array

    @classmethod
    def from_geometry(cls, l_k: jax.Array, geom, material,
                      eps: float = 1e-6) -> "Sadowsky":
        EA, EI1, EI2, GJ = _compute_stiffness(geom, material)
        return cls(
            EA_dl=jnp.asarray(EA * l_k, dtype=jnp.float64),
            EI1_dl=jnp.asarray(EI1 / l_k, dtype=jnp.float64),
            EI2_dl=jnp.asarray(EI2 / l_k, dtype=jnp.float64),
            GJ_dl=jnp.asarray(GJ / l_k, dtype=jnp.float64),
            eps_sq=jnp.asarray(eps**2, dtype=jnp.float64),
        )

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        eps_avg = 0.5 * (del_strain[0] + del_strain[1])
        k1 = del_strain[2]
        tau = del_strain[4]

        E_kirchhoff = (0.5 * self.EA_dl * eps_avg**2
                       + 0.5 * self.EI1_dl * k1**2
                       + 0.5 * self.EI2_dl * del_strain[3]**2
                       + 0.5 * self.GJ_dl * tau**2)

        E_sadowsky = 0.5 * self.EI1_dl * tau**4 / (k1**2 + self.eps_sq)

        return E_kirchhoff + E_sadowsky


class Audoly(eqx.Module):
    """Audoly elastic energy: Kirchhoff + geometry-dependent nonlinear term.

    E = E_kirchhoff + nonlinear_prefactor * (nu*k1^2 + tau^2)^2 * phi(v)
    where v = v_prefactor * k1, phi(v) is the Audoly correction function.

    Input del_strain = [eps0, eps1, k1, k2, tau] (5-component).
    """
    EA_dl: jax.Array
    EI1_dl: jax.Array
    EI2_dl: jax.Array
    GJ_dl: jax.Array
    nonlinear_prefactor: jax.Array  # (3*EI1*w^4) / (h^2 * dl^3)
    v_prefactor: jax.Array          # sqrt(12*(1-nu^2)) * w^2 / (h * dl)
    nu: jax.Array
    eps: jax.Array

    @classmethod
    def from_geometry(cls, l_k: jax.Array, geom, material,
                      nu: float | None = None, eps: float = 1e-6) -> "Audoly":
        EA, EI1, EI2, GJ = _compute_stiffness(geom, material)

        h = geom.r0
        A = geom.axs if geom.axs else jnp.pi * h**2
        w = A / h

        if nu is None:
            nu_val = float(jnp.clip(2.0 * EI1 / GJ - 1.0, -0.99, 0.5))
        else:
            nu_val = float(nu)

        l_k_f = float(l_k)
        nonlinear_pref = (3.0 * EI1 * w**4) / (h**2 * l_k_f**3)
        v_pref = float(jnp.sqrt(12.0 * (1.0 - nu_val**2))) * w**2 / (h * l_k_f)

        return cls(
            EA_dl=jnp.asarray(EA * l_k, dtype=jnp.float64),
            EI1_dl=jnp.asarray(EI1 / l_k, dtype=jnp.float64),
            EI2_dl=jnp.asarray(EI2 / l_k, dtype=jnp.float64),
            GJ_dl=jnp.asarray(GJ / l_k, dtype=jnp.float64),
            nonlinear_prefactor=jnp.asarray(nonlinear_pref, dtype=jnp.float64),
            v_prefactor=jnp.asarray(v_pref, dtype=jnp.float64),
            nu=jnp.asarray(nu_val, dtype=jnp.float64),
            eps=jnp.asarray(eps, dtype=jnp.float64),
        )

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        eps_avg = 0.5 * (del_strain[0] + del_strain[1])
        k1 = del_strain[2]
        tau = del_strain[4]

        E_kirchhoff = (0.5 * self.EA_dl * eps_avg**2
                       + 0.5 * self.EI1_dl * k1**2
                       + 0.5 * self.EI2_dl * del_strain[3]**2
                       + 0.5 * self.GJ_dl * tau**2)

        v = self.v_prefactor * k1
        phi = _compute_phi(v, self.eps)
        bracket = self.nu * k1**2 + tau**2
        E_audoly = self.nonlinear_prefactor * bracket**2 * phi

        return E_kirchhoff + E_audoly


class Wunderlich(eqx.Module):
    """Wunderlich elastic energy (nonlocal — requires eta_prime from neighbors).

    E = 0.5*EA*dl*eps_avg^2 + 0.5*(EI2/dl)*k2^2
      + 0.5*(EI1/dl) * [(k1*(1+eta^2))^2 / (W*eta')] * log[(1+W*eta'/2)/(1-W*eta'/2)]

    eta = tau / k1, eta' precomputed via finite differences across elements.
    When eta' ~ 0 or k1 ~ 0, falls back to 0.5*(EI1/dl)*k1^2.

    Input del_strain = [eps0, eps1, k1, k2, tau] (5-component).
    """
    EA_dl: jax.Array
    EI1_dl: jax.Array
    EI2_dl: jax.Array
    W: jax.Array
    eps: jax.Array

    @classmethod
    def from_geometry(cls, l_k: jax.Array, geom, material,
                      W: float | None = None, eps: float = 1e-12) -> "Wunderlich":
        EA, EI1, EI2, GJ = _compute_stiffness(geom, material)

        h = geom.r0
        if W is None:
            A = geom.axs if geom.axs else jnp.pi * h**2
            W = A / h

        return cls(
            EA_dl=jnp.asarray(EA * l_k, dtype=jnp.float64),
            EI1_dl=jnp.asarray(EI1 / l_k, dtype=jnp.float64),
            EI2_dl=jnp.asarray(EI2 / l_k, dtype=jnp.float64),
            W=jnp.asarray(W, dtype=jnp.float64),
            eps=jnp.asarray(eps, dtype=jnp.float64),
        )

    def __call__(self, del_strain: jax.Array,
                 eta_prime: jax.Array = jnp.array(0.0)) -> jax.Array:
        eps_avg = 0.5 * (del_strain[0] + del_strain[1])
        k1 = del_strain[2]
        k2 = del_strain[3]
        tau = del_strain[4]

        E_stretch = 0.5 * self.EA_dl * eps_avg**2
        E_bend2 = 0.5 * self.EI2_dl * k2**2

        eta = tau / (k1 + self.eps)
        W_eta_prime = self.W * eta_prime

        x_clamped = jnp.clip(W_eta_prime, -1.98, 1.98)
        log_ratio = (jnp.log(jnp.clip(1.0 + x_clamped / 2.0, 1e-12, None))
                     - jnp.log(jnp.clip(1.0 - x_clamped / 2.0, 1e-12, None)))

        bracket = k1 * (1.0 + eta**2)
        safe_W_ep = jnp.where(jnp.abs(W_eta_prime) > self.eps, W_eta_prime, 1.0)
        wunderlich_term = bracket**2 / safe_W_ep * log_ratio

        use_full = (jnp.abs(eta_prime) > self.eps) & (jnp.abs(k1) > self.eps)
        E_bend1 = jnp.where(
            use_full,
            0.5 * self.EI1_dl * wunderlich_term,
            0.5 * self.EI1_dl * k1**2,
        )

        return E_stretch + E_bend1 + E_bend2


def _compute_phi(v: jax.Array, eps: jax.Array) -> jax.Array:
    """Audoly's phi(v) function.

    phi(v) = (4/v^2) * [0.5 - (cosh(s) - cos(s)) / (s * (sinh(s) + sin(s)))]
    where s = sqrt(|v|/2).

    Returns 0 when |v| < 1e-4 for numerical stability.
    """
    abs_v = jnp.abs(v)
    eps_sq = eps**2
    s = jnp.sqrt(abs_v / 2.0 + eps_sq)

    numer = jnp.cosh(s) - jnp.cos(s)
    denom_inner = s * (jnp.sinh(s) + jnp.sin(s)) + eps
    frac = numer / denom_inner

    v_sq_reg = v**2 + eps_sq
    phi_full = (4.0 / v_sq_reg) * (0.5 - frac)

    return jnp.where(abs_v < 1e-4, 0.0, phi_full)
