"""Analytical strain gradient and Hessian w.r.t. DOFs.

Direct port of the reference implementation's analytical derivatives.
Works in the reference's 9-DOF position + 2-DOF theta space internally,
then maps to the JAX 11-DOF layout [n0(3), θe, n1(3), θf, n2(3)] at the end.

Reference DOF order: [n0x,n0y,n0z, n1x,n1y,n1z, n2x,n2y,n2z, θe, θf] (11 DOFs)
JAX DOF order:       [n0x,n0y,n0z, θe, n1x,n1y,n1z, θf, n2x,n2y,n2z] (11 DOFs)
"""
import jax
import jax.numpy as jnp
import numpy as np


# Mapping: for each JAX DOF index, which reference DOF index does it correspond to?
# jax: [n0x(0), n0y(1), n0z(2), θe(3), n1x(4), n1y(5), n1z(6), θf(7), n2x(8), n2y(9), n2z(10)]
# ref: [n0x(0), n0y(1), n0z(2), n1x(3), n1y(4), n1z(5), n2x(6), n2y(7), n2z(8), θe(9), θf(10)]
_JAX_TO_REF = np.array([0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8])


def _cross_mat(v):
    """Skew-symmetric cross-product matrix."""
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _grad_hess_stretch(n0, n1, n2, l_ke, l_kf):
    """Stretch gradient and Hessian in 9-DOF position space.

    Returns: grad (9,), hess (9, 9) for ε_avg = 0.5*(ε0 + ε1)
    """
    I3 = jnp.eye(3)
    inv_lk0 = 1.0 / l_ke
    inv_lk1 = 1.0 / l_kf

    # Edge 0
    e0 = n1 - n0
    len0 = jnp.linalg.norm(e0)
    t0 = e0 / jnp.maximum(len0, 1e-12)
    eps0 = len0 * inv_lk0 - 1.0
    dF0 = t0 * inv_lk0

    # Edge 1
    e1 = n2 - n1
    len1 = jnp.linalg.norm(e1)
    t1 = e1 / jnp.maximum(len1, 1e-12)
    eps1 = len1 * inv_lk1 - 1.0
    dF1 = t1 * inv_lk1

    # Gradient (9-DOF)
    grad = jnp.zeros(9)
    grad = grad.at[0:3].set(-0.5 * dF0)
    grad = grad.at[3:6].set(0.5 * dF0 - 0.5 * dF1)
    grad = grad.at[6:9].set(0.5 * dF1)

    # Hessian (9x9)
    # M = 2/l_k * [(1/l_k - 1/|e|)*I + e⊗e/|e|^3]
    M0 = 2.0 * inv_lk0 * ((inv_lk0 - 1.0 / jnp.maximum(len0, 1e-12)) * I3
                           + jnp.outer(e0, e0) / jnp.maximum(len0**3, 1e-36))
    M2_0 = 0.5 * (M0 - 2.0 * jnp.outer(dF0, dF0))
    M2_0 = jnp.where(jnp.abs(eps0) > 1e-14, M2_0 / eps0, jnp.zeros((3, 3)))

    M1 = 2.0 * inv_lk1 * ((inv_lk1 - 1.0 / jnp.maximum(len1, 1e-12)) * I3
                           + jnp.outer(e1, e1) / jnp.maximum(len1**3, 1e-36))
    M2_1 = 0.5 * (M1 - 2.0 * jnp.outer(dF1, dF1))
    M2_1 = jnp.where(jnp.abs(eps1) > 1e-14, M2_1 / eps1, jnp.zeros((3, 3)))

    hess = jnp.zeros((9, 9))
    hess = hess.at[0:3, 0:3].set(0.5 * M2_0)
    hess = hess.at[3:6, 3:6].set(0.5 * M2_0 + 0.5 * M2_1)
    hess = hess.at[6:9, 6:9].set(0.5 * M2_1)
    hess = hess.at[0:3, 3:6].set(-0.5 * M2_0)
    hess = hess.at[3:6, 0:3].set(-0.5 * M2_0)
    hess = hess.at[3:6, 6:9].set(-0.5 * M2_1)
    hess = hess.at[6:9, 3:6].set(-0.5 * M2_1)

    return grad, hess


def _grad_hess_curvature(n0, n1, n2, m1e, m2e, m1f, m2f):
    """Curvature gradient and Hessian in reference 11-DOF space.

    Returns: grad_k1 (11,), grad_k2 (11,), hess_k1 (11,11), hess_k2 (11,11)
    DOF order: [n0(0:3), n1(3:6), n2(6:9), θe(9), θf(10)]
    """
    I3 = jnp.eye(3)
    ee = n1 - n0
    ef = n2 - n1
    norm_e = jnp.linalg.norm(ee)
    norm_f = jnp.linalg.norm(ef)
    te = ee / jnp.maximum(norm_e, 1e-12)
    tf = ef / jnp.maximum(norm_f, 1e-12)

    chi = 1.0 + jnp.dot(te, tf)
    chi_inv = 1.0 / jnp.maximum(chi, 1e-12)
    kb = 2.0 * jnp.cross(te, tf) * chi_inv

    tt = (te + tf) * chi_inv
    td1 = (m1e + m1f) * chi_inv
    td2 = (m2e + m2f) * chi_inv

    k1 = 0.5 * jnp.dot(kb, m2e + m2f)
    k2 = -0.5 * jnp.dot(kb, m1e + m1f)

    inv_ne = 1.0 / jnp.maximum(norm_e, 1e-12)
    inv_nf = 1.0 / jnp.maximum(norm_f, 1e-12)
    ne2 = norm_e**2
    nf2 = norm_f**2

    # ── Gradient ──
    Dk1De = inv_ne * (-k1 * tt + jnp.cross(tf, td2))
    Dk1Df = inv_nf * (-k1 * tt - jnp.cross(te, td2))
    Dk2De = inv_ne * (-k2 * tt - jnp.cross(tf, td1))
    Dk2Df = inv_nf * (-k2 * tt + jnp.cross(te, td1))

    gk1 = jnp.zeros(11)
    gk1 = gk1.at[0:3].set(-Dk1De)
    gk1 = gk1.at[3:6].set(Dk1De - Dk1Df)
    gk1 = gk1.at[6:9].set(Dk1Df)
    gk1 = gk1.at[9].set(-0.5 * jnp.dot(kb, m1e))
    gk1 = gk1.at[10].set(-0.5 * jnp.dot(kb, m1f))

    gk2 = jnp.zeros(11)
    gk2 = gk2.at[0:3].set(-Dk2De)
    gk2 = gk2.at[3:6].set(Dk2De - Dk2Df)
    gk2 = gk2.at[6:9].set(Dk2Df)
    gk2 = gk2.at[9].set(-0.5 * jnp.dot(kb, m2e))
    gk2 = gk2.at[10].set(-0.5 * jnp.dot(kb, m2f))

    # ── Hessian ──
    tt_o_tt = jnp.outer(tt, tt)
    te_o_te = jnp.outer(te, te)
    tf_o_tf = jnp.outer(tf, tf)
    te_o_tf = jnp.outer(te, tf)

    # κ1
    tf_x_td2 = jnp.cross(tf, td2)
    te_x_td2 = jnp.cross(te, td2)

    D2k1De2 = (1/ne2)*(2*k1*tt_o_tt - jnp.outer(tf_x_td2,tt) - jnp.outer(tt,tf_x_td2)) \
        - (k1*chi_inv/ne2)*(I3-te_o_te) + (0.5/ne2)*jnp.outer(kb,m2e)
    D2k1Df2 = (1/nf2)*(2*k1*tt_o_tt + jnp.outer(te_x_td2,tt) + jnp.outer(tt,te_x_td2)) \
        - (k1*chi_inv/nf2)*(I3-tf_o_tf) + (0.5/nf2)*jnp.outer(kb,m2f)
    D2k1DeDf = (-k1*chi_inv/(norm_e*norm_f))*(I3+te_o_tf) \
        + (1/(norm_e*norm_f))*(2*k1*tt_o_tt - jnp.outer(tf_x_td2,tt) + jnp.outer(tt,te_x_td2) - _cross_mat(td2))

    # κ2
    tf_x_td1 = jnp.cross(tf, td1)
    te_x_td1 = jnp.cross(te, td1)

    D2k2De2 = (1/ne2)*(2*k2*tt_o_tt + jnp.outer(tf_x_td1,tt) + jnp.outer(tt,tf_x_td1)) \
        - (k2*chi_inv/ne2)*(I3-te_o_te) - (0.5/ne2)*jnp.outer(kb,m1e)
    D2k2Df2 = (1/nf2)*(2*k2*tt_o_tt - jnp.outer(te_x_td1,tt) - jnp.outer(tt,te_x_td1)) \
        - (k2*chi_inv/nf2)*(I3-tf_o_tf) - (0.5/nf2)*jnp.outer(kb,m1f)
    D2k2DeDf = (-k2*chi_inv/(norm_e*norm_f))*(I3+te_o_tf) \
        + (1/(norm_e*norm_f))*(2*k2*tt_o_tt + jnp.outer(tf_x_td1,tt) - jnp.outer(tt,te_x_td1) + _cross_mat(td1))

    # θ-θ
    D2k1Dte2 = -0.5*jnp.dot(kb,m2e)
    D2k1Dtf2 = -0.5*jnp.dot(kb,m2f)
    D2k2Dte2 = 0.5*jnp.dot(kb,m1e)
    D2k2Dtf2 = 0.5*jnp.dot(kb,m1f)

    # Position-θ cross terms (for κ1)
    D2k1DeDte = inv_ne*(0.5*jnp.dot(kb,m1e)*tt - chi_inv*jnp.cross(tf,m1e))
    D2k1DeDtf = inv_ne*(0.5*jnp.dot(kb,m1f)*tt - chi_inv*jnp.cross(tf,m1f))
    D2k1DfDte = inv_nf*(0.5*jnp.dot(kb,m1e)*tt + chi_inv*jnp.cross(te,m1e))
    D2k1DfDtf = inv_nf*(0.5*jnp.dot(kb,m1f)*tt + chi_inv*jnp.cross(te,m1f))

    # Position-θ cross terms (for κ2)
    D2k2DeDte = inv_ne*(0.5*jnp.dot(kb,m2e)*tt - chi_inv*jnp.cross(tf,m2e))
    D2k2DeDtf = inv_ne*(0.5*jnp.dot(kb,m2f)*tt - chi_inv*jnp.cross(tf,m2f))
    D2k2DfDte = inv_nf*(0.5*jnp.dot(kb,m2e)*tt + chi_inv*jnp.cross(te,m2e))
    D2k2DfDtf = inv_nf*(0.5*jnp.dot(kb,m2f)*tt + chi_inv*jnp.cross(te,m2f))

    # Assemble in REFERENCE 11-DOF order: pos(0:9), θe(9), θf(10)
    def _assemble_kappa_hess(De2, Df2, DeDf, Dte2, Dtf2,
                              DeDte, DeDtf, DfDte, DfDtf):
        H = jnp.zeros((11, 11))
        # Position-position (3x3 blocks: n0=0:3, n1=3:6, n2=6:9)
        H = H.at[0:3,0:3].set(De2)
        H = H.at[0:3,3:6].set(-De2 + DeDf)
        H = H.at[0:3,6:9].set(-DeDf)
        H = H.at[3:6,0:3].set(-De2 + DeDf.T)
        H = H.at[3:6,3:6].set(De2 - DeDf - DeDf.T + Df2)
        H = H.at[3:6,6:9].set(DeDf - Df2)
        H = H.at[6:9,0:3].set(-DeDf.T)
        H = H.at[6:9,3:6].set(DeDf.T - Df2)
        H = H.at[6:9,6:9].set(Df2)
        # θ-θ
        H = H.at[9,9].set(Dte2)
        H = H.at[10,10].set(Dtf2)
        # Position-θ (n0-θe, n1-θe, n2-θe, n0-θf, ...)
        H = H.at[0:3,9].set(-DeDte)
        H = H.at[3:6,9].set(DeDte - DfDte)
        H = H.at[6:9,9].set(DfDte)
        H = H.at[9,0:3].set(-DeDte)
        H = H.at[9,3:6].set(DeDte - DfDte)
        H = H.at[9,6:9].set(DfDte)
        H = H.at[0:3,10].set(-DeDtf)
        H = H.at[3:6,10].set(DeDtf - DfDtf)
        H = H.at[6:9,10].set(DfDtf)
        H = H.at[10,0:3].set(-DeDtf)
        H = H.at[10,3:6].set(DeDtf - DfDtf)
        H = H.at[10,6:9].set(DfDtf)
        return H

    hk1 = _assemble_kappa_hess(D2k1De2,D2k1Df2,D2k1DeDf,D2k1Dte2,D2k1Dtf2,
                                D2k1DeDte,D2k1DeDtf,D2k1DfDte,D2k1DfDtf)
    hk2 = _assemble_kappa_hess(D2k2De2,D2k2Df2,D2k2DeDf,D2k2Dte2,D2k2Dtf2,
                                D2k2DeDte,D2k2DeDtf,D2k2DfDte,D2k2DfDtf)

    return gk1, gk2, hk1, hk2


def _grad_hess_twist(n0, n1, n2):
    """Twist gradient and Hessian (Panetta's formulation) in reference 11-DOF.

    Returns: grad (11,), hess (11, 11)
    DOF order: [n0(0:3), n1(3:6), n2(6:9), θe(9), θf(10)]
    """
    ee = n1 - n0
    ef = n2 - n1
    norm_e = jnp.linalg.norm(ee)
    norm_f = jnp.linalg.norm(ef)
    te = ee / jnp.maximum(norm_e, 1e-12)
    tf = ef / jnp.maximum(norm_f, 1e-12)

    chi = 1.0 + jnp.dot(te, tf)
    chi_inv = 1.0 / jnp.maximum(chi, 1e-12)
    kb = 2.0 * jnp.cross(te, tf) * chi_inv
    tt = (te + tf) * chi_inv

    ne2 = norm_e**2
    nf2 = norm_f**2

    # Gradient
    g = jnp.zeros(11)
    g = g.at[0:3].set(-0.5 / norm_e * kb)
    g = g.at[6:9].set(0.5 / norm_f * kb)
    g = g.at[3:6].set(0.5 / norm_e * kb - 0.5 / norm_f * kb)
    g = g.at[9].set(-1.0)
    g = g.at[10].set(1.0)

    # Hessian (position-position only, Panetta)
    D2De2 = -0.5/ne2 * (jnp.outer(kb, te+tt) + 2*chi_inv*_cross_mat(tf))
    D2Df2 = -0.5/nf2 * (jnp.outer(kb, tf+tt) - 2*chi_inv*_cross_mat(te))
    D2DfDe = 0.5/(norm_e*norm_f) * (2*chi_inv*_cross_mat(te) - jnp.outer(kb,tt))
    D2DeDf = 0.5/(norm_e*norm_f) * (-2*chi_inv*_cross_mat(tf) - jnp.outer(kb,tt))

    H = jnp.zeros((11, 11))
    H = H.at[0:3,0:3].set(D2De2)
    H = H.at[0:3,3:6].set(-D2De2 + D2DfDe)     # ref uses DfDe here
    H = H.at[3:6,0:3].set(-D2De2 + D2DeDf)     # ref uses DeDf here
    H = H.at[3:6,3:6].set(D2De2 - (D2DeDf + D2DfDe) + D2Df2)
    H = H.at[0:3,6:9].set(-D2DfDe)             # ref uses -DfDe
    H = H.at[6:9,0:3].set(-D2DeDf)             # ref uses -DeDf
    H = H.at[6:9,3:6].set(D2DeDf - D2Df2)     # ref uses DeDf - Df2
    H = H.at[3:6,6:9].set(D2DfDe - D2Df2)     # ref uses DfDe - Df2
    H = H.at[6:9,6:9].set(D2Df2)
    # θ blocks are zero (Panetta)

    return g, H


def _reindex_ref_to_jax(grad_ref, hess_ref):
    """Map from reference 11-DOF [n0,n1,n2,θe,θf] to JAX [n0,θe,n1,θf,n2]."""
    idx = _JAX_TO_REF  # For each JAX position, pick from the ref position
    grad_jax = grad_ref[idx]
    hess_jax = hess_ref[idx][:, idx]
    return grad_jax, hess_jax


def grad_hess_strain_ref(q_local, m1e, m2e, m1f, m2f, l_ke, l_kf):
    """Compute analytical gradient and Hessian of 4 strain components.

    Matches the reference implementation exactly.

    Args:
        q_local: (11,) JAX DOFs [n0(3), θe, n1(3), θf, n2(3)]
        m1e, m2e, m1f, m2f: (3,) material directors (treated as constants)
        l_ke, l_kf: scalar reference edge lengths

    Returns:
        grad_strain: (11, 4) in JAX DOF order
        hess_strain: (11, 11, 4) in JAX DOF order
    """
    # Extract positions from JAX layout
    n0 = q_local[0:3]
    n1 = q_local[4:7]
    n2 = q_local[8:11]

    # Stretch (9-DOF position space → pad to ref 11-DOF)
    gs, hs = _grad_hess_stretch(n0, n1, n2, l_ke, l_kf)
    grad_eps_ref = jnp.zeros(11).at[0:9].set(gs)
    hess_eps_ref = jnp.zeros((11, 11)).at[0:9, 0:9].set(hs)

    # Curvature (ref 11-DOF)
    gk1, gk2, hk1, hk2 = _grad_hess_curvature(n0, n1, n2, m1e, m2e, m1f, m2f)

    # Twist (ref 11-DOF)
    gt, ht = _grad_hess_twist(n0, n1, n2)

    # Reindex all from reference to JAX DOF order
    ge_j, he_j = _reindex_ref_to_jax(grad_eps_ref, hess_eps_ref)
    gk1_j, hk1_j = _reindex_ref_to_jax(gk1, hk1)
    gk2_j, hk2_j = _reindex_ref_to_jax(gk2, hk2)
    gt_j, ht_j = _reindex_ref_to_jax(gt, ht)

    grad_strain = jnp.stack([ge_j, gk1_j, gk2_j, gt_j], axis=-1)   # (11, 4)
    hess_strain = jnp.stack([he_j, hk1_j, hk2_j, ht_j], axis=-1)   # (11, 11, 4)

    return grad_strain, hess_strain
