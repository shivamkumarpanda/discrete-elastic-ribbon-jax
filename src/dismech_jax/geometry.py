"""Geometry I/O: load from text files, generate initial rod geometry."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from .systems.rod import Rod, BC
from .states import TripletState


def load_geometry_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load node positions and edge connectivity from a .txt file.

    Format:
        *Nodes
        x,y,z
        ...
        *Edges
        i,j  (1-indexed)
        ...

    Returns:
        nodes: (N, 3) array of positions
        edges: (M, 2) array of 0-indexed edge connectivity
    """
    path = Path(path)
    nodes = []
    edges = []
    section = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*Nodes"):
                section = "nodes"
                continue
            elif line.startswith("*Edges"):
                section = "edges"
                continue

            if section == "nodes":
                parts = line.split(",")
                nodes.append([float(p) for p in parts])
            elif section == "edges":
                parts = line.split(",")
                edges.append([int(p) - 1 for p in parts])  # 1-indexed -> 0-indexed

    return np.array(nodes, dtype=np.float64), np.array(edges, dtype=np.int64)


def ensure_initial_geometry(geometry_dir: str | Path, n_nodes: int, L: float) -> Path:
    """Create initial geometry file for a straight rod along x-axis.

    Returns the path to the geometry file.
    """
    geometry_dir = Path(geometry_dir)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    path = geometry_dir / f"horizontal_rod_n{n_nodes}.txt"
    if path.exists():
        return path

    dl = L / (n_nodes - 1)
    lines = ["*Nodes"]
    for i in range(n_nodes):
        x = i * dl
        lines.append(f"{x},0,0")
    lines.append("*Edges")
    for i in range(1, n_nodes):
        lines.append(f"{i},{i + 1}")

    path.write_text("\n".join(lines) + "\n")
    return path


def create_rod_from_nodes(
    nodes: np.ndarray,
    geom,
    material,
    gravity: float = -9.81,
):
    """Create a Rod, initial q, and aux from node positions.

    Args:
        nodes: (N, 3) array of node positions
        geom: Geometry dataclass
        material: Material dataclass
        gravity: gravitational acceleration (applied in z)

    Returns:
        rod, q0, aux, mass
    """
    N = nodes.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 nodes.")

    # Build q0: [x0,y0,z0, θ0, x1,y1,z1, θ1, ..., xN-1,yN-1,zN-1]
    n_dof = 4 * N - 1
    q0 = jnp.zeros(n_dof)
    for i in range(N):
        q0 = q0.at[4 * i].set(nodes[i, 0])
        q0 = q0.at[4 * i + 1].set(nodes[i, 1])
        q0 = q0.at[4 * i + 2].set(nodes[i, 2])

    # Edge lengths
    edges = nodes[1:] - nodes[:-1]
    l_ks = jnp.array(np.linalg.norm(edges, axis=1))

    # Tangent vectors
    tangents = edges / np.linalg.norm(edges, axis=1, keepdims=True)

    # Initial material directors: d1 perpendicular to tangent
    # Convention: d1 points in z (matching reference dismech), d2 = t × d1
    N_edges = N - 1
    d1s = np.zeros((N_edges, 3))
    for i in range(N_edges):
        t = tangents[i]
        # Choose d1 perpendicular to t, preferring z-direction
        if abs(t[2]) < 0.9:
            perp = np.array([0.0, 0.0, 1.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        d1 = perp - np.dot(perp, t) * t
        d1 = d1 / np.linalg.norm(d1)
        d1s[i] = d1

    # Build triplet auxiliary states
    N_triplets = N - 2
    batch_q = Rod.global_q_to_batch_q(q0)

    ts_list = []
    d1_list = []
    betas = []
    for k in range(N_triplets):
        te = tangents[k]
        tf = tangents[k + 1]
        d1e = d1s[k]
        d1f = d1s[k + 1]
        ts_list.append(jnp.array([te, tf]))
        d1_list.append(jnp.array([d1e, d1f]))
        betas.append(0.0)

    ts_arr = jnp.stack(ts_list)
    d1_arr = jnp.stack(d1_list)
    betas_arr = jnp.array(betas)
    batch_aux = jax.vmap(TripletState)(ts_arr, d1_arr, betas_arr)

    from .stencils import Triplet  # local import to avoid circular
    batch_l_ks = jax.vmap(lambda i: jax.lax.dynamic_slice(l_ks, (i,), (2,)))(
        jnp.arange(N_triplets)
    )
    triplets = jax.vmap(lambda q_loc, a, lk: Triplet.init(q_loc, a, l_k=lk))(
        batch_q, batch_aux, batch_l_ks
    )

    # Mass
    mass = Rod.get_mass(geom, material, l_ks)

    # External forces (gravity)
    F_ext = jnp.zeros(n_dof)
    F_ext = F_ext.at[2::4].set(mass[2::4] * gravity)

    rod = Rod(
        triplets=triplets,
        F_ext=F_ext,
        bc=BC(jnp.empty(0, dtype=jnp.int32), jnp.empty(0), jnp.empty(0)),
    )

    return rod, q0, batch_aux, mass


def fix_nodes(rod: Rod, q0: jax.Array, node_indices: np.ndarray) -> tuple[Rod, jax.Array]:
    """Constrain specified nodes (3 position DOFs + twist angle per node).

    Returns updated rod with BC applied, and q0 with BC values set.
    """
    n_dof = q0.shape[0]
    N = (n_dof + 1) // 4  # number of nodes
    node_set = set(int(n) for n in node_indices)
    dof_indices = []
    for n in node_indices:
        n = int(n)
        dof_indices.extend([4 * n, 4 * n + 1, 4 * n + 2])
        # Fix theta (twist) only when BOTH endpoints of the edge are fixed
        # (matching reference convention: theta[k] is fixed iff node k AND node k+1 are both fixed)
        if n < N - 1 and (n + 1) in node_set:
            dof_indices.append(4 * n + 3)
    dof_indices = jnp.array(dof_indices, dtype=jnp.int32)

    # BC: q[idx] = 0 * lambda + q0[idx] (constant position)
    xb_c = q0[dof_indices]
    xb_m = jnp.zeros_like(xb_c)

    bc = BC(idx_b=dof_indices, xb_m=xb_m, xb_c=xb_c)
    return rod.with_bc(bc), q0


def move_nodes(
    q: jax.Array,
    rod: Rod,
    node_indices: np.ndarray,
    displacement: float,
    direction: int,
) -> tuple[jax.Array, Rod]:
    """Move specified nodes by a displacement in the given direction.

    Updates both q and the BC constants in the rod.
    Preserves BC idx_b identity to avoid JIT recompilation.

    Args:
        q: current DOF vector
        rod: current rod with BCs
        node_indices: which nodes to move
        displacement: amount to move
        direction: 0=x, 1=y, 2=z

    Returns:
        updated q, updated rod
    """
    # Vectorized displacement
    dof_indices = jnp.array([4 * n + direction for n in node_indices], dtype=jnp.int32)
    q = q.at[dof_indices].add(displacement)

    # Update BC constants to match new positions (preserve idx_b identity)
    if rod.bc.idx_b.shape[0] > 0:
        new_xb_c = q[rod.bc.idx_b]
        new_bc = BC(idx_b=rod.bc.idx_b, xb_m=rod.bc.xb_m, xb_c=new_xb_c)
        rod = eqx.tree_at(lambda r: r.bc, rod, new_bc)

    return q, rod
