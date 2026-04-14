#!/usr/bin/env python3
"""
Build a triangular ribbon mesh from dismech_jax simulation pkl data for the
twist_induced_bifurcation example.

Differences from the reference dismech-python-general version:
- JAX DOF layout is interleaved: q = [x0,y0,z0,th0, x1,y1,z1,th1, ..., x_{n-1},y_{n-1},z_{n-1}]
  (length 4n-1, no theta on last node).
- The pkl does not store geom_params or material directors — width is inferred
  from the filename ("1b6" -> 1/6, "1b12" -> 1/12) times L (default 0.1 m), and
  m2 is recomputed from q via parallel transport.
- Per-frame rail frames (position, tangent, director, twist angle) are exported
  at start/end boundaries so the Three.js viewer can rotate & translate the
  guard rails to match the boundary condition (twist at the start boundary).

Usage:
  python ribbon_mesh_from_pkl.py path/to/file.pkl \
      --output path/to/ribbon_data.json \
      [--width W] [--length L] [--fps 30]
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ─── DOF layout helpers (JAX interleaved) ────────────────────────────────────

def n_nodes_from_dof(n_dof):
    # 4n - 1 == n_dof  =>  n = (n_dof + 1) / 4
    return (n_dof + 1) // 4


def positions_from_q(q, n_nodes):
    pos = np.empty((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        pos[i] = q[4 * i:4 * i + 3]
    return pos


def thetas_from_q(q, n_nodes):
    n_edges = n_nodes - 1
    th = np.empty(n_edges, dtype=np.float64)
    for i in range(n_edges):
        th[i] = q[4 * i + 3]
    return th


# ─── Material director reconstruction ────────────────────────────────────────

def _parallel_transport_single(u, t_start, t_end):
    b = np.cross(t_start, t_end)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        return u.copy()
    axis = b / b_norm
    angle = np.arccos(np.clip(np.dot(t_start, t_end), -1.0, 1.0))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return u * cos_a + np.cross(axis, u) * sin_a + axis * np.dot(axis, u) * (1 - cos_a)


def compute_frames_from_q(q, n_nodes):
    """Return (positions, tangents, a1, a2, m1, m2, theta) for each edge.
    m1 = cos*a1 + sin*a2 (surface-normal director, perpendicular to ribbon plane)
    m2 = -sin*a1 + cos*a2 (in-plane width director)
    """
    positions = positions_from_q(q, n_nodes)
    theta = thetas_from_q(q, n_nodes)
    n_edges = n_nodes - 1

    tangents = np.zeros((n_edges, 3))
    for i in range(n_edges):
        d = positions[i + 1] - positions[i]
        n = np.linalg.norm(d)
        tangents[i] = d / (n + 1e-12)

    # Parallel-transport the reference frame from the END edge backward.
    # The end boundary is fixed (tangent = +x, theta = 0), so anchoring there
    # is physically correct — the start side will then carry the full twist
    # AND any holonomy from the buckled shape, which is what we want to see.
    a1 = np.zeros((n_edges, 3))
    a2 = np.zeros((n_edges, 3))
    last = n_edges - 1
    a1_init = np.cross(tangents[last], np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(a1_init) < 1e-8:
        a1_init = np.cross(tangents[last], np.array([0.0, 0.0, -1.0]))
    a1[last] = a1_init / np.linalg.norm(a1_init)
    a2[last] = np.cross(tangents[last], a1[last])
    for i in range(last - 1, -1, -1):
        a1[i] = _parallel_transport_single(a1[i + 1], tangents[i + 1], tangents[i])
        a1[i] -= np.dot(a1[i], tangents[i]) * tangents[i]
        nrm = np.linalg.norm(a1[i])
        a1[i] /= nrm if nrm > 1e-12 else 1.0
        a2[i] = np.cross(tangents[i], a1[i])

    ct = np.cos(theta)[:, None]
    st = np.sin(theta)[:, None]
    m1 = ct * a1 + st * a2  # normal to ribbon surface
    m2 = -st * a1 + ct * a2  # width-direction in-plane
    # Normalize m2
    norms = np.linalg.norm(m2, axis=1, keepdims=True)
    m2 = m2 / np.where(norms < 1e-12, 1.0, norms)
    return positions, tangents, a1, a2, m1, m2, theta


# ─── Mesh construction (matches reference) ───────────────────────────────────

def build_centerline_points_and_directors(nodes, m_width_per_edge, width):
    """m_width_per_edge is the in-plane width director m2 per edge."""
    n_nodes = nodes.shape[0]
    n_edges = n_nodes - 1
    n_pts = 2 * n_nodes - 1
    w2 = width / 2.0

    centers = np.zeros((n_pts, 3))
    directors = np.zeros((n_pts, 3))

    for i in range(n_nodes):
        centers[2 * i] = nodes[i]
        if i == 0:
            d = m_width_per_edge[0]
        elif i == n_nodes - 1:
            d = m_width_per_edge[n_edges - 1]
        else:
            d = m_width_per_edge[i - 1] + m_width_per_edge[i]
            nrm = np.linalg.norm(d)
            d = d / (nrm if nrm > 1e-12 else 1.0)
        directors[2 * i] = d

    for i in range(n_edges):
        centers[2 * i + 1] = (nodes[i] + nodes[i + 1]) * 0.5
        directors[2 * i + 1] = m_width_per_edge[i]

    left_boundary = centers - w2 * directors
    right_boundary = centers + w2 * directors
    return centers, directors, left_boundary, right_boundary


def triangulate_strip(left_i, right_i, left_ip1, right_ip1, n_width_segments):
    triangles = []
    for k in range(n_width_segments):
        t0 = k / n_width_segments
        t1 = (k + 1) / n_width_segments
        p00 = (1 - t0) * left_i + t0 * right_i
        p10 = (1 - t1) * left_i + t1 * right_i
        p01 = (1 - t0) * left_ip1 + t0 * right_ip1
        p11 = (1 - t1) * left_ip1 + t1 * right_ip1
        triangles.append((p00, p10, p11))
        triangles.append((p00, p11, p01))
    return triangles


def build_frame_mesh(nodes, m_width_per_edge, width):
    centers, directors, left_boundary, right_boundary = \
        build_centerline_points_and_directors(nodes, m_width_per_edge, width)
    n_pts = centers.shape[0]
    vertex_list = []
    face_list = []
    for i in range(n_pts - 1):
        d = np.linalg.norm(centers[i + 1] - centers[i])
        n_width = max(1, int(round(width / max(d, 1e-12))))
        n_width = min(n_width, 32)
        tris = triangulate_strip(
            left_boundary[i], right_boundary[i],
            left_boundary[i + 1], right_boundary[i + 1], n_width
        )
        for (a, b, c) in tris:
            base = len(vertex_list)
            vertex_list.extend([a, b, c])
            face_list.append((base, base + 1, base + 2))
    return (np.array(vertex_list, dtype=np.float64),
            np.array(face_list, dtype=np.int64))


# ─── Axis remapping: sim (x along length, z up) -> viewer (x along length, y up) ──

def rotate_to_viewer(v):
    """Sim (x,y,z) -> viewer (x, z, y) so sim's x-z plane is viewer's x-y plane."""
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = v[..., 2]
    out[..., 2] = v[..., 1]
    return out


# ─── Width from filename ─────────────────────────────────────────────────────

def width_from_pkl_name(pkl_path, L):
    """Parse '1bK' in filename -> ribbon width = L / K."""
    m = re.search(r"1b(\d+)", Path(pkl_path).stem)
    if not m:
        return None
    return L / float(m.group(1))


# ─── Rail frame extraction ───────────────────────────────────────────────────

def rail_info(positions, tangents, theta, start_nodes, end_nodes,
              a1_ref, a2_ref):
    """Compute rail origin/tangent/director at each boundary directly from
    the boundary edges' theta and the INITIAL reference directors
    (a1_ref, a2_ref, per edge at t=0). This bypasses spatially-parallel-
    transported frames, which pick up shape-dependent holonomy that
    wobbles as the ribbon breathes — the clamped boundary's physical
    reference frame is fixed, only theta moves.
    """
    def boundary_frame(node_idx_set):
        node_idx = np.asarray(sorted(node_idx_set), dtype=int)
        origin = positions[node_idx].mean(axis=0)
        edge_idx = [e for e in range(len(tangents))
                    if e in node_idx_set and (e + 1) in node_idx_set]
        if len(edge_idx) == 0:
            edge_idx = [min(node_idx[0], len(tangents) - 1)]
        edge_idx = np.asarray(edge_idx, dtype=int)
        tan = tangents[edge_idx].mean(axis=0)
        tan = tan / (np.linalg.norm(tan) + 1e-12)
        th = float(theta[edge_idx].mean())
        a1 = a1_ref[edge_idx].mean(axis=0)
        a1 -= np.dot(a1, tan) * tan
        a1 /= (np.linalg.norm(a1) + 1e-12)
        a2 = np.cross(tan, a1)
        width_dir = -np.sin(th) * a1 + np.cos(th) * a2
        normal = np.cross(width_dir, tan)
        return origin, tan, width_dir, normal

    start_set = set(int(n) for n in start_nodes)
    end_set = set(int(n) for n in end_nodes)
    return {
        "start": boundary_frame(start_set),
        "end": boundary_frame(end_set),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run(pkl_path, output_path, width_override=None, L_override=None,
        every=1, max_frames=None, fps=None,
        start_x_threshold=0.01, end_x_threshold=0.09):
    pkl_path = Path(pkl_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_pkl(pkl_path)
    qs = np.asarray(data["qs"])
    times = np.asarray(data["times"], dtype=np.float64)
    n_dof = qs.shape[1]
    n_nodes = n_nodes_from_dof(n_dof)

    L = float(L_override) if L_override is not None else 0.1
    width = (float(width_override) if width_override is not None
             else width_from_pkl_name(pkl_path, L))
    if width is None:
        raise ValueError("Unable to infer ribbon width; pass --width.")

    # Use the INITIAL positions to identify start/end boundary nodes.
    pos0 = positions_from_q(qs[0], n_nodes)
    start_nodes = np.where(pos0[:, 0] <= start_x_threshold)[0].tolist()
    end_nodes = np.where(pos0[:, 0] >= end_x_threshold)[0].tolist()

    n_frames_full = qs.shape[0]
    total_sim_time = float(times[-1]) if len(times) else 0.0

    # Frame selection
    if fps is not None and fps > 0:
        dt_sim = 1.0 / fps
        snapshot_times = np.arange(0.0, total_sim_time + 1e-12, dt_sim)
        indices = []
        for t in snapshot_times:
            i = np.searchsorted(times, t, side="right") - 1
            i = max(0, min(i, n_frames_full - 1))
            indices.append(i)
        indices = sorted(set(indices))
    else:
        e = max(1, int(every))
        indices = list(range(0, n_frames_full, e))
    if max_frames is not None and len(indices) > max_frames:
        indices = indices[: max_frames]

    frame_times = times[indices]
    frames_out = []

    # Initial reference directors per edge (from the flat ribbon at t=0).
    # Because boundary edges are clamped, their tangent never changes, so
    # (a1_ref, a2_ref) at those edges stay valid for the full trajectory.
    _, _, a1_ref, a2_ref, _, _, _ = compute_frames_from_q(qs[0], n_nodes)

    for ti in tqdm(indices, desc="Converting to mesh", unit="frame"):
        q = qs[ti]
        pos, tan, a1, a2, m1, m2, theta = compute_frames_from_q(q, n_nodes)
        vertices, faces = build_frame_mesh(pos, m2, width)
        vertices = rotate_to_viewer(vertices)

        rails = rail_info(pos, tan, theta, start_nodes, end_nodes, a1_ref, a2_ref)
        rails_out = {}
        for side, (origin, t_vec, w_vec, n_vec) in rails.items():
            rails_out[side] = {
                "origin": rotate_to_viewer(origin).tolist(),
                "tangent": rotate_to_viewer(t_vec).tolist(),
                "width_dir": rotate_to_viewer(w_vec).tolist(),
                "normal": rotate_to_viewer(n_vec).tolist(),
            }
        # Twist angle at start/end boundary (mean theta over boundary edges).
        start_set = set(int(n) for n in start_nodes)
        end_set = set(int(n) for n in end_nodes)
        s_edges = [e for e in range(n_nodes - 1) if e in start_set and (e + 1) in start_set]
        e_edges = [e for e in range(n_nodes - 1) if e in end_set and (e + 1) in end_set]
        rails_out["start"]["theta"] = float(np.mean(theta[s_edges])) if s_edges else 0.0
        rails_out["end"]["theta"] = float(np.mean(theta[e_edges])) if e_edges else 0.0

        frames_out.append({
            "vertices": vertices.flatten().tolist(),
            "faces": faces.flatten().tolist(),
            "simTime": float(times[ti]),
            "rails": rails_out,
        })

    export_data = {
        "frames": frames_out,
        "frame_times": frame_times.tolist(),
        "n_frames": len(frames_out),
        "width": width,
        "length": L,
        "n_nodes": n_nodes,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, separators=(",", ":"))
    js_path = output_path.with_suffix(".js")
    with open(js_path, "w") as f:
        f.write("var ribbonData = ")
        json.dump(export_data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"Wrote {output_path} and {js_path}: {len(frames_out)} frames, width={width}, L={L}, n_nodes={n_nodes}")


def main():
    p = argparse.ArgumentParser(description="Build ribbon mesh from dismech_jax pkl (twist/shear+twist)")
    p.add_argument("pkl", type=str)
    p.add_argument("--output", "-o", type=str, required=True)
    p.add_argument("--width", "-w", type=float, default=None)
    p.add_argument("--length", "-L", type=float, default=0.1)
    p.add_argument("--fps", "-f", type=float, default=None)
    p.add_argument("--every", "-e", type=int, default=1)
    p.add_argument("--max-frames", "-m", type=int, default=None)
    p.add_argument("--start-x-threshold", type=float, default=0.01)
    p.add_argument("--end-x-threshold", type=float, default=0.09)
    args = p.parse_args()
    run(args.pkl, args.output, args.width, args.length,
        every=args.every, max_frames=args.max_frames, fps=args.fps,
        start_x_threshold=args.start_x_threshold, end_x_threshold=args.end_x_threshold)


if __name__ == "__main__":
    main()
