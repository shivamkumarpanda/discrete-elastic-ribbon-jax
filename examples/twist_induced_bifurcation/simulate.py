#!/usr/bin/env python3
"""Twist-induced bifurcation simulation using dismech_jax.

Protocol: gravity ramp -> axial compression -> boundary twist.
Plots |H_mid|/L and |theta_mid|/L vs DeltaTheta/L.

Usage:
    python simulate.py --config config.yaml --pkl-dir out/pkls --plot-dir out/plots
    python simulate.py --config config.yaml --pkl-dir out/pkls --plot-dir out/plots \
        --nodes 45 --wbyl 1/12 --energy-model sano
"""
import argparse
import pickle
import time as time_mod
from fractions import Fraction
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dismech_jax as dm


# ─── Config loading ──────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_w_by_l(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def w_by_l_to_filename(ratio, n_nodes=None, energy_model=None):
    frac = Fraction(ratio).limit_denominator(1000)
    base = f"1b{frac.denominator}"
    if n_nodes:
        base += f"_n{n_nodes}"
    if energy_model:
        base += f"_{energy_model}"
    return f"{base}.pkl"


# ─── Simulation setup ────────────────────────────────────────────────────────

def setup_from_config(cfg, w_by_l, n_nodes):
    """Create rod, model, sim_params, and boundary condition callback from config."""
    g = cfg["geometry"]
    m = cfg["material"]
    L = float(g["L"])
    h = float(g["h"])
    w = w_by_l * L

    geom = dm.Geometry(
        length=L, r0=h,
        axs=w * h, jxs=w * h**3 / 3.0,
        ixs1=w * h**3 / 12.0, ixs2=h * w**3 / 12.0,
    )
    material = dm.Material(
        density=float(m["density"]),
        youngs_rod=float(m["youngs_rod"]),
        poisson_rod=float(m["poisson_rod"]),
    )

    # Create geometry
    script_dir = Path(__file__).parent
    geo_dir = script_dir / "initial_geometry"
    geo_path = dm.ensure_initial_geometry(geo_dir, n_nodes, L)
    nodes, edges = dm.load_geometry_txt(geo_path)

    # Create rod
    gravity_env = cfg["environment"]["gravity"]
    rod, q0, aux, mass = dm.create_rod_from_nodes(
        nodes, geom, material, gravity=gravity_env[2]
    )

    # Fix boundary nodes
    bc_cfg = cfg["boundary_condition"]
    start_thresh = float(bc_cfg["start_x_threshold"])
    end_thresh = float(bc_cfg["end_x_threshold"])
    node_positions = nodes
    start_nodes = np.where(node_positions[:, 0] <= start_thresh)[0]
    end_nodes = np.where(node_positions[:, 0] >= end_thresh)[0]
    fixed_nodes = np.union1d(start_nodes, end_nodes)
    rod, q0 = dm.fix_nodes(rod, q0, fixed_nodes)

    # Energy model
    dl = L / (n_nodes - 1)
    em_cfg = cfg.get("energy_model", {})
    energy_name = em_cfg.get("name", "sano").lower()

    if energy_name == "kirchhoff":
        model = dm.Kirchhoff.from_geometry(jnp.float64(dl), geom, material)
    elif energy_name == "sano":
        zeta = em_cfg.get("sano_zeta")
        if zeta is not None:
            zeta = float(zeta)
        model = dm.Sano.from_geometry(jnp.float64(dl), geom, material, zeta=zeta)
    elif energy_name == "sadowsky":
        model = dm.Sadowsky.from_geometry(jnp.float64(dl), geom, material)
    elif energy_name == "audoly":
        model = dm.Audoly.from_geometry(jnp.float64(dl), geom, material)
    elif energy_name == "wunderlich":
        W = em_cfg.get("wunderlich_W")
        if W is not None:
            W = float(W)
        model = dm.Wunderlich.from_geometry(jnp.float64(dl), geom, material, W=W)
    else:
        raise ValueError(f"Unknown energy model: {energy_name}")

    # Sim params
    s = cfg["simulation"]
    sim_params = dm.SimParams(
        dt=float(s["dt"]),
        total_time=float(s["total_time"]),
        tol=float(s["tol"]),
        ftol=float(s["ftol"]),
        dtol=float(s["dtol"]),
        max_iter=int(s["max_iter"]),
        log_step=int(s["log_step"]),
        static_sim=bool(s["static_sim"]),
    )

    # Before-step callback (boundary conditions)
    bc_callback = create_bc_callback(cfg, start_nodes, mass, n_nodes)

    return rod, model, sim_params, mass, q0, aux, bc_callback, fixed_nodes, start_nodes


def create_bc_callback(cfg, start_nodes, mass_vec, n_nodes):
    """Create time-dependent boundary condition callback for twist protocol.

    Phases:
      - Gravity ramp (settle flat)
      - Compression: move start nodes in +x
      - Twist: increment twist angle at all constrained start-boundary edges
    """
    bc = cfg["boundary_condition"]
    u0 = float(bc["u0"])
    w0 = float(bc["w0"])
    ramp_end = float(bc["gravity_ramp_end_time"])
    g_ramp = np.array(bc["gravity_during_ramp"], dtype=np.float64)
    g_after = np.array(bc["gravity_after_ramp"], dtype=np.float64)
    phases = bc["motion_phases"]

    # Find all constrained edges at the start boundary:
    # An edge is constrained if BOTH its endpoints are in the fixed node set.
    start_set = set(int(n) for n in start_nodes)
    twist_edges = np.array(
        [e for e in range(n_nodes - 1) if e in start_set and (e + 1) in start_set],
        dtype=np.int64,
    )
    print(f"  Twist edges (constrained at start boundary): {twist_edges.tolist()}")

    def before_step(rod, q, u, aux, t, dt_step):
        # Gravity ramp
        g = g_ramp if t <= ramp_end else g_after
        F_ext = jnp.zeros_like(q)
        F_ext = F_ext.at[2::4].set(mass_vec[2::4] * g[2])
        rod = rod.with_F_ext(F_ext)

        # Move/twist based on current phase
        for ph in phases:
            t_start = float(ph["start_time"])
            t_end = float(ph["end_time"])
            phase_type = ph.get("type", "compression")

            if t_start <= t < t_end:
                if phase_type == "compression":
                    direction = int(ph["direction"])
                    reverse = ph.get("reverse", False)
                    sign = -1 if reverse else 1
                    q, rod = dm.move_nodes(q, rod, start_nodes, u0 * dt_step * sign, direction)
                elif phase_type == "twist":
                    # Twist boundary edges: direction=3 maps to twist DOF q[4*edge+3]
                    q, rod = dm.move_nodes(q, rod, twist_edges, w0 * dt_step, 3)
                break

        return rod, q, u, aux

    return before_step


# ─── Run simulation ──────────────────────────────────────────────────────────

def run_simulation(cfg, w_by_l, n_nodes):
    """Run a single simulation and return results."""
    rod, model, sim_params, mass, q0, aux, bc_callback, fixed_nodes, start_nodes = \
        setup_from_config(cfg, w_by_l, n_nodes)

    stepper = dm.TimeStepper(rod, model, sim_params, mass, q0, aux)
    stepper.before_step = bc_callback

    # Adaptive dt
    adt = cfg.get("adaptive_dt", {})
    if adt.get("enabled", True):
        stepper.adaptive_dt = True
        stepper.max_dq_threshold = float(adt.get("max_dq_threshold", 0.1))
        stepper.dt_reduction_factor = float(adt.get("dt_reduction_factor", 0.5))
        base_dt = sim_params.dt
        stepper.min_dt = base_dt / float(adt.get("min_dt_ratio", 1e6))
        stepper.max_dt = base_dt * float(adt.get("max_dt_ratio", 2.0))
        stepper.max_dt_reductions = int(adt.get("max_dt_reductions", 40))

    # Tracking
    tr = cfg.get("tracking", {})
    if tr.get("track_forces", True):
        stepper.set_nodes_to_track_forces(fixed_nodes)
    if tr.get("track_elastic_energy", False):
        stepper.enable_elastic_energy_tracking()
    if tr.get("track_condition_number", False):
        stepper.enable_condition_number_tracking()

    print(f"  Starting simulation: W/L={w_by_l:.4f}, n={n_nodes}, model={cfg.get('energy_model',{}).get('name','sano')}")
    t0 = time_mod.time()
    result = stepper.simulate()
    elapsed = time_mod.time() - t0
    print(f"  Done in {elapsed:.1f}s, {len(result.qs)} logged steps")

    return result, fixed_nodes, start_nodes


# ─── Analysis & plotting ─────────────────────────────────────────────────────

def extract_twist_bifurcation_data(qs, times, n_nodes, L, w0, twist_start_time,
                                    start_thresh=0.01):
    """Extract H_mid, theta_mid, and DeltaTheta/L from trajectory.

    Returns:
        h_mid: z-displacement of midpoint node (array)
        theta_mid: twist angle at midpoint edge (array)
        delta_theta_over_L: cumulative applied twist / L (array)
        L_val: ribbon length
    """
    q0 = qs[0]
    N = n_nodes

    # Node positions from q
    def get_pos(q):
        return np.array([q[4 * i:4 * i + 3] for i in range(N)])

    pos0 = get_pos(q0)
    x_coords = pos0[:, 0]
    L_val = x_coords.max() - x_coords.min()
    mid_node = np.argmin(np.abs(x_coords - L_val / 2.0))

    # Midpoint edge for twist angle
    mid_edge = min(mid_node, N - 2)
    theta_mid_dof = 4 * mid_edge + 3

    z_mid_init = pos0[mid_node, 2]

    h_mid = []
    theta_mid = []
    delta_theta_over_L = []

    for i, q in enumerate(qs):
        t = times[i]

        # H_mid: z-displacement of midpoint
        h_mid.append(q[4 * mid_node + 2] - z_mid_init)

        # theta_mid: twist angle at midpoint edge
        theta_mid.append(q[theta_mid_dof])

        # DeltaTheta: cumulative twist applied at boundary
        if t >= twist_start_time:
            dt_val = w0 * (t - twist_start_time)
        else:
            dt_val = 0.0
        delta_theta_over_L.append(dt_val / L_val)

    return (np.array(h_mid), np.array(theta_mid),
            np.array(delta_theta_over_L), L_val)


def plot_twist_bifurcation(h_mid, theta_mid, delta_theta_over_L, L_val,
                            w_by_l, n_nodes, energy_model, plot_dir):
    """Plot |H_mid|/L and |theta_mid|/L vs DeltaTheta/L bifurcation diagrams."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    h_norm = np.abs(h_mid) / L_val
    theta_norm = np.abs(theta_mid) / L_val
    frac = Fraction(w_by_l).limit_denominator(1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: |H_mid|/L vs DeltaTheta/L
    ax1.plot(delta_theta_over_L, h_norm, linewidth=2, color='C0')
    ax1.set_xlabel(r"$\Delta\Theta / L$ (rad/m)", fontsize=12)
    ax1.set_ylabel(r"$|H_{\mathrm{mid}}| / L$", fontsize=12)
    ax1.set_title(f"W/L=1/{frac.denominator}, n={n_nodes}, {energy_model}", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: |theta_mid|/L vs DeltaTheta/L
    ax2.plot(delta_theta_over_L, theta_norm, linewidth=2, color='C1')
    ax2.set_xlabel(r"$\Delta\Theta / L$ (rad/m)", fontsize=12)
    ax2.set_ylabel(r"$|\theta_{\mathrm{mid}}| / L$ (rad/m)", fontsize=12)
    ax2.set_title(f"W/L=1/{frac.denominator}, n={n_nodes}, {energy_model}", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = f"1b{frac.denominator}_n{n_nodes}"
    out = plot_dir / f"{energy_model}_twist_bifurcation_{safe}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Twist-induced bifurcation (JAX)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pkl-dir", type=str, required=True)
    parser.add_argument("--plot-dir", type=str, required=True)
    parser.add_argument("--wbyl", type=str, nargs="+", default=None)
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--energy-model", type=str, default=None,
                        choices=["sano", "wunderlich", "kirchhoff", "sadowsky", "audoly"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    L = float(cfg["geometry"]["L"])

    # W/L ratios
    w_by_l_cfg = cfg["geometry"]["w_by_l_ratio"]
    if isinstance(w_by_l_cfg, list):
        w_by_l_ratios = [parse_w_by_l(x) for x in w_by_l_cfg]
    else:
        w_by_l_ratios = [parse_w_by_l(w_by_l_cfg)]
    if args.wbyl:
        w_by_l_ratios = [parse_w_by_l(s) for s in args.wbyl]

    # Node counts
    n_nodes_list = args.nodes if args.nodes else cfg.get("nodes", [21])
    if isinstance(n_nodes_list, int):
        n_nodes_list = [n_nodes_list]

    # Energy model override
    energy_model = (args.energy_model or cfg.get("energy_model", {}).get("name", "sano")).lower()
    if args.energy_model:
        if "energy_model" not in cfg:
            cfg["energy_model"] = {}
        cfg["energy_model"]["name"] = energy_model

    pkl_dir = Path(args.pkl_dir)
    pkl_dir.mkdir(parents=True, exist_ok=True)

    # Twist phase start time (for DeltaTheta calculation)
    bc_cfg = cfg["boundary_condition"]
    w0 = float(bc_cfg["w0"])
    twist_start_time = None
    for ph in bc_cfg["motion_phases"]:
        if ph.get("type") == "twist":
            twist_start_time = float(ph["start_time"])
            break
    if twist_start_time is None:
        raise ValueError("No twist phase found in config motion_phases")

    for w_by_l in w_by_l_ratios:
        for n_nodes in n_nodes_list:
            pkl_path = pkl_dir / w_by_l_to_filename(w_by_l, n_nodes, energy_model)
            print(f"\n=== W/L={w_by_l:.4f}, n={n_nodes} -> {pkl_path.name} ===")

            try:
                result, fixed_nodes, start_nodes = run_simulation(cfg, w_by_l, n_nodes)

                # Save
                save_data = {
                    "qs": np.stack(result.qs),
                    "times": np.array(result.times),
                    "fixed_nodes": fixed_nodes,
                    "elastic_energies": result.elastic_energies,
                    "forces": result.forces,
                }
                with open(pkl_path, "wb") as f:
                    pickle.dump(save_data, f)
                print(f"  Saved {pkl_path}")

                # Plot
                h_mid, theta_mid, dt_over_L, L_val = extract_twist_bifurcation_data(
                    result.qs, result.times, n_nodes, L, w0, twist_start_time,
                    float(bc_cfg["start_x_threshold"]),
                )
                plot_twist_bifurcation(h_mid, theta_mid, dt_over_L, L_val,
                                        w_by_l, n_nodes, energy_model, args.plot_dir)

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
