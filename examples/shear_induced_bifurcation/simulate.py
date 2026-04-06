#!/usr/bin/env python3
"""Shear-induced bifurcation simulation using dismech_jax.

Port of the reference simulate.py from discrete-elastic-ribbon.

Usage:
    python simulate.py --config config.yaml --pkl-dir out/pkls --plot-dir out/plots
    python simulate.py --config config.yaml --pkl-dir out/pkls --plot-dir out/plots --nodes 21 --wbyl 1/12
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


def w_by_l_to_filename(ratio, n_nodes=None):
    frac = Fraction(ratio).limit_denominator(1000)
    base = f"1b{frac.denominator}"
    return f"{base}_n{n_nodes}.pkl" if n_nodes else f"{base}.pkl"


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
    bc_callback = create_bc_callback(cfg, start_nodes, mass)

    return rod, model, sim_params, mass, q0, aux, bc_callback, fixed_nodes, start_nodes


def create_bc_callback(cfg, start_nodes, mass_vec):
    """Create time-dependent boundary condition callback.

    Args:
        cfg: full YAML config dict
        start_nodes: indices of start (left) boundary nodes
        mass_vec: diagonal mass vector for gravity force computation
    """
    bc = cfg["boundary_condition"]
    u0 = float(bc["u0"])
    ramp_end = float(bc["gravity_ramp_end_time"])
    g_ramp = np.array(bc["gravity_during_ramp"], dtype=np.float64)
    g_after = np.array(bc["gravity_after_ramp"], dtype=np.float64)
    phases = bc["motion_phases"]

    def before_step(rod, q, u, aux, t, dt_step):
        # Gravity ramp: update F_ext z-component (applied from first step)
        g = g_ramp if t <= ramp_end else g_after
        F_ext = jnp.zeros_like(q)
        # Apply gravity to z DOFs (indices 2, 6, 10, ...)
        F_ext = F_ext.at[2::4].set(mass_vec[2::4] * g[2])
        rod = rod.with_F_ext(F_ext)

        # Move start nodes based on current phase
        for ph in phases:
            t_start = float(ph["start_time"])
            t_end = float(ph["end_time"])
            direction = int(ph["direction"])
            reverse = ph.get("reverse", False)
            sign = -1 if reverse else 1

            if t_start <= t < t_end:
                disp = u0 * dt_step * sign
                q, rod = dm.move_nodes(q, rod, start_nodes, disp, direction)
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

def extract_h_mid_delta_w(qs, n_nodes, start_thresh=0.01):
    """Extract H_mid (z of middle node) and delta_W (y of start nodes) from trajectory."""
    q0 = qs[0]
    N = n_nodes

    # Node positions from q
    def get_pos(q):
        return np.array([q[4 * i:4 * i + 3] for i in range(N)])

    pos0 = get_pos(q0)
    x_coords = pos0[:, 0]
    L_val = x_coords.max() - x_coords.min()
    mid_idx = np.argmin(np.abs(x_coords - L_val / 2.0))
    start_indices = np.where(x_coords <= start_thresh)[0]

    z_mid_init = pos0[mid_idx, 2]
    y_start_init = np.mean(pos0[start_indices, 1])

    delta_h = []
    delta_w = []
    for q in qs:
        pos = get_pos(q)
        delta_h.append(pos[mid_idx, 2] - z_mid_init)
        delta_w.append(np.mean(pos[start_indices, 1]) - y_start_init)

    return np.array(delta_h), np.array(delta_w), L_val


def plot_bifurcation(delta_h, delta_w, L_val, w_by_l, n_nodes, energy_model, plot_dir):
    """Plot |H_mid|/L vs delta_W/L bifurcation diagram."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dh_norm = np.abs(delta_h) / L_val
    dw_norm = delta_w / L_val

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dw_norm, dh_norm, linewidth=2)
    frac = Fraction(w_by_l).limit_denominator(1000)
    ax.set_xlabel("ΔW/L", fontsize=12)
    ax.set_ylabel("|H_mid/L|", fontsize=12)
    ax.set_title(f"W/L=1/{frac.denominator}, n={n_nodes}, {energy_model}", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe = f"1b{frac.denominator}_n{n_nodes}"
    out = plot_dir / f"{energy_model}_bifurcation_{safe}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shear-induced bifurcation (JAX)")
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

    for w_by_l in w_by_l_ratios:
        for n_nodes in n_nodes_list:
            pkl_path = pkl_dir / w_by_l_to_filename(w_by_l, n_nodes)
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
                delta_h, delta_w, L_val = extract_h_mid_delta_w(
                    result.qs, n_nodes,
                    float(cfg["boundary_condition"]["start_x_threshold"]),
                )
                plot_bifurcation(delta_h, delta_w, L_val, w_by_l, n_nodes,
                                 energy_model, args.plot_dir)

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
