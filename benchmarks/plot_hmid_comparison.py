#!/usr/bin/env python3
"""Plot |H_mid|/L vs ΔW/L bifurcation diagram for multiple node counts.

Two subplots: shear increasing (left) and shear decreasing (right).
Each N value is a separate curve.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# Colors for different N values
N_STYLES = {
    21:  {'color': '#FF0000', 'linestyle': '-',  'label': 'N=21'},
    45:  {'color': '#0066FF', 'linestyle': '--', 'label': 'N=45'},
    63:  {'color': '#00AA44', 'linestyle': '-.', 'label': 'N=63'},
    85:  {'color': '#8B00FF', 'linestyle': ':',  'label': 'N=85'},
    105: {'color': '#D2691E', 'linestyle': (0, (5, 2)), 'label': 'N=105'},
}


def load_jax_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_bifurcation_data(data):
    """Extract |H_mid|/L and ΔW/L from simulation data."""
    N = data['N']
    L = 0.1
    mid = N // 2
    qs = data['qs']
    times = data['times']

    # JAX DOF layout: [x0,y0,z0,θ0, x1,y1,z1,θ1, ...]
    z_mid = qs[:, 4 * mid + 2]
    z_mid_0 = z_mid[0]
    delta_h = z_mid - z_mid_0

    start_nodes = data['start_nodes']
    start_y = np.mean(qs[:, [4*n + 1 for n in start_nodes]], axis=1)
    delta_w = start_y - start_y[0]

    return np.abs(delta_h) / L, delta_w / L, times


def split_increasing_decreasing(dw_norm):
    """Split into shear-increasing and shear-decreasing masks."""
    ddw = np.diff(dw_norm)

    # Find shear start (first sustained increase)
    shear_start = 0
    for i in range(len(ddw) - 3):
        if np.all(ddw[i:i+3] > 1e-8):
            shear_start = i
            break

    # Find switch point (first decrease after increase)
    switch = len(dw_norm)
    for i in range(shear_start + 5, len(ddw)):
        if ddw[i] < -1e-8:
            switch = i
            break

    inc = np.zeros(len(dw_norm), dtype=bool)
    inc[shear_start:switch] = True
    dec = np.zeros(len(dw_norm), dtype=bool)
    dec[switch:] = True

    return inc, dec


def main():
    results_dir = Path('/data/shivam/Ribbon/bench_results')
    node_counts = [21, 45, 63, 85]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for N in node_counts:
        pkl_path = results_dir / f'jax_result_n{N}_wl12.pkl'
        if not pkl_path.exists():
            print(f"  Skip N={N}: {pkl_path} not found")
            continue

        data = load_jax_pkl(pkl_path)
        h_norm, dw_norm, times = extract_bifurcation_data(data)
        inc_mask, dec_mask = split_increasing_decreasing(dw_norm)

        style = N_STYLES.get(N, {'color': 'gray', 'linestyle': '-', 'label': f'N={N}'})

        # Left: shear increasing
        if np.any(inc_mask):
            ax1.plot(dw_norm[inc_mask], h_norm[inc_mask],
                     color=style['color'], linestyle=style['linestyle'],
                     linewidth=2, label=style['label'], alpha=0.9)

        # Right: shear decreasing
        if np.any(dec_mask):
            ax2.plot(dw_norm[dec_mask], h_norm[dec_mask],
                     color=style['color'], linestyle=style['linestyle'],
                     linewidth=2, label=style['label'], alpha=0.9)

        print(f"  N={N}: {np.sum(inc_mask)} inc, {np.sum(dec_mask)} dec points, "
              f"final_t={times[-1]:.2f}")

    for ax, title in [(ax1, r'Shear Increasing ($\Delta W/L \uparrow$)'),
                       (ax2, r'Shear Decreasing ($\Delta W/L \downarrow$)')]:
        ax.set_xlabel(r'$\Delta W / L$', fontsize=13)
        ax.set_ylabel(r'$|H_{mid}| / L$', fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.55)
        ax.set_ylim(-0.02, 0.32)

    fig.suptitle(r'JAX Bifurcation Diagram: W/L = 1/12, Sano Model', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = results_dir / 'jax_discretization_study_hmid.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
