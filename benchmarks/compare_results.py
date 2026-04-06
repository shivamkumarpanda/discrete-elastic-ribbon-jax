#!/usr/bin/env python3
"""Compare JAX vs reference simulation results: speed and accuracy.

Expects pickle files from run_reference.py and run_jax.py.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_result(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_node_positions(q, N):
    """Extract (N, 3) node positions from DOF vector."""
    return np.array([q[4*i:4*i+3] for i in range(N)])


def extract_bifurcation_metrics(qs, N, start_thresh=0.01):
    """Extract H_mid/L and delta_W/L from trajectory."""
    q0 = qs[0]
    pos0 = get_node_positions(q0, N)
    x_coords = pos0[:, 0]
    L_val = x_coords.max() - x_coords.min()
    mid_idx = np.argmin(np.abs(x_coords - L_val / 2.0))
    start_idx = np.where(x_coords <= start_thresh)[0]

    z_mid_init = pos0[mid_idx, 2]
    y_start_init = np.mean(pos0[start_idx, 1])

    delta_h = np.array([get_node_positions(q, N)[mid_idx, 2] - z_mid_init for q in qs])
    delta_w = np.array([np.mean(get_node_positions(q, N)[start_idx, 1]) - y_start_init for q in qs])

    return delta_h / L_val, delta_w / L_val, L_val


def compare(ref_path, jax_path, out_dir='/data/shivam/Ribbon/bench_results/comparison'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_result(ref_path)
    jax = load_result(jax_path)

    N = ref['N']
    print(f"\n{'='*70}")
    print(f"COMPARISON: N={N}, W/L=1/{int(1/ref['w_by_l'])}, Sano model")
    print(f"{'='*70}")

    # ── Speed ──
    print(f"\n--- Speed ---")
    ref_steps = ref['n_steps']
    jax_steps = jax['n_steps']
    ref_time = ref['elapsed']
    jax_time = jax['elapsed']
    print(f"  Reference (NumPy/PyTorch): {ref_time:8.1f}s | {ref_steps:5d} steps | {ref_steps/ref_time:6.1f} steps/s | {ref_time/ref_steps*1000:5.1f} ms/step")
    print(f"  JAX:                       {jax_time:8.1f}s | {jax_steps:5d} steps | {jax_steps/jax_time:6.1f} steps/s | {jax_time/jax_steps*1000:5.1f} ms/step")
    speedup = ref_time / jax_time if jax_time > 0 else float('inf')
    print(f"  Speedup: {speedup:.2f}x")

    # ── Trajectory comparison ──
    ref_qs = ref['qs']
    jax_qs = jax['qs']

    # The two might have different number of logged steps due to adaptive dt
    # Compare at common timestep count (min of the two)
    n_common = min(len(ref_qs), len(jax_qs))
    print(f"\n--- Trajectory (first {n_common} logged steps) ---")

    # Per-step q difference
    # Reference q layout: [x0,y0,z0,...,xN-1,yN-1,zN-1, θ0,θ1,...,θN-2] (3N + N-1 DOFs)
    # JAX q layout: [x0,y0,z0,θ0, x1,y1,z1,θ1, ..., xN-1,yN-1,zN-1] (4N - 1 DOFs)
    def ref_node_pos(q, N):
        """Extract node positions from reference DOF layout."""
        return q[:3*N].reshape(N, 3)

    def jax_node_pos(q, N):
        """Extract node positions from JAX DOF layout."""
        return np.array([q[4*i:4*i+3] for i in range(N)])

    ref_dof_len = ref_qs[0].shape[0]
    jax_dof_len = jax_qs[0].shape[0]
    ref_is_separated = (ref_dof_len == 3*N + (N-1))  # separated layout
    jax_is_interleaved = (jax_dof_len == 4*N - 1)     # interleaved layout
    print(f"  DOF layouts: ref={ref_dof_len} ({'separated' if ref_is_separated else 'interleaved'}), "
          f"jax={jax_dof_len} ({'interleaved' if jax_is_interleaved else 'separated'})")

    q_diffs = []
    for i in range(n_common):
        ref_q = ref_qs[i]
        jax_q = jax_qs[i]

        ref_pos = ref_node_pos(ref_q, N) if ref_is_separated else jax_node_pos(ref_q, N)
        jax_pos = jax_node_pos(jax_q, N) if jax_is_interleaved else ref_node_pos(jax_q, N)

        pos_diff = np.max(np.abs(ref_pos - jax_pos))
        q_diffs.append(pos_diff)

    q_diffs = np.array(q_diffs)
    print(f"  Max position difference across all steps: {q_diffs.max():.2e}")
    print(f"  Mean position difference: {q_diffs.mean():.2e}")
    print(f"  Position diff at step 100: {q_diffs[min(100, n_common-1)]:.2e}")
    print(f"  Position diff at step 500: {q_diffs[min(500, n_common-1)]:.2e}")
    print(f"  Position diff at final step: {q_diffs[-1]:.2e}")

    # ── Bifurcation diagram ──
    ref_dh, ref_dw, L_val = extract_bifurcation_metrics(ref_qs, N)
    jax_dh, jax_dw, _ = extract_bifurcation_metrics(jax_qs, N)

    print(f"\n--- Bifurcation diagram ---")
    print(f"  Reference: {len(ref_dh)} points, max |H_mid/L|={np.max(np.abs(ref_dh)):.4f}")
    print(f"  JAX:       {len(jax_dh)} points, max |H_mid/L|={np.max(np.abs(jax_dh)):.4f}")

    # Plot overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(ref_dw, np.abs(ref_dh), 'b-', linewidth=2, label='Reference (NumPy/PyTorch)', alpha=0.8)
    ax.plot(jax_dw, np.abs(jax_dh), 'r--', linewidth=2, label='JAX', alpha=0.8)
    ax.set_xlabel('ΔW/L')
    ax.set_ylabel('|H_mid/L|')
    ax.set_title('Bifurcation Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(q_diffs, 'k-', linewidth=1)
    ax.set_xlabel('Logged step')
    ax.set_ylabel('Max |Δpos|')
    ax.set_title('Position Difference vs Step')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Elastic energy comparison
    if ref.get('elastic_energies') and jax.get('elastic_energies'):
        ref_E = np.array(ref['elastic_energies'])
        jax_E = np.array(jax['elastic_energies'])
        n_E = min(len(ref_E), len(jax_E))
        if n_E > 0:
            ax.plot(ref_E[:n_E], 'b-', linewidth=1.5, label='Reference', alpha=0.8)
            ax.plot(jax_E[:n_E], 'r--', linewidth=1.5, label='JAX', alpha=0.8)
            ax.set_xlabel('Logged step')
            ax.set_ylabel('Elastic Energy')
            ax.set_title('Elastic Energy')
            ax.legend()
            ax.grid(True, alpha=0.3)

            E_diff = np.abs(ref_E[:n_E] - jax_E[:n_E])
            rel_E_diff = E_diff / (np.abs(ref_E[:n_E]) + 1e-30)
            print(f"\n--- Elastic Energy ---")
            print(f"  Max absolute diff: {E_diff.max():.2e}")
            print(f"  Max relative diff: {rel_E_diff.max():.2e}")
            print(f"  Mean relative diff: {rel_E_diff.mean():.2e}")

    plt.tight_layout()
    plot_path = out_dir / f'comparison_n{N}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Speed: JAX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than reference")
    print(f"  Accuracy: max position diff = {q_diffs.max():.2e}")
    print(f"  Steps: ref={ref_steps}, jax={jax_steps}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='/data/shivam/Ribbon/bench_results/ref_result_n21_wl12.pkl')
    parser.add_argument('--jax', type=str, default='/data/shivam/Ribbon/bench_results/jax_result_n21_wl12.pkl')
    args = parser.parse_args()
    compare(args.ref, args.jax)
