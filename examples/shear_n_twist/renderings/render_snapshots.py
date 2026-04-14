#!/usr/bin/env python3
"""Render PNG snapshots from ribbon_data.json using matplotlib (headless).

Produces frames in <outdir>/frame_0000.png ... one per exported mesh frame.
Use this when you can't run the CCapture-based HTML snapshotter in a browser.

Usage:
  python render_snapshots.py ribbon_data.json --outdir snapshots_1b12 [--step 1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rail_corners(rf, along, height, width):
    """Return the 8 corners of the rail box oriented by the rail frame.
    Local dims: along-tangent, along-normal (height), along-width_dir."""
    t = np.asarray(rf["tangent"]); n = np.asarray(rf["normal"]); w = np.asarray(rf["width_dir"])
    o = np.asarray(rf["origin"])
    hx, hy, hz = along / 2, height / 2, width / 2
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1)]
    return np.array([o + sx*hx*t + sy*hy*n + sz*hz*w for (sx, sy, sz) in signs])


BOX_FACES = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (1,2,6,5), (0,3,7,4)]


def render_frame(frame, width, length, out_path, view_azim, view_elev):
    verts = np.asarray(frame["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(frame["faces"], dtype=np.int64).reshape(-1, 3)
    tris = verts[faces]

    fig = plt.figure(figsize=(10, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    pc = Poly3DCollection(tris, facecolor="#88ccff", edgecolor="#3377bb",
                          linewidths=0.15, alpha=1.0)
    ax.add_collection3d(pc)

    # Rails
    rail_along = 0.02
    rail_height = 0.03
    rail_width = max(width * 1.5, 0.02)
    for side, color in [("start", "#888888"), ("end", "#666666")]:
        rf = frame["rails"][side]
        cs = rail_corners(rf, rail_along, rail_height, rail_width)
        polys = [cs[list(f)] for f in BOX_FACES]
        ax.add_collection3d(Poly3DCollection(polys, facecolor=color,
                                             edgecolor="black", linewidths=0.4, alpha=1.0))

    # Axes: keep equal scaling; center around length / 2
    cx, cz = length / 2, 0
    R = max(length * 0.6, 6 * width)
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(-R, R)
    ax.set_zlim(cz - R, cz + R)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_axis_off()
    ax.set_title(f"t = {frame['simTime']:.2f} s", fontsize=13, y=0.95)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--elev", type=float, default=22)
    args = ap.parse_args()

    data = json.load(open(args.json))
    width = float(data["width"])
    length = float(data.get("length", 0.1))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    frames = data["frames"][:: args.step]
    for i, fr in enumerate(frames):
        p = outdir / f"frame_{i:04d}.png"
        render_frame(fr, width, length, p, args.azim, args.elev)
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  {i+1}/{len(frames)}  -> {p.name}  (t={fr['simTime']:.2f}s)")
    print(f"Wrote {len(frames)} frames to {outdir}/")


if __name__ == "__main__":
    main()
