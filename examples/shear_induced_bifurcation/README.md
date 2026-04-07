# Shear-Induced Bifurcation of an Elastic Ribbon

This example simulates shear-induced out-of-plane buckling of a thin elastic ribbon, following the protocol from Sano & Wada (2019). A flat ribbon is first compressed axially, then sheared transversely until it buckles out of plane, and finally the shear is reversed to study hysteresis.

## Physical Setup

A thin ribbon of length `L = 0.1 m`, thickness `h = 1 mm`, and width `w = (W/L) * L` is modeled as a discrete elastic rod (DER) with `N` nodes along its centerline. The ribbon lies initially along the x-axis from `x = 0` to `x = L`.

```
        Fixed end (x >= 0.09)
        ┌──────────────┐
        │              │
  y ↑   │   RIBBON     │
    │   │  (flat, z=0) │
    │   │              │
    └───┤──────────────├──→ x
        │              │
        └──────────────┘
        Moving end (x <= 0.01)
```

### Material Properties

| Parameter     | Value     | Description           |
|---------------|-----------|-----------------------|
| `E`           | 10 GPa    | Young's modulus       |
| `nu`          | 0.5       | Poisson ratio         |
| `rho`         | 1000 kg/m^3 | Density             |
| `h`           | 1 mm      | Thickness             |
| `L`           | 0.1 m     | Length                |
| `W/L`         | 1/40 to 1/6 | Width-to-length ratio |

### Cross-Section Properties (derived)

The ribbon cross-section `w x h` gives the following rod section properties:

| Property | Formula          | Description                        |
|----------|------------------|------------------------------------|
| `A_xs`   | `w * h`          | Cross-sectional area               |
| `I_xs1`  | `w * h^3 / 12`   | Second moment about local e1 axis  |
| `I_xs2`  | `h * w^3 / 12`   | Second moment about local e2 axis  |
| `J_xs`   | `w * h^3 / 3`    | Torsional constant                 |

## Boundary Conditions

### Fixed Nodes

Both ends of the ribbon are clamped:

- **Moving end ("start"):** all nodes with `x <= 0.01` (leftmost ~10% of ribbon)
- **Fixed end ("end"):** all nodes with `x >= 0.09` (rightmost ~10% of ribbon)

Clamping means all translational DOFs `(x, y, z)` are constrained. These nodes are removed from the free DOF set in the Newton solve. The moving end nodes are displaced prescriptively during loading phases.

### Loading Protocol

The simulation proceeds in 4 phases over `t = 0` to `12.5 s`:

```
Phase 0: Gravity ramp     t ∈ [0, 0.05]        gz = -1000 m/s^2
Phase 1: Compression       t ∈ [0.05, 2.55]     move start nodes in +x by u0*dt per step
Phase 2: Forward shear     t ∈ [2.55, 7.55]     move start nodes in +y by u0*dt per step
Phase 3: Reverse shear     t ∈ [7.55, 12.5]     move start nodes in -y by u0*dt per step
```

where `u0 = 0.01 m/s` is the prescribed displacement rate.

#### Phase 0 — Gravity Ramp (settling)

A strong artificial gravity `g_z = -1000 m/s^2` is applied for 0.05 s to push the ribbon flat. After this phase, gravity is turned off (`g = 0`) for the remainder. This ensures the ribbon starts from a well-defined flat configuration regardless of initial perturbations.

#### Phase 1 — Axial Compression

The start (left) nodes are moved in the `+x` direction at rate `u0 * dt` per step. This compresses the ribbon axially along its length. The compression is small — total displacement is `u0 * (2.55 - 0.05) = 0.025 m` over 2.5 s.

#### Phase 2 — Forward Shear

The start nodes are moved in the `+y` direction (transverse to the ribbon axis). As the shear displacement `DeltaW` increases, the ribbon's midpoint eventually buckles out of the `z = 0` plane. The critical shear displacement depends on `W/L`:

- Narrow ribbons (`W/L = 1/40`) buckle at small shear
- Wide ribbons (`W/L = 1/6`) require more shear to buckle

The bifurcation diagram (`|H_mid|/L` vs `DeltaW/L`) captures this transition.

#### Phase 3 — Reverse Shear

The start nodes are moved in the `-y` direction, reversing the shear. The ribbon may exhibit hysteresis — the out-of-plane displacement does not retrace the forward path exactly. This is the most numerically challenging phase, requiring adaptive time-stepping with dt reductions down to `O(1e-4)` or smaller.

### How `move_nodes` Works

At each time step, the `before_step` callback:

1. Updates the external force vector `F_ext` based on the current gravity phase
2. Calls `dm.move_nodes(q, rod, start_nodes, displacement, direction)` which:
   - Shifts the target DOFs in `q` by the prescribed displacement
   - Updates the rod's reference position to reflect the new BC
   - The constraint is enforced by excluding these DOFs from the Newton solve

The displacement per step scales with `dt`, so adaptive time-stepping naturally adjusts the displacement magnitude. When dt is halved on a retry, `move_nodes` is re-applied with the smaller displacement (the BC rollback mechanism handles this).

## Energy Models

Five energy models are available, all built on the Kirchhoff baseline:

| Model       | Key Feature                                      |
|-------------|--------------------------------------------------|
| `kirchhoff` | Quadratic: stretch + bend + twist                |
| `sano`      | Kirchhoff + nonlinear twist-curvature coupling   |
| `audoly`    | Kirchhoff + geometry-dependent nonlinear term     |
| `sadowsky`  | Kirchhoff + regularized twist-curvature coupling |
| `wunderlich`| Kirchhoff + nonlocal eta' (curvature derivative) |

Select via `energy_model.name` in `config.yaml` or `--energy-model` CLI flag.

## Running

```bash
# Single run: Sano model, W/L=1/12, 45 nodes
python simulate.py --config config.yaml --pkl-dir out/pkls --plot-dir out/plots \
    --nodes 45 --wbyl 1/12 --energy-model sano

# Multiple W/L ratios
python simulate.py --config config.yaml --pkl-dir out/pkls --plot-dir out/plots \
    --nodes 45 --wbyl 1/40 1/20 1/12 1/6

# Quick test (fewer nodes, shorter time)
python simulate.py --config config_quick_test.yaml --pkl-dir out/pkls --plot-dir out/plots
```

## Output

- **Pickle files** in `--pkl-dir`: trajectory `qs`, `times`, forces, elastic energies
- **Plots** in `--plot-dir`: `|H_mid|/L` vs `DeltaW/L` bifurcation diagrams

## Key Quantities

| Symbol       | Definition                                             |
|--------------|--------------------------------------------------------|
| `H_mid`      | z-displacement of the midpoint node (at `x = L/2`)    |
| `DeltaW`     | Mean y-displacement of the start (moving) nodes        |
| `|H_mid|/L`  | Normalized out-of-plane deflection (bifurcation order parameter) |
| `DeltaW/L`   | Normalized shear displacement (control parameter)      |

## Adapting for Other Boundary Conditions

To create a new loading protocol (e.g., shear + twist):

1. Copy this directory as a starting template
2. Modify `motion_phases` in `config.yaml` to define your phases:
   ```yaml
   motion_phases:
     - { start_time: 0.05, end_time: 2.0, direction: 0 }             # compression
     - { start_time: 2.0,  end_time: 6.0, direction: 1 }             # shear
     - { start_time: 6.0,  end_time: 10.0, direction: 1, reverse: true } # reverse
   ```
   `direction`: 0 = x, 1 = y, 2 = z. `reverse: true` negates the displacement sign.

3. For rotational BCs (twist), you'll need to modify the `before_step` callback in `simulate.py`. The current `dm.move_nodes` only handles translational displacement. For twist, you'd rotate the node positions and update the material frame accordingly. The key function signature is:
   ```python
   def before_step(rod, q, u, aux, t, dt_step):
       # rod: current Rod pytree (contains BC info, F_ext)
       # q: current DOF vector [x0,y0,z0,theta0, x1,y1,z1,theta1, ...]
       # u: current velocity vector
       # aux: auxiliary state (material frame angles)
       # t: current simulation time
       # dt_step: current time step size (varies with adaptive dt)
       # Returns: (rod, q, u, aux) — updated state
   ```

4. The DOF layout per node is `[x, y, z, theta]` where `theta` is the material frame twist angle. To impose a twist BC, increment `q[4*i + 3]` for the target nodes.

5. Make sure any BC displacement scales with `dt_step` (not the base `dt`) so adaptive time-stepping works correctly with BC rollback on retry.
