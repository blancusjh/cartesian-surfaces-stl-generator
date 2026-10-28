# Cartesian Surfaces STL Generator

Generate a closed axisymmetric singlet from two Cartesian-oval surfaces,
with adaptive tessellation and headless binary STL export. The analytic
near-axis oval formulation is retained; the surfaces satisfy their signed
Cartesian-oval optical-path equations, including virtual conjugates.

## Quick start

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python main.py --config queue/params.json --output lens.stl --report lens.json
python -m pytest -v
```

No viewer or display is needed. The CLI also works as `cartesian-lens` after
installation. Without `--config`, it uses the legacy example values shown
below, interpreted as centimetres.

Output STL coordinates are **millimetres**. STL itself does not encode
units: select millimetres when importing into a slicer or CAD application.
The JSON report records the units, diameter, thickness, positive volume,
vertex/triangle counts and sampled fidelity measurements.

## Optical parameters and units

```json
{
  "z0": 100000.0,
  "z1": 60.0,
  "z2": 30.0,
  "n0": 1.0,
  "n1": 1.5,
  "t": 4.0,
  "units": "cm"
}
```

| Parameter | Meaning |
| --- | --- |
| `z0` | First object coordinate is `-z0` |
| `z1` | Shared intermediate image coordinate |
| `z2` | Final image coordinate |
| `n0` | Positive surrounding refractive index |
| `n1` | Distinct positive lens refractive index |
| `t` | Positive second-vertex axial shift from the first vertex |
| `units` | Input lengths: `cm` or `mm`; omitted means legacy `cm` |

The first surface uses object/image coordinates `(-z0, z1)`. The second
uses `(z1, z2)`, so its first conjugate is virtual in the example.
Coordinates are absolute along the optical axis. A surface at `t_shift`
uses local signed object distance `z0+t_shift` and local image distance
`zi-t_shift`. The shift is added to its sag exactly once.

The CLI converts all lengths to mm before geometry generation.
`--input-units mm` overrides configuration units. Legacy `rho_points`
configuration entries are accepted but do not control adaptive meshing.

## Using a nominal 5 micrometre printer resolution

```bash
python main.py --output lens.stl --resolution-um 5 --surface-error-um 1
```

The default nominal resolution is 5 µm. The default numerical surface-error
budget is resolution/5, or **1 µm**. This is a practical tessellation
heuristic, not a claim that the printed lens has 1 µm or 5 µm surface accuracy.
Changing `--resolution-um` changes that default budget; an explicit
`--surface-error-um` takes precedence.

The meridian subdivides where seven interior analytic points depart from a
straight chord by more than 30% of the requested budget. The azimuth segment
count follows the circular sagitta bound
`r_max * (1 - cos(pi/n)) <= 0.3 * budget`. This avoids a uniform 5 µm grid
over the whole lens while allowing the mesh to follow the curved surfaces.

Before export, each meridional interval's two triangle shapes are audited on
a 17 by 17 interior parameter grid in one angular sector. Rotational symmetry
covers the other sectors. At each sample, the analytic oval is inverted at
the mesh point's cylindrical radius; the vertical discrepancy is an upper
bound on that sampled point's distance to the analytic surface. The report
records the maximum sampled discrepancy and the maximum float32 vertex
rounding displacement. Their sum must fit the requested budget.

This is **sampled validation, not a certified global Hausdorff error bound**.
The sample grid can miss an extremum. Tighten the budget and compare results
for demanding work. The tests include a denser, differently spaced audit and
a refinement convergence check. Steep or unsupported branches are rejected
when the requested sampled budget cannot be met.

A printer's quoted resolution may describe XY pixels or voxels, rather than
axial layer height. Cure spread, shrinkage, material index/dispersion, surface
roughness and finishing affect the result independently. Neither this mesh
audit nor a constant optical path for each individual surface establishes
the focusing performance of a complete manufactured lens. Measure the
printed part and perform optical/ray validation for the intended application.

## A finite rim for printing

The full intersection lens has **zero edge thickness**: its two surfaces
meet at a mathematical knife edge. For a finite printable rim, choose a
clear aperture smaller than the reported full-intersection diameter:

```bash
python main.py --aperture-diameter-mm 180 --output lens_with_rim.stl
```

Both exact oval branches are trimmed at that radius, then connected by a
cylindrical side wall. The mesh shares each edge ring with the adjacent
optical surface. The report records finite-rim status, edge thickness and
a sampled minimum axial thickness. The circular side wall follows the same
azimuth sagitta budget. Choose an edge thickness appropriate for your
material, printer and handling; this option supplies no strength guarantee.
The Python API accepts `aperture_radius_mm`. Successful GitHub Actions runs
also attach the 180 mm finite-rim example STL and its metrics as a downloadable
artifact, retained for 14 days.

## Mesh and resource checks

The profile begins and ends at exact axis poles. Each pole is one vertex;
the full optical surfaces share one intersection ring (trimmed lenses have
two distinct edge rings joined by the cylindrical rim). There are no repeated
seam vertices or collapsed triangles. Every edge must appear twice with
opposite directions, every triangle must have positive area, and signed
volume must be positive. The same checks run after float32 STL quantization.

The default limits are 1,000,000 vertices and 2,000,000 triangles. Counts
are checked before allocating the mesh. Profile refinement also has a
10,000-point limit per surface and a depth limit. Exceeding a limit fails
with an actionable error; it never silently lowers fidelity.

```bash
python main.py --surface-error-um 2 --max-vertices 1000000 --max-faces 2000000
```

Invalid/nonfinite inputs, singular optical parameters, nonpositive
thickness, failed intersection solves, or a nonmonotonic radial branch
produce errors. This implementation supports the near-axis lens bounded by
an intersection between the two vertex planes. It does not select arbitrary
disconnected oval branches.

## Python API

The core geometry API takes a single consistent length unit. To interpret
`surface_error_mm` and the metrics correctly, provide lengths in mm:

```python
from singlet import SigmaCurve, OmegaLens
from surface_generation import build_lens_mesh, export_to_stl

front = SigmaCurve(1_000_000, 600, 1.0, 1.5)
back = SigmaCurve(-600, 300, 1.5, 1.0, t_shift=40)
lens = OmegaLens(front, back)
vertices, faces, metrics = build_lens_mesh(lens, surface_error_mm=0.001)
export_to_stl("lens.stl", vertices, faces)
print(metrics)
```

`SigmaCurve.get_points(rhof)` includes the exact pole. Without `rhof`, it
uses the lens intersection extent after `OmegaLens` construction.
`revolve_curve(profile, n_ang)` now interprets `n_ang` as the actual number
of azimuth segments (the old implementation used twice that count).
`export_to_stl` now writes binary STL, in bounded chunks.

## Optional interactive view

```bash
pip install -e '.[view]'
python main.py --output lens.stl --view
```

Viewing is explicit and happens after the validated file is written.

## Project layout

| File | Responsibility |
| --- | --- |
| `main.py` | CLI configuration, units, export and JSON metrics |
| `singlet.py` | Analytic Cartesian ovals, signed optical paths and bounded intersection |
| `math_utils.py` | Bounded root helpers |
| `surface_generation.py` | Adaptive profile, manifold mesh, sampled audit and binary STL |
| `viewer.py` | Optional VisPy and Matplotlib helpers |
| `queue/params.json` | Legacy example, with explicit cm input units |
| `qa/test_lens.py` | Asserted optical, geometric, topology, CLI and STL regressions |
