# Cartesian Surfaces STL Generator

Generate an axisymmetric optical singlet from two Cartesian ovoid surfaces.
The project calculates the meridional profile of each refracting surface,
finds their intersection, and revolves the enclosed profile about the optical
axis to produce a triangular mesh suitable for STL export.

The current implementation is intended for exploring Cartesian-ovoid lens
geometries. It includes interactive 2D and 3D visualizations and a small STL
export API.

## How it works

1. `SigmaCurve` represents a Cartesian ovoid in the meridional `(z, r)` plane.
2. `OmegaLens` finds the intersection of two ovoids with a two-variable
   Newton-Raphson solve and joins the two surface segments into the lens
   profile.
3. `revolve_curve` rotates that profile around the `z` axis and returns mesh
   vertices and triangular faces.
4. `export_to_stl` writes that mesh as an ASCII STL file.

All examples treat distances as centimetres. The numerical code itself does
not enforce a unit system, so use one unit system consistently.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- VisPy (for the interactive 3D viewer)

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib vispy
```

## Quick start

Edit [`queue/params.json`](queue/params.json) to define the optical setup:

```json
{
  "z0": 100000.0,
  "z1": 60.0,
  "z2": 30.0,
  "n0": 1.0,
  "n1": 1.5,
  "t": 4.0
}
```

Then run:

```bash
python main.py
```

The script prints the lens diameter and intersection point, opens a VisPy 3D
mesh view, and then displays the meridional profile in Matplotlib. Close the
3D window to continue to the 2D plot.

Set `SHOW_3D` or `SHOW_2D` to `False` in `main.py` if the environment does not
support GUI windows.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `z0` | Object position for the first surface |
| `z1` | Intermediate image position shared by the two surfaces |
| `z2` | Final image position for the second surface |
| `n0` | Refractive index of the surrounding medium |
| `n1` | Refractive index of the lens material |
| `t` | Axial shift between the two surfaces; interpreted as lens width in the example |

The current `main.py` uses the constants `N_RHO` and `N_ANG` at the top of the
file for profile and angular mesh resolution. Although the sample JSON contains
`rho_points`, `main.py` does not currently read that value.

## Export an STL

`main.py` currently generates and visualizes the mesh, but it does **not**
write an STL file by itself. Add the following after the call to
`revolve_curve` in `main.py` to export one:

```python
from surface_generation import export_to_stl

export_to_stl("cartesian_singlet.stl", vertices, faces, solid_name="cartesian_singlet")
```

For a finer mesh, increase `N_RHO` and `N_ANG`. This increases the number of
vertices and triangles; confirm the resulting STL is suitable for the target
CAD or manufacturing workflow.

## Project layout

| File | Responsibility |
| --- | --- |
| `main.py` | Loads parameters and creates the lens mesh and visualizations |
| `singlet.py` | Cartesian-ovoid curves and singlet intersection logic |
| `math_utils.py` | Bisection and Newton-Raphson numerical routines |
| `surface_generation.py` | Surface of revolution, mesh construction, and STL export |
| `viewer.py` | VisPy and Matplotlib visualization helpers |
| `queue/params.json` | Example optical parameters |
