"""Headless Cartesian-oval lens CLI; output STL coordinates are millimetres."""
import argparse
import json
from pathlib import Path
from singlet import SigmaCurve, OmegaLens
from surface_generation import build_lens_mesh, export_to_stl

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration; otherwise use the legacy example in centimetres")
    parser.add_argument("--output", type=Path, default=Path("cartesian_singlet.stl"))
    parser.add_argument("--input-units", choices=("mm", "cm"), help="Override config units (legacy default: cm)")
    parser.add_argument("--resolution-um", type=float, default=5.0, help="Nominal printer resolution; sets default numerical error budget only")
    parser.add_argument("--surface-error-um", type=float, help="Sampled surface error budget; default: resolution/5")
    parser.add_argument("--aperture-diameter-mm", type=float, help="Trim to a smaller clear aperture and close a finite cylindrical rim")
    parser.add_argument("--max-vertices", type=int, default=1000000)
    parser.add_argument("--max-faces", type=int, default=2000000)
    parser.add_argument("--report", type=Path, help="Also save JSON mesh/fidelity metrics")
    parser.add_argument("--view", action="store_true", help="Open the optional 3D viewer after export")
    args = parser.parse_args(argv)
    try:
        import math
        if not math.isfinite(args.resolution_um) or args.resolution_um <= 0:
            raise ValueError("resolution-um must be positive and finite")
        params = (json.loads(args.config.read_text()) if args.config else
                  dict(z0=100000.0, z1=60.0, z2=30.0, n0=1.0, n1=1.5, t=4.0, units="cm"))
        units = args.input_units or params.get("units", "cm")
        if units not in ("mm", "cm"):
            raise ValueError("Config units must be mm or cm")
        scale = 10.0 if units == "cm" else 1.0
        z0, z1, z2, t = (float(params[key])*scale for key in ("z0", "z1", "z2", "t"))
        n0, n1 = float(params["n0"]), float(params["n1"])
        lens = OmegaLens(SigmaCurve(z0, z1, n0, n1),
                         SigmaCurve(-z1, z2, n1, n0, t_shift=t))
        error = args.surface_error_um if args.surface_error_um is not None else args.resolution_um/5
        vertices, faces, report = build_lens_mesh(lens, error/1000,
                                                 args.max_vertices, args.max_faces,
                                                 aperture_radius_mm=(args.aperture_diameter_mm/2 if args.aperture_diameter_mm is not None else None))
        report.update(input_units=units, nominal_printer_resolution_um=args.resolution_um,
                      output=str(args.output), validation="sampled analytic triangle audit; not an optical accuracy certification")
        export_to_stl(args.output, vertices, faces)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        if args.view:
            from viewer import show_vispy
            show_vispy(vertices, faces)
        return 0
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    raise SystemExit(main())
