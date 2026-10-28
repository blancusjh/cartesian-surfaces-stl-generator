"""Independent regression tests for optics, fidelity, topology and STL units."""
import json
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
from math_utils import biseccion, biseccion_mod
from singlet import SigmaCurve, OmegaLens, sigma
from surface_generation import (adaptive_profile, build_lens_mesh, export_to_stl,
                                revolve_curve, sampled_surface_error, validate_mesh)

def lens_at(shift=0.0, thickness=4.0):
    return OmegaLens(SigmaCurve(100000, 60, 1, 1.5, t_shift=shift),
                     SigmaCurve(-60, 30, 1.5, 1, t_shift=shift+thickness))

@pytest.mark.parametrize("solver", [biseccion, biseccion_mod])
def test_root_endpoints_and_bracket(solver):
    assert solver(lambda x: x, 0, 1) == 0
    assert solver(lambda x: x-1, 0, 1) == 1
    assert abs(solver(lambda x: x*x-2, 0, 2)-np.sqrt(2)) < 2e-6
    with pytest.raises(ValueError):
        solver(lambda x: 1, 0, 1)
    with pytest.raises(ValueError):
        solver(lambda x: np.nan, 0, 1)
    with pytest.raises(RuntimeError):
        solver(lambda x: x*x-2, 0, 2, tol=1e-15, max_iter=1)

def test_expansion_is_bounded_and_handles_zero_initial_bound():
    assert abs(biseccion_mod(lambda x: x-3, -1, 0)-3) < 1e-8
    with pytest.raises(ValueError):
        biseccion_mod(lambda x: x*x+1, 0, 1, max_expand=3)

@pytest.mark.parametrize("shift", [0, 1.5])
def test_independent_signed_optical_path_and_seam(shift):
    lens = lens_at(shift)
    # Independent absolute-focus oracle. No call to optical_residual here.
    for surface, end in zip((lens.s1, lens.s2), lens.rho_intersection):
        points = surface.evaluate(np.linspace(0, end, 101))
        z, r = points.T
        object_focus, image_focus = -surface.z0, surface.zi
        object_sign = np.sign(surface.t_shift-object_focus)
        image_sign = np.sign(image_focus-surface.t_shift)
        path = (surface.n0*object_sign*np.hypot(z-object_focus, r)
                +surface.ni*image_sign*np.hypot(z-image_focus, r))
        vertex_path = (surface.n0*object_sign*abs(surface.t_shift-object_focus)
                       +surface.ni*image_sign*abs(image_focus-surface.t_shift))
        np.testing.assert_allclose(path, vertex_path, atol=5e-10, rtol=0)
        np.testing.assert_allclose(points[-1], lens.intersection_point, atol=1e-9)
    profile = lens.get_points()
    assert profile[0, 1] == profile[-1, 1] == 0
    assert np.count_nonzero(np.linalg.norm(profile-lens.intersection_point, axis=1) < 1e-9) == 1
    assert lens.intersection_point[0] >= lens.s1.t_shift

@pytest.mark.parametrize("shift,thickness", [(0, 4), (1.5, 4), (0, 0.1), (0, 8)])
def test_closed_outward_manifold_and_sampled_fidelity(shift, thickness):
    lens = lens_at(shift, thickness)
    v, f, report = build_lens_mesh(lens, surface_error_mm=0.005)
    mesh = trimesh.Trimesh(v, f, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume
    assert mesh.euler_number == 2 and len(mesh.split()) == 1
    assert report["volume_mm3"] > 0
    assert np.all(mesh.area_faces > 0)
    assert report["sampled_surface_error_um"]+report["stl_rounding_bound_um"] <= 5
    assert len(np.unique(v, axis=0)) == len(v)
    # Independent denser audit, at a different grid spacing from production.
    for surface, end in zip((lens.s1, lens.s2), lens.rho_intersection):
        rho, _ = adaptive_profile(surface, end, 0.005*0.3)
        assert sampled_surface_error(surface, rho, report["azimuth_segments"], subdivisions=23) < 0.005

def test_refinement_converges_without_uniform_micron_grid():
    lens = lens_at()
    _, f1, coarse = build_lens_mesh(lens, 0.01)
    _, f2, fine = build_lens_mesh(lens, 0.0025)
    assert len(f2) > len(f1)
    assert fine["sampled_surface_error_um"] < coarse["sampled_surface_error_um"]
    assert len(f2) < 500000

def test_binary_roundtrip_has_mm_geometry(tmp_path):
    lens = lens_at(1.5)
    v, f, report = build_lens_mesh(lens, 0.005)
    output = tmp_path/"lens.stl"
    export_to_stl(output, v, f)
    data = output.read_bytes()
    assert len(data) == 84+50*len(f)
    assert struct.unpack("<I", data[80:84])[0] == len(f)
    mesh = trimesh.load_mesh(output, process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    np.testing.assert_allclose(mesh.bounds, np.vstack((v.min(axis=0), v.max(axis=0))), atol=2e-6)
    assert abs(mesh.volume-report["volume_mm3"]) < report["volume_mm3"]*1e-6

def test_cli_legacy_cm_to_mm_and_default_one_micron_budget(tmp_path):
    output, report_path = tmp_path/"lens.stl", tmp_path/"metrics.json"
    result = subprocess.run([sys.executable, "main.py", "--output", str(output),
                             "--report", str(report_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(report_path.read_text())
    assert report["input_units"] == "cm" and report["units"] == "mm"
    assert report["thickness_mm"] == 40
    assert report["surface_error_requested_um"] == 1
    assert report["nominal_printer_resolution_um"] == 5
    assert 200 < report["diameter_mm"] < 220
    assert report["faces"] < 2000000
    assert output.stat().st_size == 84+50*report["faces"]
    mesh = trimesh.load_mesh(output, process=True)
    assert mesh.is_watertight and mesh.is_volume
    assert abs(mesh.extents[2]-40) < 1e-5

def test_cli_mm_config_and_invalid_options(tmp_path):
    config = tmp_path/"params.json"
    config.write_text(json.dumps(dict(z0=100000, z1=60, z2=30, n0=1, n1=1.5, t=4, units="mm")))
    output = tmp_path/"small.stl"
    completed = subprocess.run([sys.executable, "main.py", "--config", str(config),
                                "--output", str(output), "--surface-error-um", "10"],
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["thickness_mm"] == 4
    failed = subprocess.run([sys.executable, "main.py", "--resolution-um", "nan"],
                            capture_output=True, text=True)
    assert failed.returncode != 0 and "positive and finite" in failed.stderr

def test_invalid_geometry_and_resource_guards(tmp_path):
    for args in [(1, 2, 1, 1), (0, 2, 1, 1.5), (1, 2, -1, 1.5), (1, np.nan, 1, 1.5)]:
        with pytest.raises(ValueError):
            SigmaCurve(*args)
    with pytest.raises(ValueError):
        sigma(100000, 60, np.nan, 1, 1.5)
    with pytest.raises(ValueError):
        OmegaLens(SigmaCurve(100000, 60, 1, 1.5), SigmaCurve(-60, 30, 1.5, 1))
    with pytest.raises(ValueError, match="resource limit|resource"):
        build_lens_mesh(lens_at(), 0.001, max_vertices=100, max_faces=100)
    with pytest.raises(ValueError):
        build_lens_mesh(lens_at(), 0)
    with pytest.raises(ValueError):
        revolve_curve(np.array([[0, 0], [1, 1], [1, 1], [2, 0]]), 8)
    v, f = revolve_curve(np.array([[0, 0], [1, 1], [2, 0]]), 8)
    with pytest.raises(ValueError):
        validate_mesh(v, f[:-1])
    with pytest.raises(ValueError):
        validate_mesh(v, f[:, ::-1])

def test_finite_aperture_rim_is_closed_and_has_positive_edge_thickness(tmp_path):
    lens = lens_at(1.5)
    radius = lens.intersection_point[1]*0.8
    v, f, report = build_lens_mesh(lens, 0.005, aperture_radius_mm=radius)
    mesh = trimesh.Trimesh(v, f, process=False)
    assert mesh.is_watertight and mesh.is_volume and mesh.euler_number == 2
    assert report["finite_rim"] and report["edge_thickness_mm"] > 0
    assert report["sampled_min_thickness_mm"] > 0
    assert abs(report["diameter_mm"]-2*radius) < 1e-10
    export_to_stl(tmp_path/"rim.stl", v, f)
    for invalid in (-1, 0, np.nan, lens.intersection_point[1]*1.1):
        with pytest.raises(ValueError):
            build_lens_mesh(lens, 0.005, aperture_radius_mm=invalid)
