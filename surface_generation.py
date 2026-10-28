"""Bounded adaptive tessellation and binary STL output (coordinates in mm)."""
import struct
from pathlib import Path
import numpy as np

def _integer_budget(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")

def adaptive_profile(surface, rho_end, error, max_points=10000):
    """Refine using seven interior chord probes. This is a sampled error criterion."""
    _integer_budget(max_points, "max_points", 2)
    if not np.isfinite(error) or error <= 0:
        raise ValueError("error must be positive and finite")
    result = [(0.0, surface.evaluate(0.0))]
    stack = [(0.0, rho_end, 0)]
    fractions = np.arange(1, 8)/8
    while stack:
        a, b, depth = stack.pop()
        pa, pb = surface.evaluate(a), surface.evaluate(b)
        exact = surface.evaluate(a+(b-a)*fractions)
        chord = pa+(pb-pa)*fractions[:, None]
        deviation = np.max(np.linalg.norm(exact-chord, axis=1))
        if deviation > error:
            if depth >= 30 or len(result)+len(stack)+2 >= max_points:
                raise ValueError("Profile refinement exceeds resource limit; relax surface error")
            mid = (a+b)/2
            stack.extend(((mid, b, depth+1), (a, mid, depth+1)))
        else:
            result.append((b, pb))
    return np.array([v[0] for v in result]), np.array([v[1] for v in result])

def signed_volume(vertices, faces):
    triangles = vertices[faces]
    return float(np.einsum("ij,ij->i", triangles[:, 0],
                          np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)

def validate_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must be finite Nx3 coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or len(faces) == 0:
        raise ValueError("Mesh faces must be nonempty integer Nx3 indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Face index outside vertex array")
    tri = vertices[faces]
    if np.any(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) == 0):
        raise ValueError("Mesh has degenerate triangles")
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=direction, minlength=len(unique))
    if np.any(counts != 2) or np.any(balance != 0):
        raise ValueError("Mesh must be a closed consistently oriented two-manifold")
    volume = signed_volume(vertices, faces)
    if volume <= 0:
        raise ValueError("Mesh must have positive outward-oriented volume")
    return volume

def revolve_curve(curve_pts, n_ang, max_vertices=1000000, max_faces=2000000):
    """Revolve an axis-to-axis (z,r) profile; n_ang is the azimuth segment count."""
    _integer_budget(max_vertices, "max_vertices", 5)
    _integer_budget(max_faces, "max_faces", 6)
    p = np.asarray(curve_pts, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.all(np.isfinite(p)):
        raise ValueError("Profile must be a finite Nx2 array with at least three points")
    if p[0, 1] != 0 or p[-1, 1] != 0 or np.any(p[1:-1, 1] <= 0):
        raise ValueError("Profile must have exactly two axis poles and positive interior radii")
    if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) == 0):
        raise ValueError("Profile contains duplicate neighbors")
    _integer_budget(n_ang, "n_ang", 3)
    rings = len(p)-2
    nv, nf = 2+rings*n_ang, 2*rings*n_ang
    if nv > max_vertices or nf > max_faces:
        raise ValueError(f"Mesh needs {nv} vertices/{nf} faces; resource limit exceeded")
    angle = np.arange(n_ang)*(2*np.pi/n_ang)
    vertices = np.empty((nv, 3))
    vertices[0], vertices[-1] = (0, 0, p[0, 0]), (0, 0, p[-1, 0])
    vertices[1:-1, 0] = (p[1:-1, 1, None]*np.cos(angle)).ravel()
    vertices[1:-1, 1] = (p[1:-1, 1, None]*np.sin(angle)).ravel()
    vertices[1:-1, 2] = np.repeat(p[1:-1, 0], n_ang)
    faces = np.empty((nf, 3), dtype=np.int64)
    j = np.arange(n_ang)
    j1 = (j+1) % n_ang
    faces[:n_ang] = np.column_stack((np.zeros(n_ang, dtype=int), 1+j1, 1+j))
    offset = n_ang
    for ring in range(rings-1):
        a, b = 1+ring*n_ang, 1+(ring+1)*n_ang
        faces[offset:offset+2*n_ang:2] = np.column_stack((a+j, a+j1, b+j1))
        faces[offset+1:offset+2*n_ang:2] = np.column_stack((a+j, b+j1, b+j))
        offset += 2*n_ang
    last = 1+(rings-1)*n_ang
    faces[offset:] = np.column_stack((last+j, last+j1, np.full(n_ang, nv-1)))
    if signed_volume(vertices, faces) < 0:
        faces = faces[:, ::-1].copy()
    validate_mesh(vertices, faces)
    return vertices, faces

def sampled_surface_error(surface, rho, n_ang, subdivisions=16):
    """Audit triangle interiors against corresponding analytic surface points.

    Rotational symmetry permits auditing one angular sector. At each triangle
    sample, solve the analytic meridian at the same cylindrical radius: the
    vertical discrepancy bounds point-to-surface distance from above. Includes
    pole fans; never compares only vertices or meridional midpoints.
    """
    maximum = 0.0
    angle = 2*np.pi/n_ang
    for a, b in zip(rho[:-1], rho[1:]):
        pa, pb = surface.evaluate(a), surface.evaluate(b)
        a0 = np.array([pa[1], 0, pa[0]])
        a1 = np.array([pa[1]*np.cos(angle), pa[1]*np.sin(angle), pa[0]])
        b0 = np.array([pb[1], 0, pb[0]])
        b1 = np.array([pb[1]*np.cos(angle), pb[1]*np.sin(angle), pb[0]])
        u, v = np.meshgrid(np.linspace(0, 1, subdivisions+1),
                           np.linspace(0, 1, subdivisions+1), indexing="ij")
        # A pole fan uses the same degenerate quad parameterization, but only
        # its nondegenerate triangle is emitted by revolve_curve.
        upper = (1-v)[..., None]*a0+(v-u)[..., None]*a1+u[..., None]*b1
        lower = (1-u)[..., None]*a0+v[..., None]*b1+(u-v)[..., None]*b0
        interpolated = np.where((v >= u)[..., None], upper, lower)
        target_radius = np.hypot(interpolated[..., 0], interpolated[..., 1])
        lo = np.zeros_like(u)
        hi = np.full_like(u, rho[-1])
        for _ in range(44):
            mid = (lo+hi)/2
            below = surface.evaluate(mid)[..., 1] < target_radius
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        exact_z = surface.evaluate((lo+hi)/2)[..., 0]
        maximum = max(maximum, float(np.abs(exact_z-interpolated[..., 2]).max()))
    return maximum

def _rho_at_radius(surface, end, radius):
    lo, hi = np.zeros_like(np.asarray(radius, dtype=float)), np.full_like(np.asarray(radius, dtype=float), end)
    for _ in range(48):
        mid = (lo+hi)/2
        below = surface.evaluate(mid)[..., 1] < radius
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo+hi)/2

def build_lens_mesh(lens, surface_error_mm=0.001, max_vertices=1000000, max_faces=2000000,
                    aperture_radius_mm=None):
    """Return vertices, faces, and measurable sampled fidelity statistics."""
    error = float(surface_error_mm)
    if not np.isfinite(error) or error <= 0:
        raise ValueError("surface_error_mm must be positive and finite")
    _integer_budget(max_vertices, "max_vertices", 5)
    _integer_budget(max_faces, "max_faces", 6)
    full_radius = lens.intersection_point[1]
    radius = full_radius if aperture_radius_mm is None else float(aperture_radius_mm)
    if not np.isfinite(radius) or radius <= 0 or radius > full_radius:
        raise ValueError("Aperture radius must be positive and no larger than the surface intersection")
    clipped = bool(radius < full_radius)
    ends = (tuple(float(_rho_at_radius(s, end, radius))
                  for s, end in zip((lens.s1, lens.s2), lens.rho_intersection))
            if clipped else lens.rho_intersection)
    profiles = [adaptive_profile(s, end, error*0.3,
                                max_points=min(10000, max_vertices//3))
                for s, end in zip((lens.s1, lens.s2), ends)]
    # Stable sagitta inversion: r*(1-cos(pi/n)) <= 0.3*error.
    ratio = min(1.0, 0.3*error/radius)
    n_ang = max(8, int(np.ceil(np.pi/(2*np.arcsin(np.sqrt(ratio/2))))))
    p1, p2 = profiles[0][1].copy(), profiles[1][1].copy()
    if clipped:
        p1[-1, 1] = p2[-1, 1] = radius
        profile = np.vstack((p1, p2[::-1]))
    else:
        p1[-1] = lens.intersection_point
        profile = np.vstack((p1, p2[-2::-1]))
    edge_thickness = float(p2[-1, 0]-p1[-1, 0]) if clipped else 0.0
    if clipped and edge_thickness <= 0:
        raise ValueError("Aperture must leave positive edge thickness")
    radii = np.linspace(0, radius, 129)
    front_z = lens.s1.evaluate(_rho_at_radius(lens.s1, ends[0], radii))[..., 0]
    back_z = lens.s2.evaluate(_rho_at_radius(lens.s2, ends[1], radii))[..., 0]
    if np.any(back_z[:-1] <= front_z[:-1]):
        raise ValueError("Optical surfaces cross inside the requested aperture")
    vertices, faces = revolve_curve(profile, n_ang, max_vertices, max_faces)
    sampled = max(sampled_surface_error(s, data[0], n_ang)
                  for s, data in zip((lens.s1, lens.s2), profiles))
    rim_sagitta = float(radius*(1-np.cos(np.pi/n_ang))) if clipped else 0.0
    sampled = max(sampled, rim_sagitta)
    rounding = float(np.linalg.norm(vertices.astype(np.float32).astype(float)-vertices, axis=1).max())
    if sampled+rounding > error:
        raise ValueError("Sampled triangle error plus binary STL rounding exceeds requested surface error")
    # Float32 conversion must preserve the actual exported topology as well.
    validate_mesh(vertices.astype(np.float32).astype(float), faces)
    report = dict(units="mm", vertices=len(vertices), faces=len(faces),
                  azimuth_segments=n_ang, profile_points=len(profile),
                  surface_error_requested_um=error*1000,
                  sampled_surface_error_um=sampled*1000,
                  stl_rounding_bound_um=rounding*1000,
                  rim_sagitta_bound_um=rim_sagitta*1000,
                  volume_mm3=signed_volume(vertices, faces),
                  diameter_mm=2*radius,
                  thickness_mm=lens.s2.t_shift-lens.s1.t_shift,
                  edge_thickness_mm=edge_thickness,
                  sampled_min_thickness_mm=max(0.0, float(np.min(back_z-front_z))),
                  full_intersection_diameter_mm=2*full_radius,
                  finite_rim=clipped)
    return vertices, faces, report

def compute_normal(p1, p2, p3):
    normal = np.cross(p2-p1, p3-p1)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("Degenerate triangle")
    return normal/norm

def export_to_stl(filename, vertices, faces, solid_name="cartesian_singlet"):
    """Write binary STL in chunks. STL is unitless; this project's CLI uses mm."""
    validate_mesh(vertices, faces)
    stored = np.asarray(vertices, dtype=np.float32)
    if not np.all(np.isfinite(stored)):
        raise ValueError("Coordinates cannot be represented by binary STL float32")
    validate_mesh(stored.astype(float), faces)
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    header = ("Cartesian oval; coordinates mm; "+solid_name).encode("ascii", "replace")[:80].ljust(80, b" ")
    with Path(filename).open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(faces)))
        for start in range(0, len(faces), 16384):
            triangle = stored[faces[start:start+16384]]
            normal = np.cross(triangle[:, 1].astype(float)-triangle[:, 0],
                              triangle[:, 2].astype(float)-triangle[:, 0])
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            records = np.zeros(len(triangle), dtype=dtype)
            records["normal"], records["vertices"] = normal, triangle
            stream.write(records.tobytes())

def export_to_stl_trimesh(filename, vertices, faces, solid_name="surface"):
    """Compatibility entry point; binary export no longer needs trimesh."""
    return export_to_stl(filename, vertices, faces, solid_name)
