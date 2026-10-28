"""Cartesian ovals, using signed axial focal positions and exact optical paths."""
import numpy as np
from scipy.optimize import least_squares

def sigma(z0, zi, rho, n0, ni):
    """Local (z,r) on the near-axis Cartesian-oval branch.

    The object is at -z0 and the image at zi. Negative distances denote
    virtual conjugates. All lengths must use the same unit.
    """
    values = np.asarray([z0, zi, n0, ni], dtype=float)
    if not np.all(np.isfinite(values)) or values[2] <= 0 or values[3] <= 0 or values[2] == values[3]:
        raise ValueError("Finite conjugates and distinct positive refractive indices required")
    z0, zi, n0, ni = values
    z0 = -z0
    if z0 == 0 or zi == 0 or ni*zi-n0*z0 == 0 or ni*z0-n0*zi == 0:
        raise ValueError("Singular conjugates for this Cartesian-oval formulation")
    rho = np.asarray(rho, dtype=float)
    if (not np.all(np.isfinite(rho))) or np.any(rho < 0):
        raise ValueError("rho must be finite and nonnegative")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        G = (ni**2*zi - n0**2*z0)**2 / (ni*n0*(ni*zi-n0*z0)*(ni*z0-n0*zi))
        O = (ni*z0-n0*zi) / (zi*z0*(ni-n0))
        T = (ni-n0)*(ni+n0)**2 / (4*ni*n0*zi*z0*(ni*zi-n0*z0))
        S = (ni+n0)*(ni**2*zi-n0**2*z0) / (2*ni*n0*zi*z0*(ni*zi-n0*z0))
        if not np.all(np.isfinite([G, O, T, S])):
            raise ValueError("Optical parameters exceed the numerical formulation range")
        z = (O+T*rho**2)*rho**2 / (1+S*rho**2+np.sqrt(1+(2*S-O**2*G)*rho**2))
        r = np.sqrt(rho**2-z**2)
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(r)):
        raise ValueError("Requested point is outside the real near-axis oval branch")
    return z, r

class SigmaCurve:
    def __init__(self, z0, zi, n0, ni, rho_points=60, t_shift=0):
        values = np.asarray([z0, zi, n0, ni, t_shift], dtype=float)
        if not np.all(np.isfinite(values)) or n0 <= 0 or ni <= 0 or n0 == ni:
            raise ValueError("Finite conjugates and distinct positive refractive indices required")
        if not isinstance(rho_points, (int, np.integer)) or rho_points < 2:
            raise ValueError("rho_points must be an integer >= 2")
        self.z0, self.zi, self.n0, self.ni, self.t_shift = map(float, values)
        self.rho_points = int(rho_points)
        # Object coordinate is -z0; image coordinate is zi, both absolute.
        self.local_z0, self.local_zi = self.z0+self.t_shift, self.zi-self.t_shift
        a, b = -self.local_z0, self.local_zi
        if a == 0 or b == 0 or ni*b-n0*a == 0 or ni*a-n0*b == 0:
            raise ValueError("Singular conjugates for this Cartesian-oval formulation")
        sigma(self.local_z0, self.local_zi, 0.0, self.n0, self.ni)
        self.rho_end = None

    def evaluate(self, rho):
        z, r = sigma(self.local_z0, self.local_zi, rho, self.n0, self.ni)
        return np.stack((z+self.t_shift, r), axis=-1)

    def get_points(self, rhof=None):
        if rhof is None:
            rhof = self.rho_end
        if rhof is None or not np.isfinite(rhof) or rhof <= 0:
            raise ValueError("Supply a positive rhof, or construct an OmegaLens first")
        return self.evaluate(np.linspace(0, rhof, self.rho_points))

    @property
    def points(self):
        return self.get_points()

    def lambdify(self):
        return (lambda rho: self.evaluate(rho)[..., 0],
                lambda rho: self.evaluate(rho)[..., 1])

    def optical_residual(self, z, r):
        """Signed optical-path change from the vertex, without large-distance cancellation."""
        local_z = np.asarray(z)-self.t_shift
        result = np.zeros_like(local_z+np.asarray(r), dtype=float)
        for focus, index, sign in ((-self.local_z0, self.n0, np.sign(self.local_z0)),
                                    (self.local_zi, self.ni, np.sign(self.local_zi))):
            distance = np.hypot(local_z-focus, r)
            delta = (local_z**2-2*local_z*focus+np.asarray(r)**2)/(distance+abs(focus))
            result = result+index*sign*delta
        return result

class OmegaLens:
    def __init__(self, sigma1, sigma2):
        self.s1, self.s2 = sigma1, sigma2
        width = sigma2.t_shift-sigma1.t_shift
        if width <= 0:
            raise ValueError("Second surface vertex must follow the first (positive thickness)")
        k1 = (sigma1.n0/sigma1.local_z0+sigma1.ni/sigma1.local_zi)/(2*(sigma1.ni-sigma1.n0))
        k2 = (sigma2.n0/sigma2.local_z0+sigma2.ni/sigma2.local_zi)/(2*(sigma2.ni-sigma2.n0))
        if k1 <= k2:
            raise ValueError("Near-axis surfaces do not enclose the requested lens")
        radius_scale = np.sqrt(width/(k1-k2))
        def residual(x):
            z, r = sigma1.t_shift+x[0]*width, x[1]*radius_scale
            return np.array([sigma1.optical_residual(z, r),
                             sigma2.optical_residual(z, r)])/width
        solution = least_squares(residual, [0.5, 1.0], bounds=([0, 1e-8], [1, 100]),
                                 xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200)
        if not solution.success or np.max(np.abs(residual(solution.x))) > 1e-9:
            raise ValueError("No converged physical intersection between the two vertices")
        z = sigma1.t_shift+solution.x[0]*width
        r = solution.x[1]*radius_scale
        self.rho_intersection = tuple(np.hypot(z-s.t_shift, r) for s in (sigma1, sigma2))
        point = np.array([z, r])
        for surface, rho in zip((sigma1, sigma2), self.rho_intersection):
            if not np.allclose(surface.evaluate(rho), point, rtol=1e-9, atol=width*1e-9):
                raise ValueError("Intersection lies outside the near-axis analytic branch")
            probe = surface.evaluate(np.linspace(0, rho, 257))
            if np.any(np.diff(probe[:, 1]) <= 0):
                raise ValueError("Lens requires a monotonic radial surface branch")
            surface.rho_end = rho
        self.intersection_point = tuple(point)
        self.curve = self._generate_inner_curve(*self.rho_intersection)

    def _generate_inner_curve(self, rho_i1, rho_i2):
        first, second = self.s1.get_points(rho_i1), self.s2.get_points(rho_i2)
        first[-1] = self.intersection_point
        # Intersection has a single shared ring; poles are exact axis points.
        return np.vstack((first, second[-2::-1]))

    def get_points(self):
        return self.curve
