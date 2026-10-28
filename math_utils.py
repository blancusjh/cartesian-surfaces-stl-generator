"""Bounded root helpers; failures never return an unvalidated last iterate."""
import numpy as np

def _value(f, x):
    result = float(f(x))
    if not np.isfinite(result):
        raise ValueError("Root function must return finite scalar values")
    return result

def biseccion(f, a, b, tol=1e-6, max_iter=100):
    if not np.all(np.isfinite([a, b, tol])) or a >= b or tol <= 0 or max_iter < 1:
        raise ValueError("Require finite a < b, positive tolerance and iteration budget")
    fa, fb = _value(f, a), _value(f, b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if np.signbit(fa) == np.signbit(fb):
        raise ValueError("Root must be bracketed by opposite signs")
    for _ in range(max_iter):
        c = a+(b-a)/2
        fc = _value(f, c)
        if fc == 0 or (b-a)/2 <= tol:
            return c
        if np.signbit(fa) != np.signbit(fc):
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise RuntimeError("Bisection did not converge within max_iter")

def biseccion_mod(f, a, b, tol=1e-9, max_iter=100, max_expand=32):
    if not np.all(np.isfinite([a, b, tol])) or a >= b or tol <= 0 or max_iter < 1 or max_expand < 0:
        raise ValueError("Invalid bracket, tolerance or iteration budget")
    fa, fb = _value(f, a), _value(f, b)
    for expansion in range(max_expand+1):
        if fa == 0:
            return a
        if fb == 0:
            return b
        if np.signbit(fa) != np.signbit(fb):
            return biseccion(f, a, b, tol, max_iter)
        if expansion < max_expand:
            b = a+1.5*(b-a)
            if not np.isfinite(b):
                raise ValueError("Bracket expansion overflow")
            fb = _value(f, b)
    raise ValueError("Could not bracket root within max_expand")

def newton_raphson_2d(F, x0, tol=1e-10, max_iter=50):
    """
    Multivariable Newton–Raphson for solving F(x) = 0 in R^2.

    Parameters:
        F        : function R^2 -> R^2, returns (f1, f2)
        x0       : tuple (s1, s2) initial guess
        tol      : convergence tolerance
        max_iter : maximum number of iterations

    Returns:
        (s1, s2) solution
    """
    x = np.array(x0, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)) or tol <= 0 or max_iter < 1:
        raise ValueError("Invalid Newton initial point, tolerance or iteration budget")
    
    def jacobian(F, x, h=1e-8):
        """Numerical Jacobian via finite differences."""
        J = np.zeros((2, 2))
        f0 = np.array(F(*x))
        for i in range(2):
            x_step = x.copy()
            x_step[i] += h
            fi = np.array(F(*x_step))
            J[:, i] = (fi - f0) / h
        return J

    for _ in range(max_iter):
        f_val = np.array(F(*x))
        if f_val.shape != (2,) or not np.all(np.isfinite(f_val)):
            raise ValueError("Newton residual must be two finite values")
        if np.linalg.norm(f_val, ord=2) < tol:
            return tuple(x)
        J = jacobian(F, x)
        try:
            delta = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            raise ValueError(f"Jacobian is singular at {x}")
        x += delta
        if not np.all(np.isfinite(x)):
            raise ValueError("Newton iterate is not finite")
        if np.linalg.norm(delta, ord=2) < tol and np.linalg.norm(F(*x)) < tol:
            return tuple(x)

    raise RuntimeError("Newton–Raphson did not converge.")

def intersection_between_curves(x1, y1, x2, y2, s1_0, s2_0):
    """
    Finds intersection between two parametric curves:
    Curve 1: (x1(s1), y1(s1))
    Curve 2: (x2(s2), y2(s2))

    Parameters:
        x1, y1 : functions of s1 for curve 1
        x2, y2 : functions of s2 for curve 2
        s1_0, s2_0 : initial guesses

    Returns:
        (s1, s2) intersection parameters
    """
    def F(s1, s2):
        return (
            x1(s1) - x2(s2),
            y1(s1) - y2(s2)
        )
    
    return newton_raphson_2d(F, (s1_0, s2_0))


