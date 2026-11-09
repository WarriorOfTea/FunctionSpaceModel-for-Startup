"""Continuous cubic curves and exact quadratic derivative energies."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.optimize import minimize


def quadrature(breaks, order=8):
    """Gauss points/weights on each interval; no business time step is implied."""
    breaks = np.asarray(breaks, dtype=float)
    nodes, weights = leggauss(order)
    half = np.diff(breaks) / 2
    points = (breaks[:-1, None] + half[:, None] * (1 + nodes)).ravel()
    return points, (half[:, None] * weights).ravel()


@dataclass
class CubicBasis:
    horizon: float
    segments: int

    def __post_init__(self):
        if not math.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("horizon must be positive and finite")
        if type(self.segments) is not int or not 1 <= self.segments <= 100:
            raise ValueError("segments must be an integer in [1, 100]")
        self.breaks = np.linspace(0, self.horizon, self.segments + 1)
        self.knots = np.r_[np.repeat(0., 4), self.breaks[1:-1],
                           np.repeat(self.horizon, 4)]
        self.size = len(self.knots) - 4
        self.spline = BSpline(self.knots, np.eye(self.size), 3, extrapolate=False)
        self.greville = np.array([np.mean(self.knots[i + 1:i + 4])
                                  for i in range(self.size)])
        points, weights = quadrature(self.breaks, 4)
        derivatives = self.matrix(points, 1)
        self.energy = derivatives.T @ (weights[:, None] * derivatives)
        # Derivative coefficients are convex-combined quadratic B-splines.
        self.rate_matrix = np.zeros((self.size - 1, self.size))
        for i in range(self.size - 1):
            scale = 3 / (self.knots[i + 4] - self.knots[i + 1])
            self.rate_matrix[i, i:i + 2] = (-scale, scale)

    def matrix(self, times, derivative=0):
        return self.spline(times, nu=derivative)

    def linear(self, start, end):
        return start + (end - start) * self.greville / self.horizon


def endpoint_benchmark(*, horizon=3., segments=3, initial=100., terminal=1000.,
                       scale=1000., coordinate="absolute", pause=0.):
    """Numerically solve a convex endpoint problem, then compare to its bound.

    A pure pause leaves an active interval of horizon-pause. No financial
    interpretation is assigned to this isolated benchmark.
    """
    values = (horizon, initial, terminal, scale, pause)
    if not all(math.isfinite(v) for v in values) or scale <= 0:
        raise ValueError("benchmark inputs must be finite; scale must be positive")
    if not 0 <= pause < horizon:
        raise ValueError("pause must lie in [0, horizon)")
    if coordinate not in ("absolute", "relative"):
        raise ValueError("unknown customer coordinate")
    if coordinate == "relative" and min(initial, terminal) <= 0:
        raise ValueError("relative coordinate needs positive endpoints")
    start, end = ((initial / scale, terminal / scale) if coordinate == "absolute"
                  else (0., math.log(terminal / initial)))
    basis = CubicBasis(horizon - pause, segments)

    def coefficients(free):
        return np.r_[start, free, end]

    def objective(free):
        c = coefficients(free)
        return float(c @ basis.energy @ c)

    result = minimize(objective, np.repeat((start + end) / 2, basis.size - 2),
                      jac=lambda free: 2 * (basis.energy @ coefficients(free))[1:-1],
                      method="BFGS", options={"gtol": 1e-8, "maxiter": 500})
    c = coefficients(result.x)
    bound = (end - start) ** 2 / (horizon - pause)
    value = objective(result.x)
    times = np.linspace(0, horizon - pause, 101)
    error = float(np.max(np.abs(basis.matrix(times) @ c
                               - (start + (end - start) * times / (horizon - pause)))))
    return {"coordinate": coordinate, "segments": segments, "pause_years": pause,
            "analytic_minimum": bound, "computed_penalty": value,
            "absolute_error": abs(value - bound), "maximum_coordinate_error": error,
            "solver_success": bool(result.success), "solver_message": str(result.message),
            "analytic_check_passed": abs(value - bound) <= 1e-8 * max(1., bound)
            and error <= 1e-6}
