"""A continuous-path experiment, with explicit synthetic response assumptions.

N, quality and price are cubic splines (log N for the relative coordinate).
Marketing and development spending are inferred from their exact balances.
Nonlinear optimization and independent replay provide numerical candidates,
not a global optimum or a continuous-time feasibility certificate.
"""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline
from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar

from .curves import CubicBasis, quadrature


@dataclass(frozen=True)
class VariationalConfig:
    years: float = 3.
    profit_window: float = 1.
    target_profit: float = 10_000_000.
    initial_customers: float = 100.
    initial_quality: float = .2
    initial_price: float = 12_000.
    initial_cash: float = 3_000_000.
    reserve: float = 500_000.
    capacity: float = 5000.
    price_min: float = 6000.
    price_max: float = 18_000.
    price_rate_max: float = 6000.
    quality_max: float = 1.
    quality_rate_max: float = .5
    fixed_cost: float = 1_920_000.
    service_cost: float = 1200.
    advertising_scale: float = 1_200_000.
    advertising_max: float = 3_000_000.
    acquisition_at_scale: float = 600.
    quality_acquisition_lift: float = 1.
    base_churn_hazard: float = .6
    quality_churn_reduction: float = .7
    development_cost_per_quality: float = 600_000.
    customer_scale: float = 1000.
    quality_scale: float = 1.
    customer_weight: float = 1.
    quality_weight: float = 1.
    price_weight: float = 0.
    synthetic: bool = True
    description: str = "Illustrative continuous subscription model; not calibrated."

    def __post_init__(self):
        for field in fields(self):
            if field.name in ("synthetic", "description"):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        for name in ("years", "profit_window", "initial_price", "capacity", "price_min",
                     "advertising_scale", "acquisition_at_scale", "customer_scale",
                     "quality_scale", "customer_weight", "quality_weight"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (self.profit_window <= self.years
                and self.price_min <= self.initial_price <= self.price_max
                and self.initial_customers <= self.capacity
                and self.initial_quality <= self.quality_max <= 1
                and self.initial_cash >= self.reserve
                and self.quality_churn_reduction <= 1):
            raise ValueError("inconsistent initial state or bounds")
        if not isinstance(self.synthetic, bool) or not isinstance(self.description, str):
            raise ValueError("invalid provenance fields")

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ContinuousPlan:
    knots: tuple[float, ...]
    customers: tuple[float, ...]
    quality: tuple[float, ...]
    price: tuple[float, ...]
    coordinate: str = "absolute"

    def __post_init__(self):
        for name in ("knots", "customers", "quality", "price"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.coordinate not in ("absolute", "relative"):
            raise ValueError("unknown customer coordinate")
        size = len(self.knots) - 4
        if size < 4 or any(len(c) != size for c in (self.customers, self.quality, self.price)):
            raise ValueError("invalid cubic spline sizes")
        if not all(math.isfinite(v) for group in (self.knots, self.customers, self.quality, self.price)
                   for v in group):
            raise ValueError("nonfinite spline data")
        if any(a > b for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError("knots must be ordered")
        if not (self.knots[:4] == (0.,) * 4
                and self.knots[-4:] == (self.knots[-1],) * 4
                and self.knots[-1] > 0):
            raise ValueError("expected clamped knots starting at zero")
        if any(a >= b for a, b in zip(self.knots[3:-4], self.knots[4:-3])):
            raise ValueError("interior knots must be simple")

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def evaluate(self, config, times):
        c = BSpline(self.knots, self.customers, 3, extrapolate=False)
        q = BSpline(self.knots, self.quality, 3, extrapolate=False)
        p = BSpline(self.knots, self.price, 3, extrapolate=False)
        x, dx = c(times), c(times, nu=1)
        if self.coordinate == "absolute":
            n, dn = config.customer_scale * x, config.customer_scale * dx
        else:
            if config.initial_customers <= 0:
                raise ValueError("relative coordinate requires a positive initial base")
            n = config.initial_customers * np.exp(x)
            dn = n * dx
        return {"customers": n, "customer_rate": dn, "quality": q(times),
                "quality_rate": q(times, nu=1), "price": config.initial_price * p(times),
                "price_rate": config.initial_price * p(times, nu=1)}


def economics(config, values):
    """Inverse concave advertising response and paid quality development.

    a = a_scale*(1+lift*q)*sqrt(m/m_scale)*(p_ref/p)
    d = d0*(1-reduction*q)*(p/p_ref); q' = u/c_q.
    """
    n, dn, q, dq, p = (values[key] for key in
                        ("customers", "customer_rate", "quality", "quality_rate", "price"))
    hazard = config.base_churn_hazard * (1 - config.quality_churn_reduction * q) * p / config.initial_price
    acquisition = dn + hazard * n
    denominator = config.acquisition_at_scale * (1 + config.quality_acquisition_lift * q)
    marketing = config.advertising_scale * (acquisition * p / config.initial_price / denominator) ** 2
    development = config.development_cost_per_quality * dq
    profit = (p - config.service_cost) * n - marketing - development - config.fixed_cost
    return {"hazard": hazard, "acquisition": acquisition, "marketing": marketing,
            "development": development, "profit": profit}


def penalty_lower_bound(config, coordinate):
    """Optimistic analytic bound from profit <= maximal margin*N-fixed cost."""
    margin = config.price_max - config.service_cost
    if margin <= 0:
        return {"value": None, "reason": "nonpositive maximal contribution margin"}
    average_needed = ((config.target_profit / config.profit_window + config.fixed_cost)
                      / margin)
    if coordinate == "absolute":
        value = (config.customer_weight * (max(average_needed - config.initial_customers, 0)
                 / config.customer_scale) ** 2 / (config.years - 2 * config.profit_window / 3))
    elif config.initial_customers > 0:
        value = (config.customer_weight * max(math.log(average_needed / config.initial_customers), 0) ** 2
                 / config.years) if average_needed > 0 else 0.
    else:
        raise ValueError("relative coordinate requires positive initial customers")
    return {"value": float(value), "required_average_customers_optimistic": average_needed,
            "scope": "lower bound for every admissible continuous path under this configuration"}


def control_extrema(config, plan):
    """Numerical extrema: a fine grid, then local refinement around each peak."""
    times = np.linspace(0, config.years, 2401)
    values = plan.evaluate(config, times)
    e = economics(config, values)
    extrema = {}
    for key, sign in (("marketing", -1), ("acquisition", 1)):
        data = sign * e[key]
        best = int(np.argmin(data))
        best_t, best_value = float(times[best]), float(data[best])
        local = np.flatnonzero((data[1:-1] < data[:-2]) & (data[1:-1] <= data[2:])) + 1
        # Roundoff on a constant control is not a collection of real peaks.
        tolerance = 1e-10 * max(1., float(np.max(np.abs(data))))
        local = local[(data[local - 1] - data[local] > tolerance)
                      | (data[local + 1] - data[local] > tolerance)]
        if 0 < best < len(data) - 1 and np.ptp(data) > tolerance:
            local = np.union1d(local, [best])
        for i in local:
            fit = minimize_scalar(lambda t: float(sign * economics(config, plan.evaluate(config, t))[key]),
                                  bounds=(times[i - 1], times[i + 1]), method="bounded",
                                  options={"xatol": 1e-11})
            if fit.fun < best_value:
                best_t, best_value = float(fit.x), float(fit.fun)
        extrema[key] = {"time": best_t, "value": sign * best_value}
    return values, e, extrema


class _Problem:
    def __init__(self, config, segments, coordinate):
        self.config, self.coordinate = config, coordinate
        self.basis = b = CubicBasis(config.years, segments)
        if coordinate == "relative" and config.initial_customers <= 0:
            raise ValueError("relative coordinate requires positive initial customers")
        if coordinate not in ("absolute", "relative"):
            raise ValueError("unknown customer coordinate")
        self.initial = np.array([config.initial_customers / config.customer_scale
                                 if coordinate == "absolute" else 0., config.initial_quality, 1.])
        self.cash_times = np.unique(np.r_[np.linspace(0, config.years, 8 * segments + 1),
                                         config.years - config.profit_window])
        self.times, self.weights = quadrature(self.cash_times, 8)
        self.check_times = np.unique(np.r_[self.times, self.cash_times])
        self.matrix = b.matrix(self.times)
        self.derivative = b.matrix(self.times, 1)
        self.check_matrix = b.matrix(self.check_times)
        self.check_derivative = b.matrix(self.check_times, 1)
        self.window_index = int(np.searchsorted(self.cash_times, config.years - config.profit_window))
        count = b.size - 1
        lower_n = 0. if coordinate == "absolute" else -np.inf
        upper_n = (config.capacity / config.customer_scale if coordinate == "absolute"
                   else math.log(config.capacity / config.initial_customers))
        self.bounds = Bounds(np.r_[np.repeat(lower_n, count), np.repeat(0., count),
                                  np.repeat(config.price_min / config.initial_price, count)],
                             np.r_[np.repeat(upper_n, count), np.repeat(config.quality_max, count),
                                   np.repeat(config.price_max / config.initial_price, count)])
        linear = np.zeros((2 * count, 3 * count))
        linear[:count, count:2 * count] = b.rate_matrix[:, 1:]
        linear[count:, 2 * count:] = b.rate_matrix[:, 1:]
        offset = np.r_[b.rate_matrix[:, 0] * config.initial_quality, b.rate_matrix[:, 0]]
        rate = config.price_rate_max / config.initial_price
        self.linear = LinearConstraint(linear, np.r_[np.zeros(count), np.repeat(-rate, count)] - offset,
                                       np.r_[np.repeat(config.quality_rate_max, count),
                                             np.repeat(rate, count)] - offset)

    def coefficients(self, free):
        return np.column_stack((self.initial, np.asarray(free).reshape(3, -1)))

    def add_check_times(self, controls=(), cash=()):
        self.cash_times = np.unique(np.r_[self.cash_times, cash])
        self.times, self.weights = quadrature(self.cash_times, 8)
        self.matrix = self.basis.matrix(self.times)
        self.derivative = self.basis.matrix(self.times, 1)
        self.check_times = np.unique(np.r_[self.check_times, self.times, self.cash_times, controls])
        self.check_matrix = self.basis.matrix(self.check_times)
        self.check_derivative = self.basis.matrix(self.check_times, 1)
        self.window_index = int(np.searchsorted(self.cash_times,
                                               self.config.years - self.config.profit_window))

    def values(self, coefficients, check=False):
        matrix, derivative = ((self.check_matrix, self.check_derivative) if check
                              else (self.matrix, self.derivative))
        x, q, price = coefficients @ matrix.T
        dx, dq, dp = coefficients @ derivative.T
        c = self.config
        n = c.customer_scale * x if self.coordinate == "absolute" else c.initial_customers * np.exp(x)
        dn = c.customer_scale * dx if self.coordinate == "absolute" else n * dx
        return {"customers": n, "customer_rate": dn, "quality": q, "quality_rate": dq,
                "price": c.initial_price * price, "price_rate": c.initial_price * dp}

    def components(self, free):
        coefficients = self.coefficients(free)
        c = self.config
        return {name: float(weight * row @ self.basis.energy @ row) for name, weight, row in
                (("customers", c.customer_weight, coefficients[0]),
                 ("quality", c.quality_weight / c.quality_scale ** 2, coefficients[1]),
                 ("price", c.price_weight, coefficients[2]))}

    def objective(self, free):
        return sum(self.components(free).values())

    def gradient(self, free):
        c = self.config
        coefficients = self.coefficients(free)
        weights = (c.customer_weight, c.quality_weight / c.quality_scale ** 2, c.price_weight)
        return np.concatenate([2 * weight * (self.basis.energy @ row)[1:]
                               for weight, row in zip(weights, coefficients)])

    def constraints(self, free):
        c = self.config
        coefficients = self.coefficients(free)
        e = economics(c, self.values(coefficients))
        increments = (e["profit"] * self.weights).reshape(-1, 8).sum(axis=1)
        cash = c.initial_cash + np.r_[0., np.cumsum(increments)]
        window_profit = cash[-1] - cash[self.window_index]
        checked = economics(c, self.values(coefficients, check=True))
        return np.r_[(cash - c.reserve) / 1e6, (window_profit - c.target_profit) / 1e6,
                     checked["acquisition"] / c.acquisition_at_scale,
                     (c.advertising_max - checked["marketing"]) / c.advertising_scale]

    def seed(self, factor):
        c, b = self.config, self.basis
        average = (c.target_profit / c.profit_window + c.fixed_cost) / max(c.price_max - c.service_cost, 1.)
        terminal = min(c.capacity, max(c.initial_customers, factor * average))
        end = terminal / c.customer_scale if self.coordinate == "absolute" else math.log(terminal / c.initial_customers)
        qend = min(c.quality_max, c.initial_quality + .2 * factor,
                   c.initial_quality + c.quality_rate_max * c.years)
        pend = min(c.price_max / c.initial_price,
                   1 + c.price_rate_max * c.years / c.initial_price)
        return np.r_[b.linear(self.initial[0], end)[1:],
                     b.linear(c.initial_quality, qend)[1:], b.linear(1., pend)[1:]]

    def plan(self, free):
        n, q, p = self.coefficients(free)
        return ContinuousPlan(tuple(self.basis.knots), tuple(n), tuple(q), tuple(p), self.coordinate)


@dataclass(frozen=True)
class Disturbance:
    name: str = "nominal"
    development_delay: float = 0.
    outage_start: float = 0.
    outage_duration: float = 0.
    extra_churn_hazard: float = 0.
    repair_cost: float = 0.

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("disturbance needs a name")
        for name in ("development_delay", "outage_start", "outage_duration", "extra_churn_hazard", "repair_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if self.outage_duration == 0 and (self.extra_churn_hazard or self.repair_cost):
            raise ValueError("incident costs and extra churn need a positive outage duration")


def replay(config, plan, disturbance=Disturbance(), *, samples=601):
    """Independent adaptive ODE replay of FIXED planned price/ad/development.

    Break integration at event edges, knots, delayed knots and profit-window
    edge, so even short incidents are not skipped by adaptive integration.
    """
    if type(samples) is not int or samples < 3:
        raise ValueError("samples must be an integer >=3")
    if not math.isclose(plan.knots[-1], config.years):
        raise ValueError("plan horizon differs from configuration")
    if disturbance.outage_start + disturbance.outage_duration > config.years + 1e-12:
        raise ValueError("outage extends beyond the horizon")
    c, d = config, disturbance
    values, e, extrema = control_extrema(c, plan)
    if (not all(np.all(np.isfinite(v)) for v in (*values.values(), *e.values()))
            or np.min(values["customers"]) < -1e-5
            or np.max(values["customers"]) > c.capacity + 1e-5
            or np.min(values["quality"]) < -1e-7
            or np.max(values["quality"]) > c.quality_max + 1e-7
            or np.min(values["quality_rate"]) < -1e-7
            or np.max(values["quality_rate"]) > c.quality_rate_max + 1e-7
            or np.min(values["price"]) < c.price_min - .01
            or np.max(values["price"]) > c.price_max + .01
            or np.max(np.abs(values["price_rate"])) > c.price_rate_max + .01
            or extrema["acquisition"]["value"] < -1e-5
            or extrema["marketing"]["value"] > c.advertising_max + .05):
        raise ValueError("planned path violates numerical physical/control checks")
    initial = plan.evaluate(c, 0.)
    if max(abs(float(initial[k]) - expected) for k, expected in
           (("customers", c.initial_customers), ("quality", c.initial_quality),
            ("price", c.initial_price))) > 1e-6:
        raise ValueError("plan initial state differs from configuration")
    spline_n = BSpline(plan.knots, plan.customers, 3, extrapolate=False)
    spline_q = BSpline(plan.knots, plan.quality, 3, extrapolate=False)
    spline_p = BSpline(plan.knots, plan.price, 3, extrapolate=False)

    def controls(t):
        x, dx, q, dq, pp = (spline_n(t), spline_n(t, 1), spline_q(t), spline_q(t, 1), spline_p(t))
        n = c.customer_scale * x if plan.coordinate == "absolute" else c.initial_customers * np.exp(x)
        dn = c.customer_scale * dx if plan.coordinate == "absolute" else n * dx
        values = {"customers": n, "customer_rate": dn, "quality": q, "quality_rate": dq,
                  "price": c.initial_price * pp}
        e = economics(c, values)
        return float(values["price"]), float(e["marketing"]), float(e["development"])

    edge = c.years - c.profit_window
    boundaries = np.unique(np.r_[0., c.years, edge, plan.knots,
                                 np.asarray(plan.knots) + d.development_delay,
                                 d.outage_start, d.outage_start + d.outage_duration])
    boundaries = boundaries[(boundaries >= 0) & (boundaries <= c.years)]
    state = np.array([c.initial_customers, c.initial_cash, 0., 0., 0.])
    pieces = []
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left + right) / 2
        outage = d.outage_start <= midpoint < d.outage_start + d.outage_duration
        window = midpoint >= edge

        def rhs(t, y):
            p, m, u = controls(t)
            q = float(spline_q(max(0., t - d.development_delay)))
            a = (0. if outage else c.acquisition_at_scale * (1 + c.quality_acquisition_lift * q)
                 * math.sqrt(max(m, 0.) / c.advertising_scale) * c.initial_price / p)
            hazard = c.base_churn_hazard * (1 - c.quality_churn_reduction * q) * p / c.initial_price
            if outage:
                hazard += d.extra_churn_hazard
            repair = d.repair_cost / d.outage_duration if outage else 0.
            profit = (p - c.service_cost) * y[0] - m - u - c.fixed_cost - repair
            return [a - hazard * y[0], profit, profit if window else 0., m, u]

        solution = solve_ivp(rhs, (left, right), state, method="DOP853", rtol=2e-10,
                             atol=1e-7, dense_output=True, max_step=(right - left) / 4)
        if not solution.success or not np.all(np.isfinite(solution.y)):
            raise ValueError(f"ODE replay failed: {solution.message}")
        state = solution.y[:, -1]
        pieces.append((left, right, solution.sol))

    times = np.unique(np.r_[np.linspace(0, c.years, samples), boundaries])
    states = np.empty((5, len(times)))
    minimum_cash = c.initial_cash
    minimum_cash_time = 0.
    maximum_customers = c.initial_customers
    for left, right, solution in pieces:
        mask = (times >= left) & (times <= right)
        states[:, mask] = solution(times[mask])
        # Independent interior minimization on small subintervals: numerical,
        # not an interval-arithmetic proof of feasibility.
        for a, b in zip(np.linspace(left, right, 9), np.linspace(left, right, 9)[1:]):
            result = minimize_scalar(lambda t: float(solution(t)[1]), bounds=(a, b), method="bounded",
                                     options={"xatol": 1e-10})
            for candidate_time, candidate_cash in ((result.x, result.fun), (a, solution(a)[1]), (b, solution(b)[1])):
                if candidate_cash < minimum_cash:
                    minimum_cash, minimum_cash_time = float(candidate_cash), float(candidate_time)
        maximum_customers = max(maximum_customers, float(np.max(solution(np.linspace(left, right, 65))[0])))
    planned = plan.evaluate(c, times)
    planned_economics = economics(c, planned)
    final_profit = float(states[2, -1])
    feasible = minimum_cash >= c.reserve - .05 and maximum_customers <= c.capacity + 1e-5
    summary = {"name": d.name, "disturbance": asdict(d), "feasible_cash_and_capacity": bool(feasible),
               "target_met": final_profit >= c.target_profit - .05,
               "window_profit": final_profit, "target_shortfall": max(c.target_profit - final_profit, 0.),
               "final_customers": float(states[0, -1]), "final_quality": float(spline_q(max(0., c.years - d.development_delay))),
               "minimum_cash": float(minimum_cash), "final_cash": float(states[1, -1]),
               "minimum_cash_time": minimum_cash_time,
               "marketing_spent": float(states[3, -1]), "development_spent": float(states[4, -1]),
               "required_initial_cash_for_this_plan": float(c.initial_cash + c.reserve - minimum_cash),
               "maximum_customers": maximum_customers,
               "maximum_customer_replay_error": float(np.max(np.abs(states[0] - planned["customers"]))) if d == Disturbance() else None,
               "verification": "adaptive ODE replay and numerical interior checks; not a formal certificate"}
    rows = [{"time_years": float(t), "customers": float(states[0, i]),
             "quality": float(spline_q(max(0., t - d.development_delay))),
             "price_per_year": float(planned["price"][i]), "cash": float(states[1, i]),
             "marketing_per_year": float(planned_economics["marketing"][i]),
             "development_per_year": float(planned_economics["development"][i])}
            for i, t in enumerate(times)]
    return summary, rows


def optimize_continuous(config, *, segments=3, coordinate="absolute", starts=3, maxiter=400):
    """Return the best independently checked candidate from deterministic starts."""
    if type(starts) is not int or not 1 <= starts <= 20 or type(maxiter) is not int or maxiter < 1:
        raise ValueError("invalid search limits")
    started = time.monotonic()
    problem = _Problem(config, segments, coordinate)
    attempts, candidates = [], []
    for factor in np.linspace(1.1, 1.8, starts):
        seed = problem.seed(float(factor))
        refinements = []
        for _ in range(6):
            result = minimize(problem.objective, seed, method="SLSQP",
                              jac=problem.gradient, bounds=problem.bounds,
                              constraints=[problem.linear, {"type": "ineq", "fun": problem.constraints}],
                              options={"ftol": 1e-10, "maxiter": maxiter})
            if np.min(problem.constraints(result.x)) < -1e-7:
                break
            plan = problem.plan(result.x)
            _, _, extrema = control_extrema(config, plan)
            excess = extrema["marketing"]["value"] - config.advertising_max
            shortage = -extrema["acquisition"]["value"]
            if excess > .025 or shortage > 5e-6:
                refinements.append({"marketing_excess": excess, "negative_acquisition": shortage})
                problem.add_check_times(controls=[extrema["marketing"]["time"], extrema["acquisition"]["time"]])
                seed = result.x
                continue
            try:
                checked, _ = replay(config, plan)
            except ValueError:
                break
            if checked["minimum_cash"] < config.reserve - .025:
                refinements.append({"cash_shortfall": config.reserve - checked["minimum_cash"]})
                problem.add_check_times(cash=[checked["minimum_cash_time"]])
                seed = result.x
                continue
            break
        margin = float(np.min(problem.constraints(result.x)))
        linear = problem.linear.A @ result.x
        linear_margin = float(min(np.min(linear - problem.linear.lb), np.min(problem.linear.ub - linear)))
        attempt = {"solver_success": bool(result.success), "solver_message": str(result.message),
                   "iterations": int(result.nit), "penalty": float(result.fun),
                   "minimum_scaled_constraint": margin, "minimum_rate_constraint": linear_margin,
                   "constraint_refinements": refinements}
        attempts.append(attempt)
        if margin < -1e-7 or linear_margin < -1e-7:
            attempt["accepted"] = False
            continue
        plan = problem.plan(result.x)
        try:
            check, _ = replay(config, plan)
        except ValueError as error:
            attempt.update(accepted=False, validation_error=str(error))
            continue
        # Dense validation supplements the optimizer's quadrature constraints.
        allowed = (check["feasible_cash_and_capacity"] and check["target_met"]
                   and check["maximum_customer_replay_error"] <= .01)
        attempt["accepted"] = bool(allowed)
        attempt["replay"] = check
        if allowed:
            candidates.append((float(result.fun), plan, result.x))
    candidates.sort(key=lambda item: item[0])
    bound = penalty_lower_bound(config, coordinate)
    report = {"mode": "continuous_variational", "synthetic": config.synthetic,
              "configuration": asdict(config), "coordinate": coordinate, "segments": segments,
              "status": "numerically_checked_candidate" if candidates else "no_verified_candidate",
              "global_optimality_proven": False, "analytic_lower_bound": bound,
              "attempts": attempts, "elapsed_seconds": time.monotonic() - started,
              "plan": asdict(candidates[0][1]) if candidates else None}
    if not candidates:
        report["interpretation"] = "Search failure is not proof of infeasibility."
        return report, []
    value, plan, free = candidates[0]
    nominal, rows = replay(config, plan)
    values, e, extrema = control_extrema(config, plan)
    report.update(penalty=value, penalty_components=problem.components(free), nominal=nominal,
                  gap_to_optimistic_lower_bound=value - bound["value"] if bound["value"] is not None else None,
                  maximum_absolute_customer_rate=float(np.max(np.abs(values["customer_rate"]))),
                  maximum_quality_rate=float(np.max(values["quality_rate"])),
                  maximum_price_rate=float(np.max(np.abs(values["price_rate"]))),
                  maximum_marketing_rate=extrema["marketing"]["value"],
                  tolerances={"money": .05, "customers": .01,
                              "note": "numerical tolerances, not a formal feasibility proof"})
    return report, rows
