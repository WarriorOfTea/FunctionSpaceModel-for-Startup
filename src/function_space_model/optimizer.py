"""Exact MILP formulation for a finite action grid and finite model family.

    Bounds are numerical solver bounds for this restricted problem only.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .model import Action, CASH_TOLERANCE, ModelConfig, Simulation, simulate

MONEY_SCALE = 1000.0


@dataclass
class SearchResult:
    solver_status: str
    solver_message: str
    plan: list[Action] | None
    simulations: list[Simulation]
    lower_bound: float | None
    upper_bound: float | None
    relative_gap: float | None
    elapsed_seconds: float
    target_status: str
    scenario_names: tuple[str, ...]

    def summary(self, config: ModelConfig) -> dict:
        return {
            "synthetic": config.synthetic,
            "description": config.description,
            "configuration": asdict(config),
            "mode": "single_model" if len(self.scenario_names) == 1 else "maximin_fixed_plan",
            "scope": "Listed fixed response models and monthly action grid only; no probabilities",
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "scenario_names": list(self.scenario_names),
            "months": config.months,
            "profit_window_start_month": config.months - config.profit_window_months + 1,
            "profit_window_months": config.profit_window_months,
            "target_profit": config.target_profit,
            "target_status": self.target_status,
            "verified_plan_profit_lower_bound": self.lower_bound,
            "numerical_profit_upper_bound_on_grid": self.upper_bound,
            "solver_relative_gap": self.relative_gap,
            "verified_absolute_gap": (max(0.0, self.upper_bound - self.lower_bound)
                                      if self.upper_bound is not None and self.lower_bound is not None else None),
            "elapsed_seconds": self.elapsed_seconds,
            "cash_verification_tolerance": CASH_TOLERANCE,
            "simulations": [s.summary() for s in self.simulations],
            "plan": [asdict(a) for a in self.plan] if self.plan is not None else None,
        }


class _Rows:
    def __init__(self):
        self.rows, self.cols, self.values = [], [], []
        self.lower, self.upper = [], []

    def add(self, coefficients, lower=-np.inf, upper=np.inf):
        row = len(self.lower)
        for col, value in coefficients:
            if value:
                self.rows.append(row)
                self.cols.append(int(col))
                self.values.append(float(value))
        self.lower.append(lower)
        self.upper.append(upper)

    def constraint(self, variables):
        matrix = coo_array((np.asarray(self.values),
                            (np.asarray(self.rows, dtype=np.int32),
                             np.asarray(self.cols, dtype=np.int32))),
                           shape=(len(self.lower), variables)).tocsc()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def optimize(config: ModelConfig, *, scenario_names: tuple[str, ...] | None = None,
             time_limit: float = 30.0, relative_gap: float = 1e-5) -> SearchResult:
    if not isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    if not isfinite(relative_gap) or not 0 <= relative_gap < 1:
        raise ValueError("relative_gap must be in [0, 1)")
    names = tuple(s.name for s in config.scenarios) if scenario_names is None else scenario_names
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one unique scenario")
    lookup = {s.name: s for s in config.scenarios}
    if set(names) - lookup.keys():
        raise ValueError(f"Unknown scenarios: {sorted(set(names) - lookup.keys())}")
    scenarios = [lookup[name] for name in names]
    actions = config.actions
    T, J, S = config.months, len(actions), len(scenarios)
    acquired = np.zeros((S, J))
    retention = np.zeros((S, J))
    for s, scenario in enumerate(scenarios):
        for j, action in enumerate(actions):
            acquired[s, j], churn = scenario.response(action, config)
            retention[s, j] = 1 - churn

    # Select one action shared across models. Split N into selected-action parts
    # w <= U*y, sum(w)=N: binary y makes price/churn products exactly linear.
    offset = 0

    def indices(shape):
        nonlocal offset
        count = int(np.prod(shape))
        values = np.arange(offset, offset + count).reshape(shape)
        offset += count
        return values

    y = indices((T, J))
    N = indices((S, T + 1))
    B = indices((S, T + 1))
    w = indices((S, T, J))
    z = int(indices((1,))[0])
    lower, upper = np.zeros(offset), np.full(offset, np.inf)
    integrality = np.zeros(offset, dtype=np.int32)
    upper[y], integrality[y] = 1, 1
    lower[B] = config.cash_reserve / MONEY_SCALE
    lower[z] = -np.inf
    for s in range(S):
        limits = config.initial_customers + np.arange(T + 1) * acquired[s].max()
        if not np.all(np.isfinite(limits)):
            raise ValueError("Customer bounds overflow: reduce the numerical scale of inputs")
        upper[N[s]] = limits
        upper[w[s]] = limits[:-1, None]
        lower[N[s, 0]] = upper[N[s, 0]] = config.initial_customers
        lower[B[s, 0]] = upper[B[s, 0]] = config.initial_cash / MONEY_SCALE

    rows = _Rows()
    for t in range(T):
        rows.add([(y[t, j], 1) for j in range(J)], 1, 1)
    for s in range(S):
        for t in range(T):
            rows.add([(w[s, t, j], 1) for j in range(J)] + [(N[s, t], -1)], 0, 0)
            for j in range(J):
                rows.add([(w[s, t, j], 1), (y[t, j], -upper[N[s, t]])], upper=0)
            rows.add([(N[s, t + 1], 1)]
                     + [(w[s, t, j], -retention[s, j]) for j in range(J)]
                     + [(y[t, j], -acquired[s, j]) for j in range(J)], 0, 0)
            cash = [(B[s, t + 1], 1), (B[s, t], -1)]
            cash += [(w[s, t, j], -(a.price - config.service_cost_per_customer)
                      * retention[s, j] / MONEY_SCALE) for j, a in enumerate(actions)]
            cash += [(y[t, j], a.marketing / MONEY_SCALE) for j, a in enumerate(actions)]
            fixed = config.fixed_monthly_cost / MONEY_SCALE
            rows.add(cash, -fixed, -fixed)
            rows.add([(B[s, t], 1)]
                     + [(y[t, j], -a.marketing / MONEY_SCALE) for j, a in enumerate(actions)],
                     lower=(config.cash_reserve + config.fixed_monthly_cost) / MONEY_SCALE)
            rows.add([(w[s, t, j], retention[s, j]) for j in range(J)], upper=config.capacity)
        rows.add([(z, 1), (B[s, T], -1), (B[s, T - config.profit_window_months], 1)], upper=0)

    objective = np.zeros(offset)
    objective[z] = -1
    started = perf_counter()
    raw = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=rows.constraint(offset),
               options={"time_limit": time_limit, "mip_rel_gap": relative_gap})
    elapsed = perf_counter() - started
    statuses = {0: "optimal_within_tolerance", 1: "limit_reached", 2: "infeasible_on_grid",
                3: "unbounded", 4: "solver_error"}
    status = statuses.get(raw.status, "solver_error")
    dual = getattr(raw, "mip_dual_bound", None)
    bound = -float(dual) * MONEY_SCALE if dual is not None and isfinite(dual) else None
    gap = getattr(raw, "mip_gap", None)
    gap = float(gap) if gap is not None and isfinite(gap) else None
    plan, simulations, verified_profit = None, [], None
    if raw.x is not None:
        selected = np.asarray(raw.x[y])
        if np.all(np.abs(selected - np.round(selected)) <= 1e-5):
            candidate = [actions[int(np.argmax(selected[t]))] for t in range(T)]
            checks = [simulate(config, scenario, candidate) for scenario in scenarios]
            if all(check.feasible for check in checks):
                plan, simulations = candidate, checks
                verified_profit = min(check.window_profit for check in checks)
            else:
                status = "candidate_failed_verification"
                simulations = checks
        else:
            status = "candidate_failed_verification"

    if status == "candidate_failed_verification":
        target_status = "undetermined"
    elif verified_profit is not None and verified_profit >= config.target_profit:
        target_status = "met_by_verified_plan_for_listed_models"
    elif (status in ("optimal_within_tolerance", "limit_reached")
          and bound is not None and bound < config.target_profit - CASH_TOLERANCE):
        target_status = "below_target_on_action_grid_numerically"
    elif status == "infeasible_on_grid":
        target_status = "no_feasible_plan_on_action_grid_numerically"
    else:
        target_status = "undetermined"
    return SearchResult(status, str(raw.message), plan, simulations, verified_profit,
                        bound, gap, elapsed, target_status, tuple(names))
