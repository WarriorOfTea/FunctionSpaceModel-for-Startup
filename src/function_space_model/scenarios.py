"""Common-plan constraints for an explicitly enumerated disturbance set.

The objective remains the nominal-path derivative penalty. Extra scenarios
add feasibility constraints; they do not get independent decisions.
Fast quadrature predictions are checked against the separate ODE replay.
"""

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss, legvander

from .variational import Disturbance, _Problem, economics, replay


def standard_disturbances(config):
    return (Disturbance("development_delay_3_months", development_delay=.25),
            Disturbance("service_outage_1_month", outage_start=config.years / 2,
                        outage_duration=1 / 12, extra_churn_hazard=1., repair_cost=100_000.))


def partial_gauss_matrix(order=8):
    """Map values at Gauss nodes to integrals from -1 to each node."""
    nodes, _ = leggauss(order)
    integrated = np.column_stack([Legendre.basis(k).integ()(nodes)
                                  - Legendre.basis(k).integ()(-1) for k in range(order)])
    return integrated @ np.linalg.solve(legvander(nodes, order - 1), np.eye(order))


def linear_state(initial, hazard, acquisition, half_widths, weights, partial):
    """Integrating-factor quadrature for N'=acquisition-hazard*N.

    Arrays have shape (panels, Gauss order). Returns values at Gauss nodes
    and at panel boundaries, including the initial state.
    """
    increments_h = np.sum(hazard * weights, axis=1)
    ends_h = np.r_[0., np.cumsum(increments_h)]
    nodes_h = ends_h[:-1, None] + half_widths[:, None] * (hazard @ partial.T)
    transformed = np.exp(nodes_h) * acquisition
    increments_a = np.sum(transformed * weights, axis=1)
    ends_a = np.r_[0., np.cumsum(increments_a)]
    nodes_a = ends_a[:-1, None] + half_widths[:, None] * (transformed @ partial.T)
    return np.exp(-nodes_h) * (initial + nodes_a), np.exp(-ends_h) * (initial + ends_a)


class ScenarioProblem(_Problem):
    def __init__(self, config, segments, coordinate, disturbances):
        self.disturbances = tuple(disturbances)
        if not self.disturbances:
            raise ValueError("a scenario plan needs at least one disturbance")
        names = [d.name for d in self.disturbances]
        if len(set(names)) != len(names) or "nominal" in names:
            raise ValueError("scenario names must be unique and distinct from nominal")
        for d in self.disturbances:
            if d.outage_start + d.outage_duration > config.years + 1e-12:
                raise ValueError("outage extends beyond the horizon")
        super().__init__(config, segments, coordinate)
        self.partial = partial_gauss_matrix(8)
        events = []
        for d in self.disturbances:
            events.extend([d.outage_start, d.outage_start + d.outage_duration])
            events.extend(self.basis.breaks + d.development_delay)
        self.add_check_times(cash=[t for t in events if 0 <= t <= config.years])

    def add_check_times(self, controls=(), cash=()):
        super().add_check_times(controls, cash)
        self.delayed = [self.basis.matrix(np.maximum(0., self.times - d.development_delay))
                        for d in self.disturbances]
        self.outages = [(self.times >= d.outage_start)
                        & (self.times < d.outage_start + d.outage_duration) for d in self.disturbances]

    def predictions(self, free):
        c = self.config
        coefficients = self.coefficients(free)
        nominal = self.values(coefficients)
        e = economics(c, nominal)
        p = nominal["price"]
        half = np.diff(self.cash_times) / 2
        weights = self.weights.reshape(-1, 8)
        results = []
        for disturbance, delayed, outage in zip(self.disturbances, self.delayed, self.outages):
            q = delayed @ coefficients[1]
            # On feasible nominal paths a>=0, this is exactly the original
            # advertising response at the SAME planned budget and new quality.
            a = e["acquisition"] * ((1 + c.quality_acquisition_lift * q)
                                    / (1 + c.quality_acquisition_lift * nominal["quality"]))
            a = np.where(outage, 0., a)
            hazard = (c.base_churn_hazard * (1 - c.quality_churn_reduction * q)
                      * p / c.initial_price + outage * disturbance.extra_churn_hazard)
            n, ends_n = linear_state(c.initial_customers, hazard.reshape(-1, 8),
                                    a.reshape(-1, 8), half, weights, self.partial)
            repairs = (outage * disturbance.repair_cost / disturbance.outage_duration
                       if disturbance.outage_duration > 0 else np.zeros_like(p))
            profit = ((p - c.service_cost) * n.ravel() - e["marketing"] - e["development"]
                      - c.fixed_cost - repairs)
            cash = c.initial_cash + np.r_[0., np.cumsum(np.sum(profit.reshape(-1, 8) * weights, axis=1))]
            results.append({"name": disturbance.name, "customers": n.ravel(),
                            "customers_at_boundaries": ends_n, "cash": cash,
                            "window_profit": float(cash[-1] - cash[self.window_index])})
        return results

    def constraints(self, free):
        c = self.config
        result = [super().constraints(free)]
        for prediction in self.predictions(free):
            n = np.r_[prediction["customers"], prediction["customers_at_boundaries"]]
            result.extend([(prediction["cash"] - c.reserve) / 1e6,
                           np.array([(prediction["window_profit"] - c.target_profit) / 1e6]),
                           n / c.customer_scale, (c.capacity - n) / c.customer_scale])
        return np.concatenate(result)

    def additional_checks(self, plan):
        return [replay(self.config, plan, d)[0] for d in self.disturbances]
