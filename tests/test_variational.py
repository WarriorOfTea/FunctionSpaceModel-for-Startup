from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from function_space_model.cli import main
from function_space_model.curves import CubicBasis, endpoint_benchmark
from function_space_model.variational import (ContinuousPlan, Disturbance, VariationalConfig,
                                             economics, optimize_continuous, penalty_lower_bound, replay)


def analytic_config(**changes):
    return replace(VariationalConfig(), years=1, profit_window=.4, target_profit=0,
                   initial_customers=10, initial_price=1000, initial_cash=1e6, reserve=100,
                   price_min=500, price_max=1500, fixed_cost=1000, service_cost=100,
                   advertising_scale=10000, advertising_max=1e6, acquisition_at_scale=100,
                   base_churn_hazard=0, customer_scale=100, **changes)


def linear_plan(config, terminal=110, final_quality=None, coordinate="absolute"):
    b = CubicBasis(config.years, 2)
    if final_quality is None:
        final_quality = config.initial_quality
    endpoints = ((config.initial_customers / config.customer_scale, terminal / config.customer_scale)
                 if coordinate == "absolute" else (0., math.log(terminal / config.initial_customers)))
    return ContinuousPlan(tuple(b.knots), tuple(b.linear(*endpoints)),
                          tuple(b.linear(config.initial_quality, final_quality)),
                          tuple(b.linear(1, 1)), coordinate)


class VariationalMathematicsTests(unittest.TestCase):
    def test_endpoint_minima_and_pauses_match_analytic_bounds(self):
        for coordinate in ("absolute", "relative"):
            for segments in (1, 3, 6):
                for pause in (0., .25):
                    with self.subTest(coordinate=coordinate, segments=segments, pause=pause):
                        r = endpoint_benchmark(coordinate=coordinate, segments=segments, pause=pause)
                        self.assertTrue(r["analytic_check_passed"], r)

    def test_normalization_is_invariant_to_customer_unit(self):
        first = endpoint_benchmark(initial=100, terminal=1000, scale=1000)
        second = endpoint_benchmark(initial=1, terminal=10, scale=10)
        self.assertAlmostEqual(first["computed_penalty"], second["computed_penalty"], places=10)

    def test_derivative_control_bounds_hold_between_nodes(self):
        b = CubicBasis(3, 4)
        coefficients = b.linear(.2, .8)
        derivative_coefficients = b.rate_matrix @ coefficients
        derivatives = b.matrix(np.linspace(0, 3, 1001), 1) @ coefficients
        self.assertTrue(np.all(derivatives >= min(derivative_coefficients) - 1e-12))
        self.assertTrue(np.all(derivatives <= max(derivative_coefficients) + 1e-12))

    def test_optimistic_profit_bound_is_below_known_feasible_path(self):
        c = analytic_config()
        plan = linear_plan(c)
        check, _ = replay(c, plan)
        c = replace(c, target_profit=check["window_profit"])
        lower = penalty_lower_bound(c, "absolute")["value"]
        known_energy = ((110 - 10) / c.customer_scale) ** 2 / c.years
        self.assertGreater(lower, 0)
        self.assertLessEqual(lower, known_energy)

    def test_relative_coordinate_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            endpoint_benchmark(initial=0, coordinate="relative")
        for changes in ({"initial_cash": 1}, {"quality_max": 1.2},
                        {"price_min": 0}, {"quality_weight": 0}, {"years": math.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(VariationalConfig(), **changes)
        with self.assertRaises(ValueError):
            optimize_continuous(replace(VariationalConfig(), initial_customers=0), coordinate="relative")


class ContinuousReplayTests(unittest.TestCase):
    def test_linear_path_matches_closed_form_customers_cash_and_window(self):
        c = analytic_config()
        plan = linear_plan(c)
        result, rows = replay(c, plan)
        marketing = c.advertising_scale / (1 + c.quality_acquisition_lift * c.initial_quality) ** 2

        def cash(t):
            return c.initial_cash + (c.initial_price - c.service_cost) * (10 * t + 50 * t * t) - (marketing + c.fixed_cost) * t

        self.assertLess(result["maximum_customer_replay_error"], 1e-6)
        self.assertAlmostEqual(result["final_customers"], 110, places=6)
        self.assertAlmostEqual(result["final_cash"], cash(1), places=5)
        self.assertAlmostEqual(result["window_profit"], cash(1) - cash(.6), places=5)
        self.assertAlmostEqual(result["marketing_spent"], marketing, places=5)
        self.assertAlmostEqual(rows[0]["cash"], c.initial_cash)

    def test_short_outage_is_not_skipped_and_controls_remain_fixed(self):
        c = analytic_config()
        plan = linear_plan(c)
        nominal, _ = replay(c, plan)
        event = Disturbance("short", outage_start=.431, outage_duration=.0001, repair_cost=100)
        stressed, _ = replay(c, plan, event)
        self.assertAlmostEqual(stressed["final_customers"], nominal["final_customers"] - .01, places=6)
        self.assertAlmostEqual(stressed["marketing_spent"], nominal["marketing_spent"], places=5)
        self.assertLess(stressed["final_cash"], nominal["final_cash"] - 100)

    def test_delayed_quality_keeps_actual_spending_calendar(self):
        c = analytic_config()
        plan = linear_plan(c, final_quality=.5)
        nominal, _ = replay(c, plan)
        stressed, _ = replay(c, plan, Disturbance("delay", development_delay=.25))
        self.assertAlmostEqual(stressed["final_quality"], .425, places=7)
        self.assertAlmostEqual(stressed["marketing_spent"], nominal["marketing_spent"], places=5)
        self.assertAlmostEqual(stressed["development_spent"], nominal["development_spent"], places=5)
        self.assertLess(stressed["final_customers"], nominal["final_customers"])

    def test_negative_acquisition_and_invalid_events_are_rejected(self):
        c = analytic_config()
        with self.assertRaises(ValueError):
            replay(c, linear_plan(c, terminal=1))
        with self.assertRaises(ValueError):
            Disturbance(repair_cost=1)
        with self.assertRaises(ValueError):
            replay(c, linear_plan(c), Disturbance(outage_start=.9, outage_duration=.2))

    def test_plan_json_round_trip_preserves_continuous_trajectory(self):
        c = analytic_config()
        plan = linear_plan(c)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(asdict(plan)))
            restored = ContinuousPlan.from_file(path)
        np.testing.assert_allclose(restored.evaluate(c, np.linspace(0, 1, 37))["customers"],
                                   plan.evaluate(c, np.linspace(0, 1, 37))["customers"])


class ContinuousOptimizationTests(unittest.TestCase):
    def test_business_candidate_has_independent_replay_and_lower_bound(self):
        report, _ = optimize_continuous(VariationalConfig(), segments=1, starts=1)
        self.assertIsNotNone(report["plan"], report)
        self.assertFalse(report["global_optimality_proven"])
        self.assertTrue(report["nominal"]["target_met"])
        self.assertTrue(report["nominal"]["feasible_cash_and_capacity"])
        self.assertGreaterEqual(report["penalty"], report["analytic_lower_bound"]["value"] - 1e-9)
        self.assertAlmostEqual(report["penalty"], sum(report["penalty_components"].values()))

    def test_unsuccessful_search_does_not_claim_impossibility(self):
        c = replace(VariationalConfig(), target_profit=1e12)
        report, _ = optimize_continuous(c, segments=1, starts=1, maxiter=10)
        self.assertIsNone(report["plan"])
        self.assertEqual(report["status"], "no_verified_candidate")
        self.assertIn("not proof", report["interpretation"])

    def test_advertising_peak_between_original_nodes_is_refined(self):
        config = replace(VariationalConfig(), quality_weight=10)
        report, _ = optimize_continuous(config, segments=6, starts=1)
        self.assertIsNotNone(report["plan"], report)
        self.assertTrue(report["attempts"][0]["constraint_refinements"])
        self.assertLessEqual(report["maximum_marketing_rate"], config.advertising_max + .05)

    def test_cli_exports_replayable_plan_and_actual_stress_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.json"
            config_file.write_text(json.dumps(asdict(VariationalConfig())))
            with redirect_stdout(io.StringIO()):
                result = main(["variational", str(config_file), "--segments", "1", "--starts", "1",
                               "--stress", "--output-dir", str(root / "out")])
            self.assertEqual(result, 0)
            report = json.loads((root / "out/report.json").read_text())
            plan = ContinuousPlan.from_file(root / "out/plan.json")
            check, _ = replay(VariationalConfig(), plan)
            self.assertAlmostEqual(check["window_profit"], report["nominal"]["window_profit"])
            self.assertEqual(len(report["stress_checks"]), 2)
            self.assertTrue((root / "out/trajectory.csv").exists())


if __name__ == "__main__":
    unittest.main()
