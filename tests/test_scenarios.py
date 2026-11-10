from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import leggauss

from function_space_model.cli import main
from function_space_model.curves import quadrature
from function_space_model.scenarios import (ScenarioProblem, linear_state,
                                           partial_gauss_matrix, standard_disturbances)
from function_space_model.variational import Disturbance, VariationalConfig, optimize_continuous, replay
from test_variational import analytic_config, linear_plan


class ScenarioIntegrationTests(unittest.TestCase):
    def test_partial_integrals_are_exact_for_polynomials(self):
        nodes, _ = leggauss(8)
        matrix = partial_gauss_matrix(8)
        for degree in range(8):
            expected = (nodes ** (degree + 1) - (-1) ** (degree + 1)) / (degree + 1)
            np.testing.assert_allclose(matrix @ nodes ** degree, expected, atol=1e-14)

    def test_linear_state_matches_closed_form_nonzero_hazard(self):
        breaks = np.linspace(0, 3, 13)
        times, weights = quadrature(breaks, 8)
        initial, hazard, acquisition = 10., .7, 30.
        n, boundaries = linear_state(initial, np.full((12, 8), hazard), np.full((12, 8), acquisition),
                                     np.diff(breaks) / 2, weights.reshape(-1, 8), partial_gauss_matrix(8))
        expected = lambda t: acquisition / hazard + (initial - acquisition / hazard) * np.exp(-hazard * t)
        np.testing.assert_allclose(n.ravel(), expected(times), atol=1e-10)
        np.testing.assert_allclose(boundaries, expected(breaks), atol=1e-10)

    def test_predictions_match_independent_replay_including_short_outage(self):
        config = analytic_config()
        events = (Disturbance("delay", development_delay=.25),
                  Disturbance("short", outage_start=.431, outage_duration=.0001,
                              extra_churn_hazard=1., repair_cost=100.))
        for coordinate in ("absolute", "relative"):
            plan = linear_plan(config, final_quality=.5, coordinate=coordinate)
            problem = ScenarioProblem(config, 2, coordinate, events)
            free = np.r_[plan.customers[1:], plan.quality[1:], plan.price[1:]]
            for event, predicted in zip(events, problem.predictions(free)):
                with self.subTest(event=event.name, coordinate=coordinate):
                    checked, _ = replay(config, plan, event)
                    self.assertAlmostEqual(predicted["customers_at_boundaries"][-1], checked["final_customers"], places=6)
                    self.assertAlmostEqual(predicted["cash"][-1], checked["final_cash"], places=4)
                    self.assertAlmostEqual(predicted["window_profit"], checked["window_profit"], places=4)

    def test_invalid_names_and_event_horizons_are_rejected(self):
        c = VariationalConfig()
        for events in ((), (Disturbance(),), (Disturbance("x"), Disturbance("x")),
                       (Disturbance("late", outage_start=3, outage_duration=.1),)):
            with self.subTest(events=events), self.assertRaises(ValueError):
                ScenarioProblem(c, 1, "absolute", events)


class CommonPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = VariationalConfig()
        cls.events = standard_disturbances(cls.config)
        cls.report, _ = optimize_continuous(cls.config, segments=1, starts=1, disturbances=cls.events)

    def test_one_plan_meets_all_three_scenarios(self):
        r = self.report
        self.assertIsNotNone(r["plan"], r)
        self.assertEqual(r["objective_scope"], "nominal_path_penalty")
        self.assertFalse(r["global_optimality_proven"])
        checks = [r["nominal"]] + r["scenario_checks"]
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(s["target_met"] and s["feasible_cash_and_capacity"] for s in checks))
        self.assertTrue(all(s["window_profit"] >= self.config.target_profit - .05 for s in checks))

    def test_fixed_spending_is_identical_between_scenarios(self):
        r = self.report
        for check in r["scenario_checks"]:
            self.assertAlmostEqual(check["marketing_spent"], r["nominal"]["marketing_spent"], places=3)
            self.assertAlmostEqual(check["development_spent"], r["nominal"]["development_spent"], places=3)
        for error in r["scenario_prediction_errors"]:
            self.assertLess(error["final_customer_error"], .01)
            self.assertLess(error["window_profit_error"], .05)

    def test_nominal_success_does_not_override_failed_scenario_replay(self):
        fake = {"name": "forced_failure", "minimum_cash": self.config.initial_cash,
                "minimum_cash_time": 0., "target_met": False, "feasible_cash_and_capacity": True}
        with patch.object(ScenarioProblem, "additional_checks", return_value=[fake]):
            report, _ = optimize_continuous(self.config, segments=1, starts=1, disturbances=self.events)
        self.assertIsNone(report["plan"])
        self.assertEqual(report["status"], "no_verified_candidate")
        self.assertIn("not proof", report["interpretation"])

    def test_cli_exports_three_scenario_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps(asdict(self.config)))
            with redirect_stdout(io.StringIO()):
                code = main(["variational", str(path), "--scenario-plan", "--segments", "1", "--starts", "1",
                             "--output-dir", str(root / "out")])
            self.assertEqual(code, 0)
            report = json.loads((root / "out/report.json").read_text())
            self.assertEqual(report["mode"], "continuous_scenario_plan")
            self.assertEqual(len(report["scenario_checks"]), 2)
            csv = (root / "out/trajectory.csv").read_text()
            for name in ("nominal", "development_delay_3_months", "service_outage_1_month"):
                self.assertIn(name, csv)


if __name__ == "__main__":
    unittest.main()
