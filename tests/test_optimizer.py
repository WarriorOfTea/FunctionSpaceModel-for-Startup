from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch

from scipy.optimize import OptimizeResult

from function_space_model.model import simulate
from function_space_model.optimizer import optimize
from test_model import small_config


def exhaustive_optimum(config, scenarios):
    scores = []
    for plan in product(config.actions, repeat=config.months):
        results = [simulate(config, scenario, plan) for scenario in scenarios]
        if all(result.feasible for result in results):
            scores.append(min(result.window_profit for result in results))
    return max(scores) if scores else None


class OptimizerTests(unittest.TestCase):
    def test_single_model_matches_exhaustive_search(self):
        config = small_config()
        expected = exhaustive_optimum(config, config.scenarios[:1])
        result = optimize(config, scenario_names=("slow",), time_limit=10, relative_gap=0)
        self.assertEqual(result.solver_status, "optimal_within_tolerance")
        self.assertAlmostEqual(result.lower_bound, expected, places=5)
        self.assertAlmostEqual(result.upper_bound, expected, places=5)

    def test_shared_plan_maximin_matches_exhaustive_search(self):
        config = small_config()
        expected = exhaustive_optimum(config, config.scenarios)
        result = optimize(config, time_limit=10, relative_gap=0)
        self.assertEqual(result.solver_status, "optimal_within_tolerance")
        self.assertAlmostEqual(result.lower_bound, expected, places=5)
        self.assertAlmostEqual(result.upper_bound, expected, places=5)
        self.assertTrue(all(check.feasible for check in result.simulations))
        for check in result.simulations:
            self.assertEqual([row.price for row in check.rows], [a.price for a in result.plan])

    def test_upfront_infeasibility_is_preserved_in_optimizer(self):
        config = small_config(initial_cash=120, fixed_monthly_cost=30)
        result = optimize(config, time_limit=10)
        self.assertEqual(result.solver_status, "infeasible_on_grid")
        self.assertIsNone(result.plan)
        self.assertIsNone(result.lower_bound)

    def test_capacity_constraints_are_preserved_in_optimizer(self):
        result = optimize(small_config(capacity=5), time_limit=10)
        self.assertEqual(result.solver_status, "infeasible_on_grid")
        self.assertIsNone(result.plan)

    def test_unprofitable_business_has_negative_objective(self):
        config = small_config(service_cost_per_customer=100, prices=(10,), marketing_budgets=(0,))
        result = optimize(config, time_limit=10)
        self.assertLess(result.lower_bound, 0)
        self.assertAlmostEqual(result.lower_bound, exhaustive_optimum(config, config.scenarios))

    def test_zero_initial_base_and_zero_advertising(self):
        config = small_config(initial_customers=0, marketing_budgets=(0,))
        result = optimize(config, time_limit=10)
        self.assertAlmostEqual(result.lower_bound, -100)

    def test_verified_plan_can_meet_target(self):
        result = optimize(small_config(target_profit=1), time_limit=10)
        self.assertEqual(result.target_status, "met_by_verified_plan_for_listed_models")

    def test_last_month_has_no_useless_marketing(self):
        result = optimize(small_config(), time_limit=10)
        self.assertEqual(result.plan[-1].marketing, 0)

    def test_limit_without_candidate_is_not_infeasibility(self):
        with patch("function_space_model.optimizer.milp", return_value=OptimizeResult(
                status=1, message="time limit", x=None, mip_dual_bound=None, mip_gap=None)):
            result = optimize(small_config(), time_limit=0.1)
        self.assertEqual(result.solver_status, "limit_reached")
        self.assertEqual(result.target_status, "undetermined")
        self.assertIsNone(result.plan)

    def test_invalid_selection_and_limits_are_rejected(self):
        for kwargs in ({"scenario_names": ()}, {"scenario_names": ("unknown",)},
                       {"scenario_names": ("slow", "slow")}, {"time_limit": 0},
                       {"relative_gap": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                optimize(small_config(), **kwargs)

    def test_solver_error_is_not_used_as_evidence_against_target(self):
        with patch("function_space_model.optimizer.milp", return_value=OptimizeResult(
                status=4, message="numerical error", x=None, mip_dual_bound=0, mip_gap=None)):
            result = optimize(small_config(), time_limit=0.1)
        self.assertEqual(result.solver_status, "solver_error")
        self.assertEqual(result.target_status, "undetermined")


if __name__ == "__main__":
    unittest.main()
