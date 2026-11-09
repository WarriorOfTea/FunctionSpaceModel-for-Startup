from dataclasses import replace
import math
import unittest

from function_space_model.model import Action, ModelConfig, Scenario, simulate


def small_config(**changes):
    config = ModelConfig(
        months=3, profit_window_months=2, initial_customers=20,
        initial_cash=10000, cash_reserve=100, fixed_monthly_cost=50,
        service_cost_per_customer=2, capacity=1000, reference_price=20,
        advertising_scale=100, target_profit=1000,
        prices=(10, 20), marketing_budgets=(0, 100),
        scenarios=(Scenario("slow", 0.08, 0.1), Scenario("fast", 0.12, 0.05)),
    )
    return replace(config, **changes)


class AccountingTests(unittest.TestCase):
    def test_no_marketing_matches_closed_form_decay_and_cash(self):
        config = small_config(months=5, profit_window_months=3)
        scenario = config.scenarios[0]
        result = simulate(config, scenario, [Action(20, 0)] * 5)
        self.assertTrue(result.feasible)
        expected_profit = 0
        for t, row in enumerate(result.rows, 1):
            expected_paying = 20 * 0.9 ** t
            self.assertAlmostEqual(row.paying_customers, expected_paying)
            expected_profit += 18 * expected_paying - 50
        self.assertAlmostEqual(result.rows[-1].cash_end, 10000 + expected_profit)

    def test_new_clients_do_not_pay_in_acquisition_month(self):
        config = small_config(initial_customers=0)
        result = simulate(config, config.scenarios[0], [Action(20, 100)] * 3)
        self.assertEqual(result.rows[0].revenue, 0)
        self.assertGreater(result.rows[0].acquired_customers, 0)
        self.assertGreater(result.rows[1].revenue, 0)

    def test_upfront_reserve_blocks_plan_even_if_receipts_would_cover_it(self):
        config = small_config(initial_cash=120, fixed_monthly_cost=30)
        result = simulate(config, config.scenarios[0], [Action(20, 0)] * 3)
        self.assertFalse(result.feasible)
        self.assertEqual(result.violation.stage, "cash_upfront")
        self.assertEqual(result.violation.month, 1)
        self.assertEqual(result.rows, [])
        self.assertEqual(result.minimum_cash, 90)
        self.assertIsNone(result.window_profit)

    def test_service_payments_can_breach_reserve_after_receipts(self):
        config = small_config(initial_cash=200, fixed_monthly_cost=0,
                              service_cost_per_customer=100)
        result = simulate(config, config.scenarios[0], [Action(20, 0)] * 3)
        self.assertFalse(result.feasible)
        self.assertEqual(result.violation.stage, "cash_end")
        self.assertEqual(len(result.rows), 1)
        self.assertIsNone(result.window_profit)

    def test_capacity_rejects_instead_of_silently_dropping_customers(self):
        config = small_config(capacity=5)
        result = simulate(config, config.scenarios[0], [Action(20, 0)] * 3)
        self.assertFalse(result.feasible)
        self.assertEqual(result.violation.stage, "capacity")

    def test_target_uses_last_window_not_annualized_last_month(self):
        config = small_config()
        result = simulate(config, config.scenarios[0], [Action(20, 0)] * 3)
        expected = sum(row.profit for row in result.rows[1:])
        self.assertAlmostEqual(result.window_profit, expected)
        self.assertNotAlmostEqual(result.window_profit, result.rows[-1].profit * 2)

    def test_required_capital_is_feasible_and_any_lower_value_fails(self):
        config = small_config(initial_customers=0, months=5, profit_window_months=2)
        plan = [Action(20, 100)] * 5
        result = simulate(config, config.scenarios[0], plan)
        required = result.summary()["required_initial_cash_for_this_plan"]
        self.assertTrue(simulate(replace(config, initial_cash=required), config.scenarios[0], plan).feasible)
        self.assertFalse(simulate(replace(config, initial_cash=required - 1), config.scenarios[0], plan).feasible)

    def test_custom_plan_may_use_prices_outside_optimization_grid(self):
        config = small_config()
        self.assertTrue(simulate(config, config.scenarios[0], [Action(15, 50)] * 3).feasible)

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate(small_config(), small_config().scenarios[0], [Action(20, 0)])


class ResponseAndInputTests(unittest.TestCase):
    def test_advertising_is_increasing_concave_and_has_no_free_clients(self):
        config = small_config()
        scenario = config.scenarios[0]
        values = [scenario.response(Action(20, m), config)[0] for m in (0, 100, 200, 300)]
        self.assertEqual(values[0], 0)
        increments = [b - a for a, b in zip(values, values[1:])]
        self.assertTrue(all(v > 0 for v in increments))
        self.assertTrue(all(a > b for a, b in zip(increments, increments[1:])))

    def test_price_reduces_acquisition_and_increases_churn(self):
        config = small_config()
        scenario = config.scenarios[0]
        low = scenario.response(Action(10, 100), config)
        high = scenario.response(Action(20, 100), config)
        self.assertGreater(low[0], high[0])
        self.assertLess(low[1], high[1])

    def test_churn_is_bounded_by_one(self):
        config = small_config()
        self.assertEqual(config.scenarios[0].response(Action(1000, 0), config)[1], 1)

    def test_invalid_and_nonfinite_inputs_are_rejected(self):
        for changes in ({"initial_cash": 50}, {"capacity": 0}, {"prices": ()},
                        {"profit_window_months": 4}, {"months": True},
                        {"initial_customers": math.nan}, {"scenarios": ()},
                        {"marketing_budgets": (0, math.inf)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                small_config(**changes)
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(price=value), self.assertRaises(ValueError):
                Action(value, 0)
        with self.assertRaises(ValueError):
            small_config().scenarios[0].response(Action(1e308, 0),
                                               small_config(reference_price=1e-308))


if __name__ == "__main__":
    unittest.main()
