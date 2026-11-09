"""Independent forward accounting used to verify optimization candidates."""

from dataclasses import asdict, dataclass
import json
from math import isfinite, log1p
from pathlib import Path

CASH_TOLERANCE = 0.01  # Currency units; numerical tolerance, not extra funding.
CLIENT_TOLERANCE = 1e-7


def _number(name: str, value: float, *, minimum: float = 0, strict: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite number")
    if not isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"{name}: invalid value {value!r}")


@dataclass(frozen=True)
class Action:
    price: float
    marketing: float

    def __post_init__(self):
        _number("price", self.price, strict=True)
        _number("marketing", self.marketing)


@dataclass(frozen=True)
class Scenario:
    name: str
    acquisition_per_currency: float
    base_churn: float
    acquisition_price_elasticity: float = 1.2
    churn_price_elasticity: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario name must be nonempty")
        for field in ("acquisition_per_currency", "base_churn",
                      "acquisition_price_elasticity", "churn_price_elasticity"):
            _number(field, getattr(self, field))
        if self.base_churn > 1:
            raise ValueError("base_churn must be in [0, 1]")

    def response(self, action: Action, config: "ModelConfig") -> tuple[float, float]:
        ratio = action.price / config.reference_price
        if not isfinite(ratio) or ratio <= 0:
            raise ValueError("Price ratio is outside the supported numerical range")
        try:
            acquired = (
                self.acquisition_per_currency * config.advertising_scale
                * log1p(action.marketing / config.advertising_scale)
                * ratio ** (-self.acquisition_price_elasticity)
            )
            churn = min(1.0, self.base_churn * ratio ** self.churn_price_elasticity)
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("Response overflow: check price range and coefficients") from error
        if not isfinite(acquired) or not isfinite(churn):
            raise ValueError("Response overflow: check price range and coefficients")
        return acquired, churn


@dataclass(frozen=True)
class ModelConfig:
    months: int
    profit_window_months: int
    initial_customers: float
    initial_cash: float
    cash_reserve: float
    fixed_monthly_cost: float
    service_cost_per_customer: float
    capacity: float
    reference_price: float
    advertising_scale: float
    target_profit: float
    prices: tuple[float, ...]
    marketing_budgets: tuple[float, ...]
    scenarios: tuple[Scenario, ...]
    description: str = ""
    synthetic: bool = True

    def __post_init__(self):
        for name in ("months", "profit_window_months"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.profit_window_months > self.months:
            raise ValueError("Profit window cannot exceed the horizon")
        for name in ("initial_customers", "initial_cash", "cash_reserve",
                     "fixed_monthly_cost", "service_cost_per_customer", "target_profit"):
            _number(name, getattr(self, name))
        for name in ("capacity", "reference_price", "advertising_scale"):
            _number(name, getattr(self, name), strict=True)
        if self.initial_cash < self.cash_reserve:
            raise ValueError("Initial cash is below the mandatory reserve")
        for name in ("prices", "marketing_budgets"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must be nonempty")
            for value in values:
                _number(name, value, strict=(name == "prices"))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
        if not self.scenarios or not all(isinstance(s, Scenario) for s in self.scenarios):
            raise ValueError("At least one valid scenario is required")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("Scenario names must be unique")
        if type(self.synthetic) is not bool or not isinstance(self.description, str):
            raise ValueError("Invalid dataset metadata")

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(Action(p, m) for p in self.prices for m in self.marketing_budgets)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["scenarios"] = tuple(Scenario(**s) for s in data["scenarios"])
        data["prices"] = tuple(data["prices"])
        data["marketing_budgets"] = tuple(data["marketing_budgets"])
        return cls(**data)


@dataclass(frozen=True)
class Month:
    month: int
    price: float
    marketing: float
    customers_start: float
    churn: float
    paying_customers: float
    acquired_customers: float
    customers_end: float
    cash_start: float
    cash_after_upfront: float
    revenue: float
    service_cost: float
    fixed_cost: float
    profit: float
    cash_end: float


@dataclass(frozen=True)
class Violation:
    month: int
    stage: str
    message: str
    value: float
    limit: float


@dataclass
class Simulation:
    scenario: str
    rows: list[Month]
    violation: Violation | None
    expected_months: int
    profit_window_months: int
    initial_cash: float
    cash_reserve: float

    @property
    def feasible(self) -> bool:
        return self.violation is None and len(self.rows) == self.expected_months

    @property
    def window_profit(self) -> float | None:
        if not self.feasible:
            return None
        return sum(row.profit for row in self.rows[-self.profit_window_months:])

    @property
    def minimum_cash(self) -> float:
        values = [self.initial_cash]
        values.extend(v for row in self.rows for v in (row.cash_after_upfront, row.cash_end))
        if self.violation and self.violation.stage.startswith("cash"):
            values.append(self.violation.value)
        return min(values)

    def summary(self) -> dict:
        required_cash = None
        if self.feasible:
            # Fixed plan: changing initial cash translates the entire cash path.
            required_cash = self.initial_cash + self.cash_reserve - self.minimum_cash
        return {
            "scenario": self.scenario,
            "feasible": self.feasible,
            "completed_months": len(self.rows),
            "window_profit": self.window_profit,
            "final_customers": self.rows[-1].customers_end if self.rows else None,
            "final_cash": self.rows[-1].cash_end if self.rows else self.initial_cash,
            "minimum_cash": self.minimum_cash,
            "required_initial_cash_for_this_plan": required_cash,
            "violation": asdict(self.violation) if self.violation else None,
        }


def simulate(config: ModelConfig, scenario: Scenario, plan: list[Action] | tuple[Action, ...]) -> Simulation:
    """Evaluate any positive prices/nonnegative budgets, including off-grid plans.

    All fixed and advertising expenses precede receipts. New signups join the
    next month's eligible base; only remaining customers pay this month.
    """
    if len(plan) != config.months:
        raise ValueError(f"Expected {config.months} monthly actions, got {len(plan)}")
    result = Simulation(scenario.name, [], None, config.months,
                        config.profit_window_months, config.initial_cash, config.cash_reserve)
    customers, cash = config.initial_customers, config.initial_cash
    for t, action in enumerate(plan, 1):
        if not isinstance(action, Action):
            raise ValueError("Plan entries must be Action objects")
        after_upfront = cash - config.fixed_monthly_cost - action.marketing
        if after_upfront < config.cash_reserve - CASH_TOLERANCE:
            result.violation = Violation(t, "cash_upfront", "Reserve breached before receipts",
                                         after_upfront, config.cash_reserve)
            break
        acquired, churn = scenario.response(action, config)
        paying = (1 - churn) * customers
        if paying > config.capacity + CLIENT_TOLERANCE:
            result.violation = Violation(t, "capacity", "Paying base exceeds service capacity",
                                         paying, config.capacity)
            break
        revenue = action.price * paying
        service = config.service_cost_per_customer * paying
        profit = revenue - service - config.fixed_monthly_cost - action.marketing
        cash_end = cash + profit
        if not all(isfinite(v) for v in (after_upfront, paying, revenue, service,
                                         profit, cash_end, paying + acquired)):
            raise ValueError("Accounting overflow: reduce the numerical scale of inputs")
        result.rows.append(Month(t, action.price, action.marketing, customers, churn,
                                 paying, acquired, paying + acquired, cash, after_upfront,
                                 revenue, service, config.fixed_monthly_cost, profit, cash_end))
        if cash_end < config.cash_reserve - CASH_TOLERANCE:
            result.violation = Violation(t, "cash_end", "Reserve breached after service payments",
                                         cash_end, config.cash_reserve)
            break
        customers, cash = paying + acquired, cash_end
    return result
