"""Exhaustive comparison with the small class of constant grid actions."""

from dataclasses import asdict

from .model import ModelConfig, simulate


def compare_constant_plans(config: ModelConfig):
    records, best, best_checks = [], None, []
    for action in config.actions:
        checks = [simulate(config, scenario, [action] * config.months) for scenario in config.scenarios]
        feasible = all(check.feasible for check in checks)
        score = min(check.window_profit for check in checks) if feasible else None
        record = {**asdict(action), "feasible_in_all_models": feasible, "worst_profit": score,
                  "models": [check.summary() for check in checks]}
        records.append(record)
        if feasible and (best is None or score > best["worst_profit"]):
            best, best_checks = record, checks
    return {
        "synthetic": config.synthetic,
        "description": config.description,
        "configuration": asdict(config),
        "mode": "exhaustive_constant_plan_comparison",
        "scope": "Only constant actions from the configured grid and listed models",
        "constant_plans_checked": len(records),
        "constant_plans_feasible_in_all_models": sum(r["feasible_in_all_models"] for r in records),
        "best_constant_plan": best,
        "constant_plans": records,
        "plan": ([{"price": best["price"], "marketing": best["marketing"]}] * config.months
                 if best is not None else None),
        "simulations": [check.summary() for check in best_checks],
    }, best_checks
