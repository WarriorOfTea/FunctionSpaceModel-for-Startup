"""Command-line entry point; reports and trajectories are reusable artifacts."""

import argparse
import csv
from dataclasses import asdict, fields
import json
from pathlib import Path

from .model import Action, ModelConfig, Month, simulate


def _write_outputs(directory: Path, report: dict, simulations):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (directory / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario"] + [f.name for f in fields(Month)],
                                lineterminator="\n")
        writer.writeheader()
        for check in simulations:
            for row in check.rows:
                writer.writerow({"scenario": check.scenario, **asdict(row)})
    plan = report.get("plan")
    if plan is not None:
        (directory / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    elif (directory / "plan.json").exists():
        (directory / "plan.json").unlink()  # Do not leave a previous successful plan behind.


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Месячная модель подписочного сервиса")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("optimize", help="Найти план на сетке цен и бюджетов")
    search.add_argument("config", type=Path)
    search.add_argument("--scenario", action="append", help="Имя модели; по умолчанию все модели")
    search.add_argument("--time-limit", type=float, default=30)
    search.add_argument("--relative-gap", type=float, default=1e-5)
    search.add_argument("--output-dir", type=Path, required=True)
    forward = commands.add_parser("simulate", help="Проверить собственный помесячный план")
    forward.add_argument("config", type=Path)
    forward.add_argument("plan", type=Path)
    forward.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser("compare-constant", help="Перебрать постоянные цены и бюджеты на сетке")
    compare.add_argument("config", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = ModelConfig.from_file(args.config)
        if args.command == "optimize":
            from .optimizer import optimize
            result = optimize(config, scenario_names=tuple(args.scenario) if args.scenario else None,
                              time_limit=args.time_limit, relative_gap=args.relative_gap)
            report, checks = result.summary(config), result.simulations
            exit_code = 0 if result.plan is not None else 2
        elif args.command == "compare-constant":
            from .benchmark import compare_constant_plans
            report, checks = compare_constant_plans(config)
            exit_code = 0 if report["plan"] is not None else 2
        else:
            data = json.loads(args.plan.read_text(encoding="utf-8"))
            plan = [Action(**entry) for entry in data]
            checks = [simulate(config, scenario, plan) for scenario in config.scenarios]
            report = {"synthetic": config.synthetic, "description": config.description,
                      "configuration": asdict(config),
                      "mode": "simulation", "plan": [asdict(a) for a in plan],
                      "simulations": [check.summary() for check in checks]}
            exit_code = 0 if all(check.feasible for check in checks) else 2
        _write_outputs(args.output_dir, report, checks)
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("plan", "solver_message", "configuration", "constant_plans")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code
