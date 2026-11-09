"""Command-line entry point; reports and trajectories are reusable artifacts."""

import argparse
import csv
from dataclasses import asdict, fields, replace
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
    parser = argparse.ArgumentParser(description="Исследовательские модели подписочного сервиса")
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
    continuous = commands.add_parser("variational", help="Найти непрерывный путь с квадратичным штрафом")
    continuous.add_argument("config", type=Path)
    continuous.add_argument("--segments", type=int, default=3)
    continuous.add_argument("--coordinate", choices=("absolute", "relative"), default="absolute")
    continuous.add_argument("--quality-weight", type=float)
    continuous.add_argument("--starts", type=int, default=3)
    continuous.add_argument("--maxiter", type=int, default=400)
    continuous.add_argument("--stress", action="store_true")
    continuous.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "variational":
            return _continuous_command(args)
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


def _continuous_command(args):
    from .variational import ContinuousPlan, Disturbance, VariationalConfig, optimize_continuous, replay
    config = VariationalConfig.from_file(args.config)
    if args.quality_weight is not None:
        config = replace(config, quality_weight=args.quality_weight)
    report, rows = optimize_continuous(config, segments=args.segments, coordinate=args.coordinate,
                                      starts=args.starts, maxiter=args.maxiter)
    output_rows = [{"scenario": "nominal", **row} for row in rows]
    report["stress_checks"] = []
    if args.stress and report["plan"] is not None:
        plan = ContinuousPlan(**report["plan"])
        events = (Disturbance("development_delay", development_delay=config.years / 12),
                  Disturbance("service_outage", outage_start=config.years / 2,
                              outage_duration=config.years / 36,
                              extra_churn_hazard=1., repair_cost=100_000.))
        for event in events:
            summary, event_rows = replay(config, plan, event)
            report["stress_checks"].append(summary)
            output_rows.extend({"scenario": event.name, **row} for row in event_rows)
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plan_file = directory / "plan.json"
    if report["plan"] is not None:
        plan_file.write_text(json.dumps(report["plan"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    elif plan_file.exists():
        plan_file.unlink()
    with (directory / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        keys = ["scenario", "time_years", "customers", "quality", "price_per_year", "cash",
                "marketing_per_year", "development_per_year"]
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("plan", "attempts", "configuration")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["plan"] is not None else 2
