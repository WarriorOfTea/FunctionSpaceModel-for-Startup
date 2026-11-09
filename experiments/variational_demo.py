"""Reproducible continuous-path benchmarks, sensitivity and fixed-plan shocks."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import scipy

from function_space_model.curves import endpoint_benchmark
from function_space_model.variational import (ContinuousPlan, Disturbance, VariationalConfig,
                                             optimize_continuous, replay)


def run_experiment(config_path, output_dir, *, plot=False):
    base = VariationalConfig.from_file(config_path)
    analytic = [endpoint_benchmark(coordinate=coordinate, segments=segments, pause=pause)
                for coordinate in ("absolute", "relative") for segments in (1, 3, 6)
                for pause in (0., .25)]
    definitions = [(f"absolute_segments_{segments}", "absolute", segments, 1.) for segments in (1, 3, 6)]
    definitions += [("quality_weight_0.1", "absolute", 6, .1),
                    ("quality_weight_10", "absolute", 6, 10.),
                    ("relative_segments_6", "relative", 6, 1.)]
    cases, rows = {}, []
    for name, coordinate, segments, quality_weight in definitions:
        config = replace(base, quality_weight=quality_weight)
        report, trajectory = optimize_continuous(config, coordinate=coordinate, segments=segments)
        cases[name] = report
        rows.extend({"case": name, **row} for row in trajectory)
        print(json.dumps({"case": name, "status": report["status"], "penalty": report.get("penalty"),
                          "seconds": report["elapsed_seconds"]}), flush=True)
    baseline = cases["absolute_segments_6"]
    shocks = {}
    if baseline["plan"] is not None:
        plan = ContinuousPlan(**baseline["plan"])
        events = (Disturbance("development_delay_3_months", development_delay=.25),
                  Disturbance("service_outage_1_month", outage_start=1.5,
                              outage_duration=1 / 12, extra_churn_hazard=1., repair_cost=100_000.))
        for event in events:
            summary, trajectory = replay(base, plan, event)
            shocks[event.name] = summary
            rows.extend({"case": event.name, **row} for row in trajectory)
    report = {"scope": "synthetic continuous-path experiment; no real-company calibration",
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
              "base_configuration": asdict(base), "analytic_benchmarks": analytic,
              "business_cases": cases, "fixed_plan_stress_checks": shocks}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (directory / "paths.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "time_years", "customers", "quality",
                                "price_per_year", "cash", "marketing_per_year", "development_per_year"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        make_plot(base, rows, directory)
    return report


def make_plot(config, rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "variational-demo"})
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    chosen = (("quality_weight_0.1", "Вес качества 0,1"),
              ("absolute_segments_6", "Вес качества 1"),
              ("quality_weight_10", "Вес качества 10"),
              ("relative_segments_6", "Процентный рост, вес 1"))
    for name, label in chosen:
        data = [row for row in rows if row["case"] == name]
        times = [row["time_years"] for row in data]
        linestyle = "--" if name.startswith("relative") else "-"
        for ax, field, factor in ((axes[0, 0], "customers", 1), (axes[0, 1], "quality", 1),
                                  (axes[1, 0], "marketing_per_year", 1e6)):
            ax.plot(times, [row[field] / factor for row in data], label=label, linestyle=linestyle)
    for name, label in (("absolute_segments_6", "План без сбоев"),
                        ("development_delay_3_months", "Развитие позже на 3 месяца"),
                        ("service_outage_1_month", "Простой 1 месяц")):
        data = [row for row in rows if row["case"] == name]
        axes[1, 1].plot([row["time_years"] for row in data], [row["cash"] / 1e6 for row in data], label=label)
    axes[1, 1].axhline(config.reserve / 1e6, color="black", linestyle=":", label="Резерв")
    titles = (("Клиентская база", "Клиенты"), ("Качество продукта", "Доля решённых задач"),
              ("Рекламные расходы", "млн денежных единиц / год"), ("Деньги при неизменном плане", "млн денежных единиц"))
    for ax, (title, ylabel) in zip(axes.ravel(), titles):
        ax.set(title=title, xlabel="Время, годы", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    figure.suptitle("Условный пример: непрерывные пути при цели 10 млн прибыли последнего года", fontsize=12)
    figure.savefig(directory / "continuous_paths.png", dpi=170)
    figure.savefig(directory / "continuous_paths.svg", metadata={"Date": None})
    svg = directory / "continuous_paths.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/continuous.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports/assets/variational"))
    parser.add_argument("--plot", action="store_true", help="Requires matplotlib (uv run --with matplotlib)")
    args = parser.parse_args()
    result = run_experiment(args.config, args.output_dir, plot=args.plot)
    print(json.dumps({"analytic_passed": sum(r["analytic_check_passed"] for r in result["analytic_benchmarks"]),
                      "analytic_total": len(result["analytic_benchmarks"]),
                      "candidates": sum(r["plan"] is not None for r in result["business_cases"].values()),
                      "business_total": len(result["business_cases"])}, ensure_ascii=False))
