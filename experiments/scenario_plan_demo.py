"""Compare one nominal plan with one common plan for three fixed scenarios."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from function_space_model.scenarios import standard_disturbances
from function_space_model.variational import (ContinuousPlan, Disturbance, VariationalConfig,
                                             optimize_continuous, replay)


def run_experiment(config_path, output_dir, *, plot=False):
    config = VariationalConfig.from_file(config_path)
    events = standard_disturbances(config)
    reports = {}
    definitions = [("nominal", 6, ())] + [(f"common_{size}", size, events) for size in (1, 3, 6)]
    for name, segments, disturbances in definitions:
        result, _ = optimize_continuous(config, segments=segments, disturbances=disturbances)
        reports[name] = result
        print(json.dumps({"case": name, "status": result["status"], "penalty": result.get("penalty"),
                          "seconds": result["elapsed_seconds"]}), flush=True)
    comparisons, rows = {}, []
    for name in ("nominal", "common_6"):
        report = reports[name]
        if report["plan"] is None:
            continue
        plan = ContinuousPlan(**report["plan"])
        checks = []
        for event in (Disturbance(),) + events:
            summary, trajectory = replay(config, plan, event)
            checks.append(summary)
            rows.extend({"plan": name, "scenario": event.name, **row} for row in trajectory)
        comparisons[name] = {"scenarios": checks,
                             "minimum_cash_all_scenarios": min(s["minimum_cash"] for s in checks),
                             "required_initial_cash_for_this_plan_all_scenarios":
                             max(s["required_initial_cash_for_this_plan"] for s in checks),
                             "target_met_count": sum(s["target_met"] for s in checks),
                             "cash_and_capacity_count": sum(s["feasible_cash_and_capacity"] for s in checks),
                             "scenario_count": len(checks)}
    audit = None
    if reports["common_6"]["plan"] is not None:
        # This combination was NOT a design constraint. Keep it separate.
        event = Disturbance("combined_not_optimized", development_delay=.25,
                            outage_start=config.years / 2, outage_duration=1 / 12,
                            extra_churn_hazard=1., repair_cost=100_000.)
        audit, _ = replay(config, ContinuousPlan(**reports["common_6"]["plan"]), event)
    result = {"scope": "synthetic enumerated-scenario common plan; no global or probabilistic guarantee",
              "objective_scope": "same nominal-path penalty in both optimizations",
              "configuration": asdict(config), "disturbances": [asdict(d) for d in events],
              "searches": reports, "comparisons": comparisons,
              "out_of_design_combined_audit": audit}
    if all(reports[name]["plan"] is not None for name in ("nominal", "common_6")):
        a, b = reports["nominal"], reports["common_6"]
        result["cost_comparison"] = {
            "penalty_increase": b["penalty"] - a["penalty"],
            "penalty_increase_percent": 100 * (b["penalty"] / a["penalty"] - 1),
            "marketing_increase": b["nominal"]["marketing_spent"] - a["nominal"]["marketing_spent"],
            "development_increase": b["nominal"]["development_spent"] - a["nominal"]["development_spent"],
            "fixed_plan_capital_change":
            comparisons["common_6"]["required_initial_cash_for_this_plan_all_scenarios"]
            - comparisons["nominal"]["required_initial_cash_for_this_plan_all_scenarios"]}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                         allow_nan=False) + "\n", encoding="utf-8")
    with (directory / "paths.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["plan", "scenario", "time_years", "customers", "quality",
                                "price_per_year", "cash", "marketing_per_year", "development_per_year"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        make_plot(result, rows, directory)
    return result


def make_plot(report, rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "scenario-plan-demo"})
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    c = report["configuration"]
    for name, label, style in (("nominal", "Прежний план", "--"), ("common_6", "Общий план", "-")):
        nominal = [r for r in rows if r["plan"] == name and r["scenario"] == "nominal"]
        times = [r["time_years"] for r in nominal]
        axes[0, 0].plot(times, [r["customers"] for r in nominal], label=label, linestyle=style)
        axes[0, 1].plot(times, [r["marketing_per_year"] / 1e6 for r in nominal], label=label, linestyle=style)
        grid = np.linspace(0, c["years"], 601)
        cash = []
        for scenario in ("nominal", "development_delay_3_months", "service_outage_1_month"):
            subset = [r for r in rows if r["plan"] == name and r["scenario"] == scenario]
            cash.append(np.interp(grid, [r["time_years"] for r in subset], [r["cash"] / 1e6 for r in subset]))
        axes[1, 1].plot(grid, np.min(cash, axis=0), label=label, linestyle=style)
    labels = ("Без сбоя", "Задержка", "Простой")
    for offset, name, label in ((-.18, "nominal", "Прежний план"), (.18, "common_6", "Общий план")):
        checks = report["comparisons"][name]["scenarios"]
        bars = axes[1, 0].bar(np.arange(3) + offset, [s["window_profit"] / 1e6 for s in checks], .36, label=label)
        axes[1, 0].bar_label(bars, fmt="%.2f", fontsize=9)
    axes[1, 0].set_xticks(np.arange(3), labels)
    axes[1, 0].axhline(c["target_profit"] / 1e6, color="black", linestyle=":", label="Цель")
    axes[1, 1].axhline(c["reserve"] / 1e6, color="black", linestyle=":", label="Резерв")
    titles = (("Плановая клиентская база", "Клиенты"), ("Плановые рекламные расходы", "млн / год"),
              ("Прибыль последнего года", "млн"), ("Наименьший остаток среди сценариев", "млн"))
    for i, (ax, (title, ylabel)) in enumerate(zip(axes.ravel(), titles)):
        ax.set(title=title, ylabel=ylabel)
        if i != 2:
            ax.set_xlabel("Время, годы")
        ax.grid(alpha=.25, axis="y")
        ax.legend(fontsize=9)
    figure.suptitle("Один план для трёх заданных сценариев — условный пример", fontsize=13)
    figure.savefig(directory / "common_plan.png", dpi=170)
    figure.savefig(directory / "common_plan.svg", metadata={"Date": None})
    svg = directory / "common_plan.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/continuous.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reports/assets/common_plan"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = run_experiment(args.config, args.output_dir, plot=args.plot)
    print(json.dumps({"cost_comparison": report.get("cost_comparison"),
                      "target_counts": {k: f"{v['target_met_count']}/{v['scenario_count']}"
                                        for k, v in report["comparisons"].items()}}, ensure_ascii=False))
