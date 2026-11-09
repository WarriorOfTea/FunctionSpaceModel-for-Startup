"""Compare a calibrated constant-hazard submodel with cohort survival models.

Training uses years 0..7 only. Forecasts are conditional on observed survival
at year 7, which is the same starting information for all three methods.
This is a research experiment, not a replacement for the business optimizer.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import betaln

TRAIN_END = 7
METHODS = ("constant_churn_mle", "constant_churn_latest", "shifted_beta_geometric")


def validate_survival(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Need a finite one-dimensional survival sequence")
    if abs(values[0] - 1) > 1e-12 or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Survival must start at 1 and lie in [0, 1]")
    if np.any(np.diff(values) > 0):
        raise ValueError("Cohort survival cannot increase")
    return values


def constant_churn_mle(training):
    training = validate_survival(training)
    # Cancellations / total exposure, with right-censoring of surviving users.
    return float((1 - training[-1]) / training[:-1].sum())


def sbg_survival(alpha, beta, periods):
    if alpha <= 0 or beta <= 0:
        raise ValueError("Positive beta-shape parameters required")
    return np.exp(betaln(alpha, beta + np.asarray(periods)) - betaln(alpha, beta))


def fit_sbg(training):
    training = validate_survival(training)
    cancellations = training[:-1] - training[1:]
    periods = np.arange(1, len(training))

    def negative_log_likelihood(log_parameters):
        alpha, beta = np.exp(log_parameters)
        log_mass = betaln(alpha + 1, beta + periods - 1) - betaln(alpha, beta)
        log_censored = betaln(alpha, beta + len(training) - 1) - betaln(alpha, beta)
        return -float(cancellations @ log_mass + training[-1] * log_censored)

    starts = ((0.5, 1), (1, 1), (2, 5), (5, 20), (0.1, 0.1), (10, 100))
    fits = [minimize(negative_log_likelihood, np.log(start), method="L-BFGS-B",
                     bounds=[(-10, 10), (-10, 10)],
                     options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 2000})
            for start in starts]
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("sBG estimation did not converge")
    best = min(successful, key=lambda fit: fit.fun)
    if np.any(np.abs(best.x) >= 9.999):
        raise RuntimeError("sBG optimum reached a numerical parameter bound")
    alpha, beta = np.exp(best.x)
    return {"alpha": float(alpha), "beta": float(beta),
            "negative_log_likelihood_per_initial_customer": float(best.fun),
            "successful_starts": len(successful), "starts": len(starts)}


def forecast_methods(training, future_periods):
    """Accept training only, to keep holdout values out of parameter fitting."""
    training = validate_survival(training)
    cutoff = len(training) - 1
    future_periods = np.asarray(future_periods)
    if np.any(future_periods < cutoff) or training[-2] <= 0:
        raise ValueError("Need future periods and a nonempty cohort at the last exposure")
    churn = constant_churn_mle(training)
    latest_churn = float(1 - training[-1] / training[-2])
    sbg = fit_sbg(training)
    survival = sbg_survival(sbg["alpha"], sbg["beta"], future_periods)
    anchor = float(sbg_survival(sbg["alpha"], sbg["beta"], cutoff))
    predictions = {
        "constant_churn_mle": training[-1] * (1 - churn) ** (future_periods - cutoff),
        "constant_churn_latest": training[-1] * (1 - latest_churn) ** (future_periods - cutoff),
        "shifted_beta_geometric": training[-1] * survival / anchor,
    }
    parameters = {"constant_churn_mle": {"churn": churn},
                  "constant_churn_latest": {"churn": latest_churn},
                  "shifted_beta_geometric": sbg}
    return predictions, parameters


def run_comparison(data_path, source_path, output_directory, *, plot=False):
    with Path(data_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    years = np.array([int(row["year"]) for row in rows])
    if not np.array_equal(years, np.arange(13)):
        raise ValueError("This experiment expects published years 0..12")
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    outputs, cohort_results, pooled_errors = [], {}, {name: [] for name in METHODS}
    for cohort, column in (("Regular", "regular_percent"), ("High End", "high_end_percent")):
        observed = validate_survival([float(row[column]) / 100 for row in rows])
        future = years[years > TRAIN_END]
        predictions, parameters = forecast_methods(observed[:TRAIN_END + 1], future)
        metrics = {}
        for method in METHODS:
            errors = 100 * (predictions[method] - observed[TRAIN_END + 1:])
            pooled_errors[method].extend(errors.tolist())
            metrics[method] = {
                "mae_percentage_points": float(np.abs(errors).mean()),
                "rmse_percentage_points": float(np.sqrt(np.mean(errors ** 2))),
                "holdout_points": len(future),
                "year_12_prediction_percent": float(predictions[method][-1] * 100),
                "year_12_actual_percent": float(observed[-1] * 100),
                "year_12_signed_error_percentage_points": float(errors[-1]),
                "year_12_relative_error_percent": float(errors[-1] / (observed[-1] * 100) * 100),
            }
        for i, year in enumerate(future):
            outputs.append({"cohort": cohort, "year": int(year),
                            "actual_percent": float(observed[year] * 100),
                            **{name + "_percent": float(predictions[name][i] * 100) for name in METHODS}})
        cohort_results[cohort] = {"parameters": parameters, "metrics": metrics}

    summary = {
        "source": source,
        "training_years": [0, TRAIN_END], "holdout_years": [TRAIN_END + 1, 12],
        "conditioning": "Every method starts at the observed survival at year 7",
        "baseline_scope": "Constant-churn projection implied by the core model at fixed price and zero acquisition, calibrated here only; not a test of the full planner",
        "estimation": "Normalized grouped censored likelihood on rounded cohort percentages; counts unavailable, no standard errors or confidence intervals claimed",
        "cohorts": cohort_results,
        "aggregate": {name: {
            "mae_percentage_points": float(np.abs(errors).mean()),
            "rmse_percentage_points": float(np.sqrt(np.mean(np.square(errors)))),
            "points": len(errors), "cohorts": len(cohort_results),
        } for name, errors in pooled_errors.items()},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "not_measured": ["Profit forecast error", "Minimum-capital forecast error",
                         "Joint reserve-violation probabilities", "Out-of-sample interval coverage",
                         "Policy improvement on real intervention data"],
    }
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "retention_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (directory / "retention_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outputs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(outputs)
    if plot:
        make_plot(rows, outputs, directory)
    return summary


def make_plot(rows, predictions, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    labels = {"constant_churn_mle": "Постоянный отток: все 7 лет",
              "constant_churn_latest": "Постоянный отток: последний год",
              "shifted_beta_geometric": "sBG: неоднородность клиентов"}
    colors = {"constant_churn_mle": "#c05b2b", "constant_churn_latest": "#3e70aa",
              "shifted_beta_geometric": "#258256"}
    for axis, (cohort, column) in zip(axes, (("Regular", "regular_percent"), ("High End", "high_end_percent"))):
        years = [int(row["year"]) for row in rows]
        observed = [float(row[column]) for row in rows]
        axis.axvspan(0, TRAIN_END, color="#e7e9ed", alpha=0.6)
        axis.plot(years, observed, "o-", color="#262626", markersize=3, label="Опубликованные наблюдения")
        subset = [row for row in predictions if row["cohort"] == cohort]
        for method in METHODS:
            axis.plot([TRAIN_END] + [row["year"] for row in subset],
                      [observed[TRAIN_END]] + [row[method + "_percent"] for row in subset],
                      "--", color=colors[method], linewidth=2, label=labels[method])
        axis.axvline(TRAIN_END, color="#999999", linestyle=":")
        axis.set(title=cohort, xlabel="Возраст когорты, годы", xlim=(0, 12), ylim=(0, 102))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Осталось от исходной когорты, %")
    handles, labels_list = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_list, loc="lower center", ncol=2, fontsize=9)
    figure.suptitle("Прогноз удержания: обучение 0–7 лет, проверка 8–12 лет", fontsize=13)
    figure.tight_layout(rect=(0, 0.17, 1, 0.94))
    figure.savefig(directory / "retention_comparison.png", dpi=180)
    figure.savefig(directory / "retention_comparison.svg")
    plt.close(figure)
    svg = directory / "retention_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=root / "data/fader_hardie_2007.csv")
    parser.add_argument("--source", type=Path, default=root / "data/retention_source.json")
    parser.add_argument("--output-dir", type=Path, default=root.parent / "docs/reports/assets/literature")
    parser.add_argument("--plot", action="store_true", help="Requires matplotlib (uv run --with matplotlib)")
    args = parser.parse_args()
    report = run_comparison(args.data, args.source, args.output_dir, plot=args.plot)
    print(json.dumps({"aggregate": report["aggregate"], "cohorts": report["cohorts"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
