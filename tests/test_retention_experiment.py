import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.retention_forecasts import (
    constant_churn_mle, fit_sbg, run_comparison, sbg_survival,
)


class RetentionExperimentTests(unittest.TestCase):
    def test_constant_hazard_is_recovered_from_censored_survival(self):
        self.assertAlmostEqual(constant_churn_mle(0.8 ** np.arange(8)), 0.2)

    def test_uniform_beta_mixture_has_known_survival(self):
        periods = np.arange(13)
        np.testing.assert_allclose(sbg_survival(1, 1, periods), 1 / (periods + 1))

    def test_grouped_censored_likelihood_recovers_known_mixture(self):
        fitted = fit_sbg(1 / (np.arange(8) + 1))
        self.assertAlmostEqual(fitted["alpha"], 1, places=3)
        self.assertAlmostEqual(fitted["beta"], 1, places=3)

    def test_changing_holdout_does_not_change_parameters_or_predictions(self):
        source_data = Path("experiments/data/fader_hardie_2007.csv")
        source_meta = Path("experiments/data/retention_source.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_comparison(source_data, source_meta, root / "first")
            with source_data.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if int(row["year"]) > 7:
                    row["regular_percent"] = str(float(row["regular_percent"]) * 0.8)
                    row["high_end_percent"] = str(float(row["high_end_percent"]) * 0.8)
            modified = root / "modified.csv"
            with modified.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            second = run_comparison(modified, source_meta, root / "second")
            for cohort in first["cohorts"]:
                self.assertEqual(first["cohorts"][cohort]["parameters"], second["cohorts"][cohort]["parameters"])
            def predictions(folder):
                with (root / folder / "retention_predictions.csv").open(newline="") as stream:
                    return [{k: v for k, v in row.items() if k != "actual_percent"}
                            for row in csv.DictReader(stream)]
            self.assertEqual(predictions("first"), predictions("second"))
            self.assertNotEqual(first["aggregate"], second["aggregate"])


if __name__ == "__main__":
    unittest.main()
