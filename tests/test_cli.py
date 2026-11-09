from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

from function_space_model.cli import main
from test_model import small_config


class CliTests(unittest.TestCase):
    def test_optimize_export_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(asdict(small_config())))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["optimize", str(config), "--output-dir", str(root / "search")]), 0)
                self.assertEqual(main(["simulate", str(config), str(root / "search/plan.json"),
                                       "--output-dir", str(root / "replay")]), 0)
            search = json.loads((root / "search/report.json").read_text())
            replay = json.loads((root / "replay/report.json").read_text())
            self.assertEqual(search["simulations"], replay["simulations"])
            self.assertEqual(search["configuration"], json.loads(config.read_text()))
            self.assertEqual((root / "search/trajectory.csv").read_text(),
                             (root / "replay/trajectory.csv").read_text())

    def test_failed_run_removes_old_exported_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(asdict(small_config(initial_cash=120, fixed_monthly_cost=30))))
            output = root / "result"
            output.mkdir()
            (output / "plan.json").write_text("stale plan")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["optimize", str(config), "--output-dir", str(output)]), 2)
            self.assertFalse((output / "plan.json").exists())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["solver_status"], "infeasible_on_grid")

    def test_constant_comparison_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(asdict(small_config())))
            output = root / "comparison"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["compare-constant", str(config), "--output-dir", str(output)]), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["constant_plans_checked"], 4)
            best = report["best_constant_plan"]
            self.assertEqual(best["worst_profit"], max(record["worst_profit"]
                             for record in report["constant_plans"] if record["feasible_in_all_models"]))


if __name__ == "__main__":
    unittest.main()
