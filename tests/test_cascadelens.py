import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascadelens import evaluate, write_outputs  # noqa: E402


class CascadeLensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate(seed=42)

    def test_topology_model_improves_group_mae(self):
        metrics = self.result["metrics"]
        self.assertLess(metrics["topology_model_mae"], metrics["baseline_mae"])

    def test_recommendation_respects_budget(self):
        self.assertLessEqual(
            self.result["recommendation"]["cost_units"],
            self.result["budget"],
        )

    def test_expected_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            write_outputs(self.result, target)
            self.assertGreater((target / "dashboard.html").stat().st_size, 1000)
            self.assertGreater((target / "results.json").stat().st_size, 1000)
            self.assertGreater((target / "scenario_results.csv").stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
