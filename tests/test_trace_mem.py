import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_mem import (  # noqa: E402
    InteractionRiskModel,
    TemporalMemory,
    apply_risk_gate,
    claim_catalog,
    evaluate,
    execute_claim,
    expected_action,
    make_traces,
    write_outputs,
)


class TemporalMemoryTests(unittest.TestCase):
    def test_consolidation_supersedes_old_values(self):
        case = claim_catalog()[0]
        memory = TemporalMemory()
        for event in case.events:
            memory.write(event)
        memory.consolidate()
        self.assertEqual(memory.retrieve("address").value, "Basel")
        self.assertEqual(
            sum(record.status == "active" for record in memory.records if record.key == "address"),
            1,
        )
        memory.dispute("address")
        self.assertTrue(
            all(record.status != "active" for record in memory.records if record.key == "address")
        )


class AgentRuntimeTests(unittest.TestCase):
    def test_clean_claim_commits_expected_action(self):
        case = claim_catalog()[0]
        trace = execute_claim(case, [])
        self.assertTrue(trace.success)
        self.assertEqual(trace.proposed_action, expected_action(case))

    def test_memory_fault_interaction_is_visible(self):
        case = claim_catalog()[0]
        trace = execute_claim(case, ["consolidation", "retrieval"])
        self.assertFalse(trace.success)
        self.assertIn("active_conflict", trace.signals)
        self.assertIn("stale_retrieval", trace.signals)

    def test_gate_repairs_high_risk_action(self):
        case = claim_catalog()[0]
        model = InteractionRiskModel()
        model.fit(make_traces(2000, 42, "train"))
        trace = execute_claim(case, ["consolidation", "retrieval", "tool"])
        gated = apply_risk_gate(trace, case, model, threshold=0.50)
        self.assertTrue(gated.verified)
        self.assertFalse(gated.success)
        self.assertTrue(gated.committed_success)
        self.assertEqual(gated.committed_action, expected_action(case))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate(seed=42, train_runs=2500, test_runs=800)

    def test_gate_improves_success(self):
        metrics = self.result["metrics"]
        self.assertGreater(metrics["gated_success"], metrics["baseline_success"])

    def test_interaction_model_is_calibrated(self):
        metrics = self.result["metrics"]
        self.assertLessEqual(metrics["interaction_risk_brier"], metrics["flat_risk_brier"])

    def test_expected_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            write_outputs(self.result, target)
            for name in ("dashboard.html", "results.json", "scenario_results.csv", "reliability_map.csv"):
                self.assertGreater((target / name).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
