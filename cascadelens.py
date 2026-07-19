#!/usr/bin/env python3
"""CascadeLens: dependency-aware reliability forecasting PoC.

This intentionally dependency-free prototype creates synthetic, privacy-safe
agent traces, learns conditional error-propagation rates, compares them with an
independence baseline, and recommends verification placement under a budget.

The synthetic environment is a methodological smoke test, not evidence of
real-world insurance performance.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SEED = 42


@dataclass(frozen=True)
class Node:
    name: str
    kind: str
    parents: Tuple[str, ...] = ()
    repair_target: str | None = None
    repair_strength: float = 0.0


@dataclass(frozen=True)
class Workflow:
    name: str
    label: str
    nodes: Tuple[Node, ...]
    terminal: str


@dataclass
class StepRecord:
    node: str
    kind: str
    parents: List[str]
    parent_failed: bool
    probability: float
    failed: bool
    repaired: str | None = None


@dataclass
class RunRecord:
    workflow: str
    complexity: int
    stress: float
    success: bool
    steps: List[StepRecord]


BASE_FAILURE = {
    "retrieval": 0.055,
    "reasoning": 0.050,
    "tool": 0.045,
    "action": 0.025,
    "verify": 0.035,
}

STRESS_SENSITIVITY = {
    "retrieval": 0.12,
    "reasoning": 0.07,
    "tool": 0.18,
    "action": 0.05,
    "verify": 0.06,
}

PROPAGATION = {
    "retrieval": 0.22,
    "reasoning": 0.63,
    "tool": 0.34,
    "action": 0.76,
    "verify": 0.18,
}


def workflow_catalog() -> Dict[str, Workflow]:
    return {
        "linear": Workflow(
            name="linear",
            label="Linear claim review",
            terminal="record_decision",
            nodes=(
                Node("retrieve_policy", "retrieval"),
                Node("interpret_coverage", "reasoning", ("retrieve_policy",)),
                Node("lookup_claim", "tool", ("interpret_coverage",)),
                Node("apply_rules", "reasoning", ("lookup_claim",)),
                Node("record_decision", "action", ("apply_rules",)),
            ),
        ),
        "branch": Workflow(
            name="branch",
            label="Branch-and-merge review",
            terminal="record_decision",
            nodes=(
                Node("retrieve_policy", "retrieval"),
                Node("lookup_claim", "tool"),
                Node("check_coverage", "reasoning", ("retrieve_policy",)),
                Node("check_history", "reasoning", ("lookup_claim",)),
                Node("merge_evidence", "reasoning", ("check_coverage", "check_history")),
                Node("record_decision", "action", ("merge_evidence",)),
            ),
        ),
        "recoverable": Workflow(
            name="recoverable",
            label="Recovery-aware review",
            terminal="record_decision",
            nodes=(
                Node("retrieve_policy", "retrieval"),
                Node(
                    "verify_policy",
                    "verify",
                    ("retrieve_policy",),
                    repair_target="retrieve_policy",
                    repair_strength=0.78,
                ),
                Node("interpret_coverage", "reasoning", ("retrieve_policy", "verify_policy")),
                Node("lookup_claim", "tool", ("interpret_coverage",)),
                Node(
                    "verify_tool_result",
                    "verify",
                    ("lookup_claim",),
                    repair_target="lookup_claim",
                    repair_strength=0.68,
                ),
                Node("record_decision", "action", ("interpret_coverage", "lookup_claim", "verify_tool_result")),
            ),
        ),
        "long_horizon": Workflow(
            name="long_horizon",
            label="Long-horizon renewal review",
            terminal="record_decision",
            nodes=(
                Node("retrieve_policy", "retrieval"),
                Node("retrieve_prior_case", "retrieval"),
                Node("lookup_claim", "tool"),
                Node("check_coverage", "reasoning", ("retrieve_policy",)),
                Node("compare_prior_case", "reasoning", ("retrieve_prior_case", "lookup_claim")),
                Node("request_external_score", "tool", ("lookup_claim",)),
                Node("merge_evidence", "reasoning", ("check_coverage", "compare_prior_case", "request_external_score")),
                Node("record_decision", "action", ("merge_evidence",)),
            ),
        ),
    }


def guarded_branch(checkpoint: str) -> Workflow:
    nodes: List[Node] = [Node("retrieve_policy", "retrieval"), Node("lookup_claim", "tool")]
    if checkpoint in {"retrieval", "both"}:
        nodes.append(
            Node(
                "verify_policy",
                "verify",
                ("retrieve_policy",),
                repair_target="retrieve_policy",
                repair_strength=0.70,
            )
        )
        coverage_parents = ("retrieve_policy", "verify_policy")
    else:
        coverage_parents = ("retrieve_policy",)
    if checkpoint in {"tool", "both"}:
        nodes.append(
            Node(
                "verify_claim_lookup",
                "verify",
                ("lookup_claim",),
                repair_target="lookup_claim",
                repair_strength=0.82,
            )
        )
        history_parents = ("lookup_claim", "verify_claim_lookup")
    else:
        history_parents = ("lookup_claim",)
    nodes.extend(
        [
            Node("check_coverage", "reasoning", coverage_parents),
            Node("check_history", "reasoning", history_parents),
            Node("merge_evidence", "reasoning", ("check_coverage", "check_history")),
            Node("record_decision", "action", ("merge_evidence",)),
        ]
    )
    return Workflow(
        name=f"branch_{checkpoint}",
        label={
            "none": "No checkpoint",
            "retrieval": "Verify retrieval",
            "tool": "Verify tool result",
            "both": "Verify retrieval + tool",
        }[checkpoint],
        nodes=tuple(nodes),
        terminal="record_decision",
    )


def _clip(value: float, low: float = 0.002, high: float = 0.97) -> float:
    return max(low, min(high, value))


def simulate_run(
    workflow: Workflow,
    complexity: int,
    stress: float,
    rng: random.Random,
) -> RunRecord:
    """Execute one stochastic trace with explicit dependency propagation."""
    failed: MutableMapping[str, bool] = {}
    records: List[StepRecord] = []
    # A latent episode shock induces correlation that a product baseline misses.
    episode_shock = rng.random() < (0.045 + 0.035 * complexity + 0.09 * stress)

    for node in workflow.nodes:
        parent_failed = any(failed.get(parent, False) for parent in node.parents)
        repaired: str | None = None

        if node.kind == "verify" and node.repair_target:
            target_failed = failed.get(node.repair_target, False)
            if target_failed:
                repair_probability = _clip(
                    node.repair_strength - 0.12 * stress - 0.035 * complexity,
                    0.10,
                    0.95,
                )
                verification_failed = rng.random() >= repair_probability
                if not verification_failed:
                    failed[node.repair_target] = False
                    repaired = node.repair_target
            else:
                # A checkpoint that sees a valid result is a no-op. We model
                # recovery failure, not false-positive blocking, in this PoC.
                verification_failed = False
            failed[node.name] = verification_failed
            records.append(
                StepRecord(
                    node=node.name,
                    kind=node.kind,
                    parents=list(node.parents),
                    parent_failed=parent_failed,
                    probability=round(1.0 - repair_probability, 5) if target_failed else round(BASE_FAILURE["verify"], 5),
                    failed=verification_failed,
                    repaired=repaired,
                )
            )
            continue

        parent_fail_count = sum(1 for parent in node.parents if failed.get(parent, False))
        probability = (
            BASE_FAILURE[node.kind]
            + STRESS_SENSITIVITY[node.kind] * stress
            + 0.025 * complexity
            + PROPAGATION[node.kind] * min(1.0, parent_fail_count / max(1, len(node.parents)))
            + (0.085 if episode_shock else 0.0)
        )
        probability = _clip(probability)
        node_failed = rng.random() < probability
        failed[node.name] = node_failed
        records.append(
            StepRecord(
                node=node.name,
                kind=node.kind,
                parents=list(node.parents),
                parent_failed=parent_failed,
                probability=round(probability, 5),
                failed=node_failed,
            )
        )

    success = not failed[workflow.terminal]
    return RunRecord(
        workflow=workflow.name,
        complexity=complexity,
        stress=stress,
        success=success,
        steps=records,
    )


class ConditionalPropagationModel:
    """Laplace-smoothed conditional failure model over typed trace DAGs."""

    def __init__(self) -> None:
        self.counts: Dict[Tuple[str, bool, int, int], List[int]] = defaultdict(lambda: [0, 0])
        self.fallback: Dict[Tuple[str, bool], List[int]] = defaultdict(lambda: [0, 0])

    @staticmethod
    def stress_bucket(stress: float) -> int:
        return 0 if stress < 0.25 else 1 if stress < 0.65 else 2

    def fit(self, runs: Iterable[RunRecord]) -> None:
        for run in runs:
            bucket = self.stress_bucket(run.stress)
            for step in run.steps:
                key = (step.kind, step.parent_failed, run.complexity, bucket)
                self.counts[key][0] += int(step.failed)
                self.counts[key][1] += 1
                fallback_key = (step.kind, step.parent_failed)
                self.fallback[fallback_key][0] += int(step.failed)
                self.fallback[fallback_key][1] += 1

    @staticmethod
    def _rate(pair: Sequence[int], alpha: float = 1.0, beta: float = 1.0) -> float:
        failures, total = pair
        return (failures + alpha) / (total + alpha + beta)

    def probability(self, kind: str, parent_failed: bool, complexity: int, stress: float) -> float:
        key = (kind, parent_failed, complexity, self.stress_bucket(stress))
        if key in self.counts and self.counts[key][1] >= 8:
            return self._rate(self.counts[key])
        return self._rate(self.fallback[(kind, parent_failed)])

    def estimate_success(
        self,
        workflow: Workflow,
        complexity: int,
        stress: float,
        seed: int,
        samples: int = 3500,
    ) -> float:
        rng = random.Random(seed)
        successes = 0
        for _ in range(samples):
            failed: MutableMapping[str, bool] = {}
            for node in workflow.nodes:
                parent_failed = any(failed.get(parent, False) for parent in node.parents)
                if node.kind == "verify" and node.repair_target:
                    if not parent_failed:
                        node_failed = False
                    else:
                        repair_probability = _clip(
                            node.repair_strength - 0.12 * stress - 0.035 * complexity,
                            0.10,
                            0.95,
                        )
                        node_failed = rng.random() >= repair_probability
                else:
                    p = self.probability(node.kind, parent_failed, complexity, stress)
                    node_failed = rng.random() < p
                failed[node.name] = node_failed
                if node.kind == "verify" and node.repair_target and parent_failed and not node_failed:
                    failed[node.repair_target] = False
            successes += int(not failed[workflow.terminal])
        return successes / samples


def independence_estimate(
    workflow: Workflow,
    model: ConditionalPropagationModel,
    complexity: int,
    stress: float,
) -> float:
    """Product of isolated component success, ignoring all dependencies."""
    probability = 1.0
    for node in workflow.nodes:
        if node.kind != "verify":
            probability *= 1.0 - model.probability(node.kind, False, complexity, stress)
    return probability


def empirical_success(
    workflow: Workflow,
    complexity: int,
    stress: float,
    seed: int,
    runs: int,
) -> Tuple[float, List[RunRecord]]:
    rng = random.Random(seed)
    records = [simulate_run(workflow, complexity, stress, rng) for _ in range(runs)]
    return sum(record.success for record in records) / runs, records


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def evaluate(seed: int = SEED) -> Dict[str, object]:
    catalog = workflow_catalog()
    train_runs: List[RunRecord] = []
    train_rng = random.Random(seed)
    for workflow in catalog.values():
        for complexity in (0, 1, 2):
            for stress in (0.0, 0.45, 0.90):
                train_runs.extend(simulate_run(workflow, complexity, stress, train_rng) for _ in range(260))

    model = ConditionalPropagationModel()
    model.fit(train_runs)

    scenarios: List[Dict[str, object]] = []
    trace_example: RunRecord | None = None
    for w_index, workflow in enumerate(catalog.values()):
        for complexity in (0, 1, 2):
            for s_index, stress in enumerate((0.15, 0.60, 0.95)):
                actual, test_records = empirical_success(
                    workflow,
                    complexity,
                    stress,
                    seed + 1000 + w_index * 100 + complexity * 10 + s_index,
                    800,
                )
                predicted = model.estimate_success(
                    workflow,
                    complexity,
                    stress,
                    seed + 3000 + w_index * 100 + complexity * 10 + s_index,
                )
                baseline = independence_estimate(workflow, model, complexity, stress)
                scenarios.append(
                    {
                        "workflow": workflow.name,
                        "label": workflow.label,
                        "complexity": complexity,
                        "stress": stress,
                        "actual": round(actual, 4),
                        "topology_model": round(predicted, 4),
                        "independence_baseline": round(baseline, 4),
                    }
                )
                if trace_example is None:
                    trace_example = next((record for record in test_records if not record.success), None)

    baseline_errors = [abs(float(row["independence_baseline"]) - float(row["actual"])) for row in scenarios]
    model_errors = [abs(float(row["topology_model"]) - float(row["actual"])) for row in scenarios]
    baseline_brier = [
        (float(row["independence_baseline"]) - float(row["actual"])) ** 2 for row in scenarios
    ]
    model_brier = [(float(row["topology_model"]) - float(row["actual"])) ** 2 for row in scenarios]

    checkpoint_cost = {"none": 0.0, "retrieval": 1.0, "tool": 1.0, "both": 2.0}
    interventions: List[Dict[str, object]] = []
    for index, checkpoint in enumerate(("none", "retrieval", "tool", "both")):
        workflow = guarded_branch(checkpoint)
        prediction = model.estimate_success(
            workflow,
            1,
            0.60,
            seed + 7000,
            samples=12000,
        )
        actual, _ = empirical_success(workflow, 1, 0.60, seed + 7100 + index, 2200)
        interventions.append(
            {
                "checkpoint": checkpoint,
                "label": workflow.label,
                "cost_units": checkpoint_cost[checkpoint],
                "predicted_success": round(prediction, 4),
                "actual_success": round(actual, 4),
            }
        )

    budget = 1.0
    eligible = [row for row in interventions if float(row["cost_units"]) <= budget]
    # The PoC uses a small controlled replay budget to rank concrete guards;
    # the six-month project studies how far this can be reduced or replaced by
    # calibrated forecasts.
    recommendation = max(eligible, key=lambda row: float(row["actual_success"]))

    return {
        "meta": {
            "name": "CascadeLens",
            "seed": seed,
            "training_traces": len(train_runs),
            "test_scenarios": len(scenarios),
            "test_runs_per_scenario": 800,
            "domain": "synthetic insurance-like claim workflows",
            "scope_note": "Methodological PoC only; no customer data and no claim about field performance.",
        },
        "metrics": {
            "baseline_mae": round(mean(baseline_errors), 4),
            "topology_model_mae": round(mean(model_errors), 4),
            "relative_mae_reduction": round(1.0 - mean(model_errors) / mean(baseline_errors), 4),
            "baseline_brier": round(mean(baseline_brier), 5),
            "topology_model_brier": round(mean(model_brier), 5),
        },
        "scenarios": scenarios,
        "interventions": interventions,
        "budget": budget,
        "recommendation_basis": "controlled replay under the synthetic environment",
        "recommendation": recommendation,
        "trace_example": asdict(trace_example) if trace_example else None,
    }


def write_csv(result: Mapping[str, object], path: Path) -> None:
    rows = list(result["scenarios"])  # type: ignore[arg-type]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_dashboard(result: Mapping[str, object]) -> str:
    metrics = result["metrics"]  # type: ignore[assignment]
    scenarios = result["scenarios"]  # type: ignore[assignment]
    interventions = result["interventions"]  # type: ignore[assignment]
    recommendation = result["recommendation"]  # type: ignore[assignment]
    trace = result["trace_example"]  # type: ignore[assignment]

    # Show one representative medium-complexity scenario per topology.
    featured = [
        row
        for row in scenarios
        if row["complexity"] == 1 and math.isclose(float(row["stress"]), 0.60)
    ]
    scenario_cards = "\n".join(
        f"""
        <article class="scenario-card">
          <div class="scenario-head"><strong>{html.escape(str(row['label']))}</strong><span>stress {float(row['stress']):.2f}</span></div>
          <div class="bar-row"><label>Observed</label><div class="bar"><i style="width:{float(row['actual'])*100:.1f}%"></i></div><b>{pct(float(row['actual']))}</b></div>
          <div class="bar-row"><label>CascadeLens</label><div class="bar"><i class="cyan" style="width:{float(row['topology_model'])*100:.1f}%"></i></div><b>{pct(float(row['topology_model']))}</b></div>
          <div class="bar-row"><label>Independence</label><div class="bar"><i class="amber" style="width:{float(row['independence_baseline'])*100:.1f}%"></i></div><b>{pct(float(row['independence_baseline']))}</b></div>
        </article>
        """.strip()
        for row in featured
    )
    intervention_rows = "\n".join(
        f"""
        <tr class="{'chosen' if row['checkpoint'] == recommendation['checkpoint'] else ''}">
          <td>{html.escape(str(row['label']))}</td>
          <td>{float(row['cost_units']):.1f}</td>
          <td>{pct(float(row['predicted_success']))}</td>
          <td>{pct(float(row['actual_success']))}</td>
          <td>{'RECOMMENDED' if row['checkpoint'] == recommendation['checkpoint'] else ''}</td>
        </tr>
        """.strip()
        for row in interventions
    )
    trace_steps = "\n".join(
        f"""
        <div class="trace-step {'failed' if step['failed'] else 'ok'}">
          <span class="dot"></span>
          <div><strong>{html.escape(step['node'].replace('_', ' ').title())}</strong><small>{html.escape(step['kind'])} &middot; p(fail) {float(step['probability'])*100:.1f}%</small></div>
          <b>{'FAIL' if step['failed'] else 'PASS'}</b>
        </div>
        """.strip()
        for step in (trace["steps"] if trace else [])
    )
    result_json = json.dumps(result, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CascadeLens PoC</title>
<style>
:root{{--navy:#0b1830;--ink:#18243a;--muted:#64748b;--line:#dbe4ef;--cyan:#12b8c4;--blue:#2463eb;--amber:#e6a624;--red:#dc3c4b;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}}
.shell{{max-width:1180px;margin:auto;padding:34px 24px 60px}} .hero{{background:linear-gradient(130deg,var(--navy),#173b72);color:white;border-radius:22px;padding:34px 36px;box-shadow:0 18px 50px #0b183026}}
.eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:12px;color:#8fe7ed;font-weight:800}} h1{{margin:8px 0 6px;font-size:46px;line-height:1}} .hero p{{max-width:820px;color:#dbe7f8;font-size:17px;line-height:1.55;margin:13px 0 0}}
.pipeline{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:26px}} .stage{{background:#ffffff12;border:1px solid #ffffff25;padding:14px;border-radius:12px}} .stage b{{display:block;font-size:13px;color:#9be7ed;margin-bottom:4px}} .stage span{{font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}} .kpi,.panel,.scenario-card{{background:white;border:1px solid var(--line);border-radius:16px;box-shadow:0 9px 30px #1831530b}}
.kpi{{padding:18px 20px}} .kpi small{{color:var(--muted);font-weight:700}} .kpi strong{{display:block;font-size:28px;margin-top:6px;color:var(--navy)}} .kpi em{{font-style:normal;color:#0d8c74;font-size:12px;font-weight:700}}
.grid{{display:grid;grid-template-columns:1.35fr .85fr;gap:18px}} .panel{{padding:22px}} h2{{margin:0 0 4px;font-size:21px;color:var(--navy)}} .sub{{color:var(--muted);font-size:13px;margin:0 0 18px}}
.scenario-card{{padding:16px;margin-top:12px;box-shadow:none}} .scenario-head{{display:flex;justify-content:space-between;margin-bottom:12px}} .scenario-head span{{font-size:12px;color:var(--muted)}} .bar-row{{display:grid;grid-template-columns:95px 1fr 48px;gap:9px;align-items:center;font-size:12px;margin:8px 0}} .bar{{height:9px;border-radius:9px;background:#edf2f7;overflow:hidden}} .bar i{{display:block;height:100%;background:var(--blue);border-radius:9px}} .bar i.cyan{{background:var(--cyan)}} .bar i.amber{{background:var(--amber)}}
.trace-step{{display:grid;grid-template-columns:12px 1fr auto;gap:10px;align-items:center;padding:11px 0;border-bottom:1px solid #edf1f6}} .trace-step .dot{{width:10px;height:10px;border-radius:50%;background:#22a06b}} .trace-step.failed .dot{{background:var(--red)}} .trace-step small{{display:block;color:var(--muted);margin-top:2px}} .trace-step>b{{font-size:11px;color:#11845d}} .trace-step.failed>b{{color:var(--red)}}
.wide{{margin-top:18px}} table{{width:100%;border-collapse:collapse;font-size:13px}} th{{text-align:left;color:var(--muted);padding:10px;border-bottom:1px solid var(--line)}} td{{padding:12px 10px;border-bottom:1px solid #edf1f6}} tr.chosen{{background:#e9fbf7}} tr.chosen td:last-child{{color:#08745e;font-weight:900;font-size:11px}}
.note{{border-left:4px solid var(--amber);padding:12px 14px;background:#fff8e6;color:#5d4b19;border-radius:7px;font-size:13px;margin-top:16px}} footer{{color:var(--muted);font-size:12px;margin-top:18px;text-align:center}}
@media(max-width:800px){{.kpis,.pipeline{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}h1{{font-size:36px}}}} @media(max-width:520px){{.kpis,.pipeline{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main class="shell">
<section class="hero"><div class="eyebrow">Research proof-of-concept &middot; seed {result['meta']['seed']}</div><h1>CascadeLens</h1><p>Compile a multi-step agent workflow into a typed dependency graph, learn how local errors propagate, forecast end-to-end reliability, and place the most valuable verification checkpoint under a cost budget.</p>
<div class="pipeline"><div class="stage"><b>01 &middot; TRACE</b><span>Typed reasoning, retrieval, tool and action nodes</span></div><div class="stage"><b>02 &middot; PERTURB</b><span>Controlled faults and stochastic replay</span></div><div class="stage"><b>03 &middot; FORECAST</b><span>Conditional propagation over the workflow DAG</span></div><div class="stage"><b>04 &middot; INTERVENE</b><span>Budget-aware checkpoint recommendation</span></div></div></section>
<section class="kpis"><div class="kpi"><small>Independence MAE</small><strong>{float(metrics['baseline_mae']):.3f}</strong><span>group-level error</span></div><div class="kpi"><small>Topology-aware MAE</small><strong>{float(metrics['topology_model_mae']):.3f}</strong><em>{float(metrics['relative_mae_reduction'])*100:.0f}% lower in this synthetic run</em></div><div class="kpi"><small>Training traces</small><strong>{int(result['meta']['training_traces']):,}</strong><span>privacy-safe simulation</span></div><div class="kpi"><small>Budget recommendation</small><strong style="font-size:20px">{html.escape(str(recommendation['label']))}</strong><span>cost &le; {float(result['budget']):.1f}</span></div></section>
<section class="grid"><div class="panel"><h2>Reliability forecasts</h2><p class="sub">Medium-complexity workflows under production-like stress. Bars compare observed success with two estimators.</p>{scenario_cards}</div><div class="panel"><h2>Representative failed trace</h2><p class="sub">The terminal error is downstream; the dependency path reveals where risk first entered.</p>{trace_steps}</div></section>
<section class="panel wide"><h2>Verification placement</h2><p class="sub">One checkpoint costs one abstract latency unit. A small controlled-replay budget ranks eligible interventions; the thesis tests when calibrated forecasts can reduce that budget.</p><table><thead><tr><th>Configuration</th><th>Cost</th><th>Predicted</th><th>Observed</th><th></th></tr></thead><tbody>{intervention_rows}</tbody></table><div class="note"><strong>Scope:</strong> {html.escape(str(result['meta']['scope_note']))} The thesis would replace the simulator with real agent traces and deterministic end-state evaluators from domains such as insurance, browser use, or enterprise workflows.</div></section>
<footer>Generated by the reproducible CascadeLens PoC &middot; no external packages or API keys required</footer>
</main><script>window.CASCADELENS_DATA={result_json};</script></body></html>"""


def write_outputs(result: Mapping[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(result, output_dir / "scenario_results.csv")
    (output_dir / "dashboard.html").write_text(render_dashboard(result), encoding="utf-8")


def print_summary(result: Mapping[str, object]) -> None:
    metrics = result["metrics"]  # type: ignore[assignment]
    recommendation = result["recommendation"]  # type: ignore[assignment]
    print("CascadeLens synthetic evaluation")
    print(f"  training traces: {result['meta']['training_traces']:,}")  # type: ignore[index]
    print(f"  independence MAE: {metrics['baseline_mae']:.4f}")
    print(f"  topology-aware MAE: {metrics['topology_model_mae']:.4f}")
    print(f"  relative MAE reduction: {metrics['relative_mae_reduction'] * 100:.1f}%")
    print(
        "  recommended checkpoint (budget 1.0): "
        f"{recommendation['label']} -> replayed {pct(recommendation['actual_success'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = evaluate(seed=args.seed)
    write_outputs(result, args.output_dir)
    print_summary(result)


if __name__ == "__main__":
    main()
