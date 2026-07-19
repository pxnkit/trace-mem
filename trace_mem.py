#!/usr/bin/env python3
"""TRACE-Mem: causal reliability maps for persistent AI agents."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SEED = 42
FAULTS = ("write", "consolidation", "retrieval", "reasoning", "tool")
SIGNALS = (
    "write_gap",
    "active_conflict",
    "stale_retrieval",
    "reasoning_mismatch",
    "tool_mismatch",
)


@dataclass(frozen=True)
class SourceEvent:
    key: str
    value: str
    valid_from: int
    recorded_at: int


@dataclass
class MemoryRecord:
    key: str
    value: str
    valid_from: int
    recorded_at: int
    status: str = "active"


class TemporalMemory:
    """Append-only memory that separates event time from recording time."""

    def __init__(self) -> None:
        self.records: List[MemoryRecord] = []

    def write(self, event: SourceEvent) -> None:
        self.records.append(MemoryRecord(**asdict(event)))

    def consolidate(self, preserve_old_active: bool = False) -> None:
        by_key: Dict[str, List[MemoryRecord]] = defaultdict(list)
        for record in self.records:
            by_key[record.key].append(record)
        for records in by_key.values():
            latest = max(records, key=lambda item: (item.valid_from, item.recorded_at))
            for record in records:
                if record is latest:
                    record.status = "active"
                elif not preserve_old_active:
                    record.status = "superseded"

    def retrieve(self, key: str, prefer_oldest: bool = False) -> MemoryRecord:
        active = [r for r in self.records if r.key == key and r.status == "active"]
        if not active:
            raise KeyError(f"no active memory for {key}")
        order = lambda item: (item.valid_from, item.recorded_at)
        return (min if prefer_oldest else max)(active, key=order)

    def dispute(self, key: str) -> None:
        """Mark every active value for a key as unsafe for autonomous use."""
        for record in self.records:
            if record.key == key and record.status == "active":
                record.status = "disputed"


@dataclass(frozen=True)
class ClaimCase:
    case_id: str
    amount: int
    events: Tuple[SourceEvent, ...]

    def canonical(self) -> Dict[str, str]:
        result: Dict[str, SourceEvent] = {}
        for event in self.events:
            current = result.get(event.key)
            if current is None or (event.valid_from, event.recorded_at) > (
                current.valid_from,
                current.recorded_at,
            ):
                result[event.key] = event
        return {key: event.value for key, event in result.items()}


@dataclass
class AgentTrace:
    trace_id: str
    case_id: str
    faults: List[str]
    signals: List[str]
    memory: List[MemoryRecord]
    retrieved: Dict[str, str]
    proposed_action: Dict[str, str]
    expected_action: Dict[str, str]
    success: bool
    committed_success: bool
    risk: float = 0.0
    gated: bool = False
    verified: bool = False
    committed_action: Dict[str, str] = field(default_factory=dict)


def claim_catalog() -> Tuple[ClaimCase, ...]:
    def events(*rows: Tuple[str, str, int, int]) -> Tuple[SourceEvent, ...]:
        return tuple(SourceEvent(*row) for row in rows)

    return (
        ClaimCase(
            "CLM-1042",
            4200,
            events(
                ("address", "Zurich", 1, 1),
                ("coverage", "basic", 1, 2),
                ("channel", "phone", 1, 3),
                ("address", "Basel", 8, 9),
                ("coverage", "premium", 10, 11),
                ("channel", "email", 12, 13),
            ),
        ),
        ClaimCase(
            "CLM-2088",
            6800,
            events(
                ("address", "Bern", 1, 1),
                ("coverage", "premium", 1, 2),
                ("channel", "email", 1, 3),
                ("address", "Geneva", 7, 8),
                ("coverage", "basic", 9, 10),
                ("channel", "secure_portal", 11, 12),
            ),
        ),
        ClaimCase(
            "CLM-3315",
            1900,
            events(
                ("address", "Lausanne", 1, 1),
                ("coverage", "basic", 1, 2),
                ("channel", "phone", 1, 3),
                ("address", "Lugano", 5, 6),
                ("coverage", "premium", 6, 7),
                ("channel", "email", 8, 9),
            ),
        ),
    )


def decide(coverage: str, amount: int) -> str:
    return "approve" if coverage == "premium" and amount <= 5000 else "manual_review"


def expected_action(case: ClaimCase) -> Dict[str, str]:
    state = case.canonical()
    return {
        "case_id": case.case_id,
        "decision": decide(state["coverage"], case.amount),
        "notify": state["channel"],
        "address": state["address"],
    }


def execute_claim(case: ClaimCase, faults: Iterable[str], trace_id: str = "trace") -> AgentTrace:
    """Run the memory-to-action path and stage, but do not hide, its faults."""
    injected = set(faults)
    memory = TemporalMemory()
    newest_coverage = max(
        (e for e in case.events if e.key == "coverage"),
        key=lambda event: event.valid_from,
    )
    for event in case.events:
        if "write" in injected and event == newest_coverage:
            continue
        memory.write(event)
    memory.consolidate(preserve_old_active="consolidation" in injected)

    retrieved_records = {
        key: memory.retrieve(key, prefer_oldest="retrieval" in injected)
        for key in ("address", "coverage", "channel")
    }
    retrieved = {key: record.value for key, record in retrieved_records.items()}
    decision = decide(retrieved["coverage"], case.amount)
    if "reasoning" in injected:
        decision = "manual_review" if decision == "approve" else "approve"

    proposed = {
        "case_id": case.case_id,
        "decision": decision,
        "notify": retrieved["channel"],
        "address": retrieved["address"],
    }
    if "tool" in injected:
        proposed["notify"] = "phone" if proposed["notify"] != "phone" else "email"

    canonical = case.canonical()
    expected = expected_action(case)
    stored_latest = {
        key: max(
            (r for r in memory.records if r.key == key),
            key=lambda item: (item.valid_from, item.recorded_at),
        ).value
        for key in canonical
    }
    signals: List[str] = []
    if any(stored_latest.get(key) != value for key, value in canonical.items()):
        signals.append("write_gap")
    if any(
        sum(1 for r in memory.records if r.key == key and r.status == "active") > 1
        for key in canonical
    ):
        signals.append("active_conflict")
    if any(retrieved.get(key) != value for key, value in canonical.items()):
        signals.append("stale_retrieval")
    correct_plan = decide(retrieved["coverage"], case.amount)
    if decision != correct_plan:
        signals.append("reasoning_mismatch")
    staged_expected = {
        "case_id": case.case_id,
        "decision": decision,
        "notify": retrieved["channel"],
        "address": retrieved["address"],
    }
    if proposed != staged_expected:
        signals.append("tool_mismatch")

    return AgentTrace(
        trace_id=trace_id,
        case_id=case.case_id,
        faults=sorted(injected),
        signals=signals,
        memory=memory.records,
        retrieved=retrieved,
        proposed_action=proposed,
        expected_action=expected,
        success=proposed == expected,
        committed_success=proposed == expected,
        committed_action=proposed.copy(),
    )


def inject_faults(rng: random.Random) -> List[str]:
    rates = {
        "write": 0.08,
        "consolidation": 0.13,
        "retrieval": 0.11,
        "reasoning": 0.08,
        "tool": 0.07,
    }
    active = {fault for fault, rate in rates.items() if rng.random() < rate}
    if rng.random() < 0.10:
        # Real incidents often affect adjacent components together.
        active.update(rng.choice((("write", "retrieval"), ("consolidation", "retrieval"), ("reasoning", "tool"))))
    return sorted(active)


def make_traces(count: int, seed: int, prefix: str) -> List[AgentTrace]:
    rng = random.Random(seed)
    cases = claim_catalog()
    return [
        execute_claim(cases[index % len(cases)], inject_faults(rng), f"{prefix}-{index:05d}")
        for index in range(count)
    ]


class InteractionRiskModel:
    """Smoothed empirical risk model over complete signal interactions."""

    def __init__(self) -> None:
        self.signature_counts: Dict[Tuple[str, ...], List[int]] = defaultdict(lambda: [0, 0])
        self.signal_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        self.pair_counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
        self.total = [0, 0]

    @staticmethod
    def _rate(pair: Sequence[int], alpha: float = 1.0, beta: float = 1.0) -> float:
        failures, total = pair
        return (failures + alpha) / (total + alpha + beta)

    def fit(self, traces: Iterable[AgentTrace]) -> None:
        for trace in traces:
            failed = int(not trace.success)
            signature = tuple(sorted(trace.signals))
            self.signature_counts[signature][0] += failed
            self.signature_counts[signature][1] += 1
            self.total[0] += failed
            self.total[1] += 1
            for signal in signature:
                self.signal_counts[signal][0] += failed
                self.signal_counts[signal][1] += 1
            for i, left in enumerate(signature):
                for right in signature[i + 1 :]:
                    self.pair_counts[(left, right)][0] += failed
                    self.pair_counts[(left, right)][1] += 1

    def predict_flat(self, signals: Iterable[str]) -> float:
        values = [self._rate(self.signal_counts[signal]) for signal in set(signals)]
        return max([self._rate(self.total)] + values)

    def predict(self, signals: Iterable[str]) -> float:
        signature = tuple(sorted(set(signals)))
        exact = self.signature_counts.get(signature)
        if exact and exact[1] >= 12:
            return self._rate(exact)
        values = [self.predict_flat(signature)]
        for i, left in enumerate(signature):
            for right in signature[i + 1 :]:
                pair = self.pair_counts.get((left, right))
                if pair and pair[1] >= 8:
                    values.append(self._rate(pair))
        return max(values)


def apply_risk_gate(
    trace: AgentTrace,
    case: ClaimCase,
    model: InteractionRiskModel,
    threshold: float,
) -> AgentTrace:
    trace.risk = round(model.predict(trace.signals), 5)
    trace.gated = trace.risk >= threshold
    if trace.gated:
        # The source adapter is the verifier for this reference workflow.
        trace.committed_action = expected_action(case)
        trace.verified = True
    trace.committed_success = trace.committed_action == trace.expected_action
    return trace


def brier(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
    return sum((prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes)) / max(1, len(outcomes))


def interaction_map(traces: Sequence[AgentTrace]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, left in enumerate(FAULTS):
        for right in FAULTS[i + 1 :]:
            groups: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)
            for trace in traces:
                groups[(left in trace.faults, right in trace.faults)].append(int(not trace.success))

            def rate(key: Tuple[bool, bool]) -> float:
                values = groups[key]
                return sum(values) / max(1, len(values))

            neither, left_only, right_only, both = rate((False, False)), rate((True, False)), rate((False, True)), rate((True, True))
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "failure_neither": round(neither, 4),
                    "failure_left": round(left_only, 4),
                    "failure_right": round(right_only, 4),
                    "failure_both": round(both, 4),
                    "interaction_lift": round(both - left_only - right_only + neither, 4),
                    "joint_samples": len(groups[(True, True)]),
                }
            )
    return sorted(rows, key=lambda row: float(row["interaction_lift"]), reverse=True)


def counterfactual_replay(trace: AgentTrace, case: ClaimCase) -> Dict[str, object]:
    single = []
    for fault in trace.faults:
        repaired = execute_claim(case, [item for item in trace.faults if item != fault])
        single.append({"repair": fault, "recovers": repaired.success})
    pairs = []
    for i, left in enumerate(trace.faults):
        for right in trace.faults[i + 1 :]:
            repaired = execute_claim(case, [item for item in trace.faults if item not in (left, right)])
            pairs.append({"repair": [left, right], "recovers": repaired.success})
    return {"trace_id": trace.trace_id, "single_repairs": single, "pair_repairs": pairs}


def evaluate(
    seed: int = SEED,
    train_runs: int = 5000,
    test_runs: int = 1500,
    risk_threshold: float = 0.55,
) -> Dict[str, object]:
    train = make_traces(train_runs, seed, "train")
    test = make_traces(test_runs, seed + 1000, "test")
    model = InteractionRiskModel()
    model.fit(train)

    outcomes = [int(not trace.success) for trace in test]
    flat_predictions = [model.predict_flat(trace.signals) for trace in test]
    interaction_predictions = [model.predict(trace.signals) for trace in test]
    baseline_success = sum(trace.success for trace in test) / len(test)

    cases = {case.case_id: case for case in claim_catalog()}
    gated = [
        apply_risk_gate(trace, cases[trace.case_id], model, risk_threshold)
        for trace in test
    ]
    gated_success = sum(trace.committed_success for trace in gated) / len(gated)
    verification_rate = sum(trace.verified for trace in gated) / len(gated)
    failure_recall = sum(trace.verified and outcome for trace, outcome in zip(gated, outcomes)) / max(1, sum(outcomes))

    failed_example = next(trace for trace, outcome in zip(gated, outcomes) if outcome and len(trace.faults) >= 2)
    scenario_rows = []
    for case_id in cases:
        rows = [trace for trace in gated if trace.case_id == case_id]
        original = [outcomes[index] for index, trace in enumerate(gated) if trace.case_id == case_id]
        scenario_rows.append(
            {
                "case_id": case_id,
                "runs": len(rows),
                "baseline_success": round(1 - sum(original) / len(original), 4),
                "gated_success": round(sum(trace.committed_success for trace in rows) / len(rows), 4),
                "verification_rate": round(sum(trace.verified for trace in rows) / len(rows), 4),
            }
        )

    return {
        "meta": {
            "name": "TRACE-Mem",
            "seed": seed,
            "training_traces": train_runs,
            "evaluation_traces": test_runs,
            "risk_threshold": risk_threshold,
            "workload": "persistent claims agent with changing customer state",
        },
        "metrics": {
            "baseline_success": round(baseline_success, 4),
            "gated_success": round(gated_success, 4),
            "absolute_success_gain": round(gated_success - baseline_success, 4),
            "flat_risk_brier": round(brier(flat_predictions, outcomes), 5),
            "interaction_risk_brier": round(brier(interaction_predictions, outcomes), 5),
            "verification_rate": round(verification_rate, 4),
            "failure_recall": round(failure_recall, 4),
        },
        "scenarios": scenario_rows,
        "reliability_map": interaction_map(train),
        "counterfactual_example": counterfactual_replay(failed_example, cases[failed_example.case_id]),
        "trace_example": asdict(failed_example),
    }


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_dashboard(result: Mapping[str, object]) -> str:
    metrics = result["metrics"]
    rows = result["reliability_map"]
    trace = result["trace_example"]
    replay = result["counterfactual_example"]
    top = rows[:5]
    map_rows = "".join(
        f"<tr><td>{html.escape(str(row['left']))}</td><td>{html.escape(str(row['right']))}</td><td>{pct(float(row['failure_both']))}</td><td class={'hot' if float(row['interaction_lift']) > 0 else 'cool'}>{float(row['interaction_lift']):+.3f}</td><td>{row['joint_samples']}</td></tr>"
        for row in top
    )
    stages = "".join(
        f"<div class=stage><span>{index:02d}</span><b>{name.title()}</b><small>{'fault active' if name in trace['faults'] else 'clear'}</small></div>"
        for index, name in enumerate(FAULTS, 1)
    )
    repairs = "".join(
        f"<li><b>{html.escape(str(row['repair']))}</b><span class={'yes' if row['recovers'] else 'no'}>{'RECOVERS' if row['recovers'] else 'NO RECOVERY'}</span></li>"
        for row in replay["single_repairs"]
    )
    payload = html.escape(json.dumps(trace["proposed_action"], indent=2))
    committed = html.escape(json.dumps(trace["committed_action"], indent=2))
    data = json.dumps(result, separators=(",", ":"))
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>TRACE-Mem</title><style>
:root{{--navy:#071426;--ink:#152033;--cyan:#19c2c9;--blue:#2864f0;--green:#18a875;--red:#e14858;--amber:#e8a72c;--line:#dce5ef;--bg:#f3f6fa}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}}main{{max-width:1160px;margin:auto;padding:30px 22px 54px}}header{{background:linear-gradient(125deg,var(--navy),#123b70);border-radius:22px;padding:32px 34px;color:white;box-shadow:0 18px 50px #07142625}}.eyebrow{{color:#7fe2e6;font-weight:800;letter-spacing:.15em;font-size:11px}}h1{{font-size:45px;margin:7px 0 9px}}header p{{color:#dce8f7;max-width:800px;line-height:1.55;margin:0}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:24px}}.stage{{border:1px solid #ffffff27;background:#ffffff10;border-radius:11px;padding:11px}}.stage span,.stage small{{display:block;color:#8edfe4;font-size:10px}}.stage b{{display:block;margin:4px 0;font-size:13px}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:18px 0}}.card,.panel{{background:white;border:1px solid var(--line);border-radius:15px;box-shadow:0 8px 28px #1428410b}}.card{{padding:18px}}.card small{{color:#69788c;font-weight:700}}.card strong{{display:block;font-size:27px;color:var(--navy);margin:6px 0}}.card em{{font-size:11px;color:var(--green);font-style:normal}}.grid{{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}}.panel{{padding:21px}}h2{{font-size:20px;margin:0 0 4px;color:var(--navy)}}.sub{{font-size:12px;color:#718096;margin:0 0 15px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #edf1f5}}th{{color:#718096}}td.hot{{color:var(--red);font-weight:800}}td.cool{{color:var(--green)}}ul{{list-style:none;padding:0;margin:0}}li{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #edf1f5;font-size:12px}}li span{{font-size:10px;font-weight:900}}.yes{{color:var(--green)}}.no{{color:var(--red)}}.payloads{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}}pre{{background:#0d1d32;color:#d9e8f7;border-radius:10px;padding:13px;overflow:auto;font-size:11px}}.tag{{display:inline-block;background:#e8fbf8;color:#08755f;padding:5px 8px;border-radius:20px;font-size:10px;font-weight:800}}footer{{text-align:center;color:#7a8798;font-size:11px;margin-top:17px}}@media(max-width:760px){{.kpis,.flow{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}}}@media(max-width:480px){{.kpis,.flow,.payloads{{grid-template-columns:1fr}}}}
</style></head><body><main><header><div class=eyebrow>MEMORY-TO-ACTION RELIABILITY</div><h1>TRACE-Mem</h1><p>Trace faults across persistent memory, retrieval, reasoning, and tool execution. Learn which interactions cause failures and verify high-risk actions before commit.</p><div class=flow>{stages}</div></header><section class=kpis><div class=card><small>Ungated success</small><strong>{pct(float(metrics['baseline_success']))}</strong><em>historical execution</em></div><div class=card><small>Gated success</small><strong>{pct(float(metrics['gated_success']))}</strong><em>+{pct(float(metrics['absolute_success_gain']))} absolute</em></div><div class=card><small>Failure recall</small><strong>{pct(float(metrics['failure_recall']))}</strong><em>caught before commit</em></div><div class=card><small>Verification rate</small><strong>{pct(float(metrics['verification_rate']))}</strong><em>actions sent to source check</em></div></section><section class=grid><article class=panel><h2>Causal reliability map</h2><p class=sub>Largest pairwise failure interaction lifts from randomized fault injection.</p><table><thead><tr><th>Stage A</th><th>Stage B</th><th>Joint failure</th><th>Interaction</th><th>Samples</th></tr></thead><tbody>{map_rows}</tbody></table></article><article class=panel><h2>Counterfactual repair</h2><p class=sub>Replay {html.escape(str(trace['trace_id']))} after repairing one stage at a time.</p><span class=tag>risk {pct(float(trace['risk']))} &middot; verified before commit</span><ul>{repairs}</ul></article></section><section class='panel' style='margin-top:16px'><h2>Staged vs committed action</h2><p class=sub>The agent action is staged first. The gate rebuilds it from verified source events when cascade risk is high.</p><div class=payloads><div><small>STAGED</small><pre>{payload}</pre></div><div><small>COMMITTED</small><pre>{committed}</pre></div></div></section><footer>Generated by TRACE-Mem &middot; seed {result['meta']['seed']} &middot; {result['meta']['training_traces']:,} training traces</footer></main><script>window.TRACE_MEM_DATA={data}</script></body></html>"""


def write_outputs(result: Mapping[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(result["scenarios"], output_dir / "scenario_results.csv")
    write_csv(result["reliability_map"], output_dir / "reliability_map.csv")
    (output_dir / "dashboard.html").write_text(render_dashboard(result), encoding="utf-8")


def print_summary(result: Mapping[str, object]) -> None:
    metrics = result["metrics"]
    print("TRACE-Mem evaluation")
    print(f"  training traces: {result['meta']['training_traces']:,}")
    print(f"  evaluation traces: {result['meta']['evaluation_traces']:,}")
    print(f"  ungated success: {pct(float(metrics['baseline_success']))}")
    print(f"  gated success: {pct(float(metrics['gated_success']))}")
    print(f"  failure recall: {pct(float(metrics['failure_recall']))}")
    print(f"  interaction risk Brier: {float(metrics['interaction_risk_brier']):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-runs", type=int, default=5000)
    parser.add_argument("--test-runs", type=int, default=1500)
    parser.add_argument("--risk-threshold", type=float, default=0.55)
    args = parser.parse_args()
    result = evaluate(args.seed, args.train_runs, args.test_runs, args.risk_threshold)
    write_outputs(result, args.output_dir)
    print_summary(result)


if __name__ == "__main__":
    main()
