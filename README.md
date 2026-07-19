# TRACE-Mem

TRACE-Mem is an executable reliability layer for persistent AI agents. It
maps how faults travel from memory writes to irreversible tool actions, finds
the stages that caused a failed run, and verifies risky actions before they
are committed.

The included claims workflow is a complete reference implementation. A case
evolves across sessions: an address changes, coverage is updated, and the
preferred contact channel changes. TRACE-Mem records those facts in a
bi-temporal memory, injects controlled faults at five stages, replays failures
with targeted repairs, learns an interaction-aware risk model, and applies a
verification gate before the final action.

## What is implemented

- Append-only, bi-temporal memory with `active`, `superseded`, and `disputed`
  states.
- Fault injection at memory write, consolidation, retrieval, reasoning, and
  tool-action stages.
- Trace-level signals for write gaps, active-memory conflicts, stale
  retrievals, inconsistent reasoning, and mutated tool payloads.
- Counterfactual replay that repairs one stage at a time and records whether
  the failed action recovers.
- A causal reliability map that measures pairwise fault interaction lift.
- An interaction-aware risk model learned from historical traces.
- A pre-commit risk gate that verifies source events and rebuilds the proposed
  action when risk exceeds a configurable threshold.
- An offline HTML operations dashboard plus JSON and CSV exports.

## Run

TRACE-Mem uses only the Python standard library. Python 3.10 or newer is
required; no API keys or external services are needed.

```powershell
python .\trace_mem.py --output-dir .\results
Start-Process .\results\dashboard.html
```

Useful options:

```powershell
python .\trace_mem.py --seed 7 --train-runs 5000 --test-runs 1500 --risk-threshold 0.55
```

Generated artifacts:

- `dashboard.html`: reliability, intervention, and trace explorer
- `results.json`: complete machine-readable run output
- `scenario_results.csv`: per-case operating metrics
- `reliability_map.csv`: pairwise causal interaction map

## Test

```powershell
python -m unittest discover -s .\tests -v
```

## Runtime flow

1. The agent stages a tool action instead of immediately committing it.
2. TRACE-Mem derives reliability signals from the complete memory-to-action
   trace.
3. The learned model estimates failure risk, including interactions between
   signals.
4. Low-risk actions commit normally. High-risk actions are verified against
   the source-event adapter, rebuilt, and then committed.
5. Failed historical traces can be replayed with targeted repairs to produce
   causal attributions and improve verification policy.

The built-in claims cases contain generated data and serve as a reproducible
local workload. The runtime classes are domain-independent: another agent can
provide its own source events, staged action payload, evaluator, and verifier
without changing the reliability model.
