# CascadeLens

CascadeLens is a small, reproducible proof-of-concept for a thesis on
dependency-aware reliability forecasting in multi-step agents.

Most agent evaluations report end-to-end success or isolated component
accuracy. CascadeLens asks a different question: given a workflow of retrieval,
reasoning, tool, verification, and action steps, can we predict how errors
propagate through its topology and decide where one extra verification check
will help most?

## What this PoC demonstrates

1. A typed workflow DAG for four insurance-like claim-review patterns.
2. Controlled, privacy-safe fault injection with correlated episode shocks.
3. A Laplace-smoothed conditional propagation model learned from traces.
4. Comparison against a naive independence/product baseline.
5. Budget-aware selection of retrieval or tool-result verification.
6. A self-contained HTML dashboard for the research walkthrough.

The environment is synthetic by design. Its results demonstrate that the
method and instrumentation work; they do **not** establish performance on real
insurance claims or production agents.

## Run

Only Python 3.10+ is required; there are no external packages or API keys.

```powershell
python .\cascadelens.py --output-dir .\results
Start-Process .\results\dashboard.html
```

Generated files:

- `dashboard.html` - offline visual demo
- `results.json` - complete reproducible output
- `scenario_results.csv` - scenario-level evaluation table

The random seed defaults to `42` and can be changed with `--seed`.

## Test

```powershell
python -m unittest discover -s .\tests -v
```

## What changes in the six-month thesis

- Replace synthetic nodes with trace adapters for an agent framework or an MCP
  gateway.
- Use deterministic end-state evaluators in public tool-agent environments
  such as tau-bench or AgentDojo, then validate on one stakeholder workflow.
- Compare the conditional DAG model with independence, calibrated sequence
  models, SAUP-style uncertainty propagation, and counterfactual replay.
- Measure calibration (Brier score/ECE), failure recall, reliability-cost
  Pareto curves, cross-model transfer, and robustness to topology shifts.
- Release trace schema, fault operators, benchmark tasks, and evaluation code.

## Suggested 90-second walkthrough

1. Open `results/dashboard.html` and state the problem in one sentence.
2. Point to the four-stage pipeline: trace, perturb, forecast, intervene.
3. Compare the blue/cyan/amber bars for branch and long-horizon workflows.
4. Show that the final failed action is not necessarily the first risky step.
5. Explain the budget table and the selected checkpoint.
6. End with the scope note: the PoC validates the research mechanism; the
   thesis validates generalization on real LLM agents and stakeholder tasks.
