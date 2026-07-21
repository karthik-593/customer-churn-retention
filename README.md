# Churn → Retention Decision System (KKBox)

Predicts subscription churn on the KKBox WSDM-2018 data and turns the scores into a contact decision:
who to send a retention offer to, ranked on expected value rather than P(churn) alone. The deliverable
is a ranked contact list plus an on-demand scoring API.

Guiding constraint: grounded over impressive. Every choice is defensible line-by-line. Full rationale
in `DECISIONS.md`, the Phase-D experiment spec in `AB_DESIGN.md`, the Phase-F monitoring runbook in
`MONITORING.md`.

## Problem framing

A churn classifier ranks who will leave. The business question is who's worth contacting. So the rule
is expected-value, not a probability cutoff:

````
contact  iff  save_rate · value · P(churn) − offer ≥ 0
````

At the locked operating point (offer NT$150, save_rate 0.30, median monthly paid NT$129 → 12-mo value NT$1,548) this is a
derived break-even of P ≥ 0.323, never a hardcoded number. `save_rate` and `value` are parameters:
Phase D measures `save_rate` for the off-auto-renew book (the contacted population) and the same
rule applies.

## Approach (phases A–F)

- **A — label & cohorts.** Churn = no genuine paid renewal within 30 days of an operative expiry.
  Validated row-level against KKBox's labels (κ 0.94, precision 0.999). Unit is (customer × expiry-month);
  split is temporal, leakage controlled by point-in-time features rather than disjoint customers (that's
  the deployment reality — the model re-scores the same base every cycle).
- **B — model.** XGBoost on 12 paid point-in-time features (`base12`). Trials (`is_free`) and the
  off-auto-renew slice handled by scope/rule, not buried in the model. `payment_method_id` and lifecycle
  tested and left out (poison / topline-neutral-but-overfits), decided on the cost backtest, not AUC.
  Isotonic-calibrated, since the cost rule reads P as money.
- **C — explainability.** SHAP confirms the drivers (off-auto-renew dominates, price is a real effect)
  and shows the contact list collapses to the off-auto-renew book.
- **D — uplift (design).** Saveability ≠ P(churn). A randomised retention experiment to measure
  `save_rate` per slice, method validated on the Hillstrom RCT. Spec in `AB_DESIGN.md`.
- **E — productionisation.** Modular `src/`, MLflow registry, one-command batch pipeline, thin FastAPI
  + Docker.
- **F — monitoring.** Two-tier drift + calibration monitoring, hand-rolled. Runbook in `MONITORING.md`.

## Results (paid scorer, temporal test set)

- Test PR-AUC ≈ **0.40** (full-population refit **0.413** — the subsample isn't flattering it).
- Calibrated; reliability tracks observed, Brier improves after isotonic.
- Cost backtest @ 12-mo horizon (val-derived break-even 0.302, realized median monthly paid NT$129): **3,513**
  contacts, precision **0.499**, net **NT$287k**; beats the 14-feature variant on net value and
  precision@budget. Deployed threshold is 0.323 (median monthly paid NT$129, value NT$1,548).
- Full-base batch run scores the whole monthly cohort (~785k paid) → ranked contact list.

*Figures regenerated on `xgboost==3.3.0` (pinned). The contact count is a hard cut at the break-even, so
it wobbles ~±1% run-to-run under multithreaded training; the stable operating metric is precision@budget,
not the integer.*

## Monitoring (Phase F)

The model scores at expiry and the label resolves 30 days later, so there's no ground truth at score
time. Monitoring is split accordingly:

- tier 0 — data-quality gate. Schema, value ranges, and structure (row-count parity, silent-share). Built
  around the coalesce in `features.sql`: a broken engagement join shows up as extra dormant customers, not
  nulls, so the gate checks structure rather than null counts.
- tier 1 — score + contact-rate drift. Label-free, at score time. Tracks contact_rate (the budget),
  mean P among contacted (band composition), and tail quantiles. No PSI — for a threshold decision only
  the mass near break-even matters.
- tier 2 — calibration on matured labels. The retrain trigger. Predicted vs realized churn; the signal is
  the contacted-band gap (test: mean_pred 0.4768 vs realized 0.4987, −0.022), not global ECE.

Honest scope: ~2 labelable months and adjacent cohorts in-distribution, so no tier has caught a real
regression — detection is proven in synthetic smoke tests. Phase F ships the discipline and a provisional
retrain rule, unit-tested, on data with no failure in it. Details in `MONITORING.md`.

## Run

````bash
make all          # features → train → export → score   (Windows: .\make all)
make serve        # on-demand API at http://127.0.0.1:8000/docs
make score CUTOFF=2017-02-01   # batch-score one cohort month → reports/contact_list.csv
make smoke        # offline smoke tests (synthetic fixtures, no real data)

python -m monitoring.data_quality 2017-01-01   # tier 0 — data-quality gate
python -m monitoring.score_drift               # tier 1 — score / contact-rate drift
python -m monitoring.calibration               # tier 2 — predicted vs realized
````

Prereqs: `pip install -r requirements.txt`, and the Phase-A tables in `data/churn.duckdb`
(`pred_points`, `cohorts_s`, plus the `transactions` / `user_logs` CSV-backed views).

## Layout

````
sql/features.sql      point-in-time feature SQL (one path for train & score → no skew)
src/
  config.py           feature list + cost constants (single source of truth)
  db.py               duckdb connection
  data.py             scoring points for a cutoff month (from validated pred_points)
  features.py         build the feature table from any point set
  train.py            fit base12 + calibrate, log/register churn-base12@prod (MLflow)
  churn_pyfunc.py     bundled artifact: model + calibrator + feature order → calibrated P
  score.py            evaluate() (reconcile backtest) + score() (batch contact list)
  export_model.py     materialise prod model to api/model/ (serving needs no registry)
api/main.py           thin FastAPI: feature-vector-in → P + EV decision
monitoring/
  data_quality.py     tier 0 — schema / structure / value gate
  score_drift.py      tier 1 — contact-rate + composition + tail drift
  calibration.py      tier 2 — predicted vs realized (the spine)
tests/                offline smoke tests (synthetic fixtures, hand-computed expected values)
Dockerfile  Makefile  make.bat
notebooks/            01 eda/label · 02 model · 03 shap · 04 uplift
DECISIONS.md  AB_DESIGN.md  MONITORING.md
````

## Stack

DuckDB (SQL feature engineering) · pandas · scikit-learn · XGBoost · SHAP · MLflow (tracking + registry,
local sqlite backend) · FastAPI · Docker. Phase-D uplift and Phase-F monitoring are hand-rolled on
numpy/pandas — no Evidently, no causal-uplift library.

## Design notes

- One feature path. `features.sql` builds training and scoring rows identically (train/serve parity);
  features never carry the label.
- Calibrated bundle. The registered model emits calibrated P directly, so the calibrator can't be
  forgotten downstream and the cost threshold is always derived from config.
- Restraint, on purpose. Make over a flow engine; registry without stage-promotion ceremony; an on-demand
  API as a demo while the batch list is the product; monitoring as a generated report, not a dashboard
  stack. Nothing here that doesn't earn its place.
````
````
