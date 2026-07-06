# MONITORING.md — Phase F

The model scores at membership expiry; the label resolves 30 days later. No ground truth at score
time, so monitoring is in two parts.

- leading (no labels, every run): score + contact-rate drift, plus the data-quality gate. says the
  inputs moved. doesn't tell you the model decayed — that needs labels.
- lagging (labels in, 30 days on): predicted vs realized churn. this is what tells you the EV math
  still holds, and where the retrain trigger sits.

All hand-rolled on numpy/pandas, smoke-tested on synthetic fixtures, checked against real cohort
output. No Evidently — it'd only wrap tier 3 (per-feature attribution, the least useful tier), and
its 0.1/0.25 bands aren't tied to this model. Same reasoning as the hand-rolled uplift (DECISIONS §7).

## Scope: demonstrated, not validated

No tier has caught a real regression — the data can't produce one. ~2 labelable months, adjacent
cohorts in-distribution by construction. Detection is only proven in the smoke tests, on injected
faults (under-prediction, tail-thinning, composition shift, corrupted columns). A real cohort
passing means the pipeline ran; it doesn't mean detection works. So Phase F is the discipline and
the trigger, unit-tested, on data with no failure in it. Tier 2 catching a seasonal base-rate swing
(AB_DESIGN §11) is the case that matters, and it hasn't had cause to fire.

## Hierarchy

Run in this order:

0. data-quality / schema gate — is the table trustworthy
1. score + contact-rate drift — leading, no labels
2. calibration on matured labels — the retrain trigger
3. per-feature drift (PSI) — attribution, only if 1 or 2 fires

Not feature-drift first. The failure mode here is label shift — base rate moves with
season/promo/price (§11). It breaks the frozen calibrator but can leave covariates flat, so a
feature-drift check misses it. base12 is auto-renew-dominated anyway, so the covariate moves that
matter already show up in the score distribution. Per-feature PSI is only useful to explain a move
tier 1 or 2 already caught. Tier 3 isn't built — nothing to attribute yet.

## Tier 0 — data-quality gate (`monitoring/data_quality.py`)

Runs before any statistic. Most "model broke" cases are pipeline bugs — a join change, a CSV-view
path that didn't resolve, a units flip — not drift, and worse when they happen.

features.sql coalesces missing engagement to defaults (has_activity_60d=0, recency=60). So a broken
engagement join doesn't show up as nulls; the dropped rows come through as dormant customers and a
null check passes clean. The gate checks structure and value ranges instead.

hard-fail (non-zero exit, stops the pipeline):
- schema: 12 base12 cols + is_free present
- numeric dtype (loose — int32 vs int64 not distinguished, the API casts to signature anyway)
- no nulls in base12 (coalesce should leave none)
- binaries ∈ {0,1}; counts/amounts ≥ 0; completion_ratio ∈ [0,1]; recency ∈ [0,60]
- row parity: rows_out == distinct points_in (commitment join dropped or duplicated rows)

warn (logged, doesn't stop):
- silent_share vs ~0.213 baseline, ceiling 2× — a broken engagement join pushes this toward 1.0;
  small moves are normal cohort variation
- discount < 0 share (paid > list, a real promo/tax edge), ceiling 0.05

Catches impossible and structural faults. A units flip that stays in range but wrong-magnitude won't
trip it — that one moves the score distribution and is tier 1.

verified, cutoff 2017-01 full base: 778,107 rows == 778,107 points, silent_share 0.217 vs 0.213, all
pass. silent_share staying within 0.4pp on a different cohort and denominator is why it works as a
join-health check.

run: `python -m monitoring.data_quality 2017-01-01`

## Tier 1 — score + contact-rate drift (`monitoring/score_drift.py`)

No labels, runs at score time. The model already weighted the 12 inputs, so a covariate move that
matters lands in the score distribution — which is why this comes before per-feature drift.

Three numbers, all tied to the decision. No PSI/KS on the whole distribution: it's a threshold
decision, only the mass at and above break-even changes anything, and sub-threshold movement is free
(steps 1-2 both showed this). PSI was built and dropped — flat on both cohorts, and the bands are
arbitrary.

- contact_rate: mass ≥ break-even, i.e. the budget. the number that triggers. a drop means
  under-contacting (§3) before labels are in.
- mean_P\|contacted: who you'd contact, on average. moves when the mix inside the band changes even
  if the count doesn't. it's the leading read on tier 2's contacted-band gap — same quantity,
  predicted now and realized later.
- tail quantiles q90/95/99/max: shape. contacts come from q95 up (break-even 0.323 sits between
  q95 ≈ 0.24 and q99 ≈ 0.56).

reference = test split (PR-AUC and calibration were checked there). refresh below.

| cohort | n_paid | contact_rate | mean_P\|contacted |
|---|---|---|---|
| test (ref) | 98,267 | 3.57% | 0.4768 |
| 2017-03 (first unlabelled) | 25,231 | 3.60% (+0.03pp) | 0.4674 (−0.009) |
| 2017-01 (full base) | 770,181 | 4.06% (+0.49pp) | 0.4808 (+0.004) |

Both stable. Lines up with the §6 lock — the calendar-clock features are out of the scorer, so it
carries little month-to-month drift. (Consistent with, not proven by; I didn't re-run the keep-them
version.) January's +0.49pp is in-sample, so I'm noting it, not reading it — only a matured held-out
cohort and tier 2 can say if it's real.

cross-check: mean_P\|contacted on test is 0.4768, same as tier 2's contacted-band mean_pred. the two
modules score the model the same way.

run: `python -m monitoring.score_drift` (latest cutoff) · `python -m monitoring.score_drift 2017-01-01`

## Tier 2 — calibration on matured labels (`monitoring/calibration.py`)

Predicted calibrated P vs realized churn on a labelled cohort. Lagging — needs the 30-day window
closed — and the one tier that tells you the EV math is honest.

Reports overall mean_pred vs base_rate, per-decile reliability (same table as train.py cell 46), and
the contacted band (P ≥ break-even).

verified, test split: n=98,267 paid, base_rate 0.0465, mean_pred 0.0393, gap −0.0072 (under-predicts).
top bin 0.306 pred / 0.304 actual. matches DECISIONS §3.

What sets the trigger: global ECE is 0.0080, and it's the wrong number to watch. ~86% of it comes
from the bottom two bins — ~46k customers predicted near 0, churn ~1.5%, never contacted, no cost.
The contacted band is where it matters: mean_pred 0.4768, realized 0.4987, gap −0.022, about 3× the
global figure. The model under-states risk on the customers it tells you to contact — the §3
under-contacting risk, for real this time. Ranking still holds (precision 0.50, top decile
near-exact), so it's a calibration issue, not a model that stopped working.

→ watch the contacted-band gap and realized precision@budget. global ECE is context.
→ sign: gap = mean_pred − realized; negative = under-prediction.

run: `python -m monitoring.calibration` (test) · `python -m monitoring.calibration val` (calibrator's
fit set, sanity)

## Reference window + refresh

Tier 1 needs a reference; the code uses test. In production a fixed reference is wrong both ways —
leave it and everything eventually looks drifted; re-point it at "recent" every run and slow drift
gets absorbed.

policy: reference = a trailing window of recent cohorts that have (a) matured their labels and
(b) passed the tier-2 check. it moves on validation, not on a schedule — a cohort becomes the
reference only once labels confirm it's good. a few cohorts wide so one seasonal extreme can't set it
(§11). window length gets fixed once there are real cohorts, same as the alert bands.

This ties tier 1's reference to tier 2's verdict on purpose — tier 1 is the leading proxy for tier 2,
so the question it should answer is "have we moved away from the last cohort tier 2 cleared?". The
cost: the tiers aren't independent, so a tier-2 miss also corrupts tier 1's baseline. Fine, because
tier 2 is the ground-truth anchor — if it's wrong, everything downstream is.

Here no new data arrives, so the refresh never runs and test stays the reference. The hardcoded test
is that case.

## Retrain trigger (provisional)

On tier 2, not on a drift flag:
- contacted-band gap past ~2-3× the test reference (≈ −0.05), or
- realized precision@budget under ~0.45 (a floor below the 0.50 operating point)

Re-estimate both from the first matured production cohorts. One cohort is noise, and churn swings
month to month (§11), so a tight band fires on normal movement. Tier 1 is the early warning to look
sooner; it doesn't trigger a retrain on its own.

Model's locked and no new data flows here, so this is a written rule, not a running job.

## Not built

- live dashboard / Grafana / Prometheus — nightly batch, a generated report is enough
- alerting infra — no on-call for a portfolio batch job
- auto-retrain — model locked, no new data; retraining nothing is theatre
- Great Expectations — tier 0 is ~12 checks, a framework is overkill
- Evidently — would only wrap tier 3; if it's ever added there, pin it (API moved a lot across
  0.4 → 0.6 → 0.7)
- tier 3 per-feature PSI — nothing to attribute until tier 1 or 2 fires

## Files

```
monitoring/
  data_quality.py    tier 0 — schema / structure / value gate
  score_drift.py     tier 1 — contact-rate + composition + tail drift
  calibration.py     tier 2 — predicted vs realized
tests/
  smoke_data_quality.py · smoke_score_drift.py · smoke_calibration.py
```

Every pure metric function is separate from the db/mlflow I/O, so it's unit-tested on synthetic
fixtures with hand-computed values. The wrappers do the db and model calls.