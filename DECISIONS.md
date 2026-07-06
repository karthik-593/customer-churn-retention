# DECISIONS

## 1. What counts as churn
Churned at a membership expiry = no genuine paid renewal within 30 days after it. A hard yes/no at a
fixed date, built from the transactions log. Picked over a fuzzy rule ("after several notifications")
because that isn't cleanly measurable.

Three rules make the label correct:
- **Latest-transaction-wins expiry.** Same-day transactions collapse to the day's max expiry; the
  governing expiry at any point is the latest transaction's. Absorbs cancellations that pull the
  expiry earlier, and same-day cancel-then-rebuy.
- **A renewal must be genuine and paid** (`is_cancel = 0` AND `actual_amount_paid > 0`). A
  cancellation or a zero-amount entry isn't a renewal. `is_cancel` stays a feature, never the label.
- **Score only operative expiries** — the expiry the customer actually reached as their current
  membership end. Drop an expiry if (a) a paid renewal dated before it already extends past it
  (already paid through), or (b) any transaction lands between its governing transaction and the
  expiry, moving it (a cancel pulling back, an early renewal pushing forward). Without this,
  overlapping/long plans and cancel-rebuy sequences spawn phantom expiries that aren't real renewal
  decisions.

The data forced these rules, they weren't assumed. Validated row-level against KKBox's official
labels — train.csv, the Feb-2017 cohort, the only one inside our labelable window (train_v2 is the
April cohort, whose 30-day outcome runs past our 2017-03-31 data end). A naive label (any later
transaction is a renewal, every expiry scored) agreed at kappa 0.82 / recall 0.71. The misses: 97%
credited a renewal dated *before* the expiry, i.e. non-operative points, and those users carried ~2
expiry points in the quarter. Adding genuine-paid + operative took it to kappa 0.94, precision 0.999,
recall 0.89 (n=788k Feb cohort), and dropped ~16% of churn labels that were phantom cancel-rebuy
duplicates (per-cycle base rate 13% → ~11%). These two rules are what KKBox's own labeller does
(dataset spec examples 2 and 3: a cancellation moves the expiry; a later transaction extending the
expiry out of the month removes the user from the cohort). Residual disagreement (~4k of ~788k) is
rare cancel-rebuy edge cases their sequential labeller resolves, plus genuine 30-day-boundary
differences — left documented. Our per-cycle label is a different formulation from KKBox's
one-label-per-user; the Feb cohort is a sanity check, and precision 0.999 means our churn calls are
basically never wrong.

## 2. Prediction window
Predict AT the expiry date. Everything before expiry is the feature information. The label is the 30
days after: a renewal in that window → stayed, none → churned. So we can only label customers whose
30-day window has already passed. Training uses only customers old enough that the outcome is known.

## 3. Cost model
Contact only when expected gain beats the offer:

    P(churn) × save_rate × value > offer_cost

Cutoff = the break-even probability this implies, `offer / (save × value)`, not 0.5.

Terms (fixed where assumed, derived where possible):
- `offer_cost = ₹150` — the retention offer.
- `save_rate = 0.30` — assumed lift from contact. The softest number; Phase D measures it (uplift).
- `value = ARPU × horizon`, ARPU = ₹129/mo, derived from data (median of
  `actual_amount_paid / payment_plan_days × 30` over paid test rows — lands on the standard monthly
  plan price).

`value` is swept over a 6–24 month horizon rather than pinned, because the right horizon is a
business call and the rule should hold across it. Gross net is positive at every horizon
(6mo ≈ ₹24k → 24mo ≈ ₹1.22M). A pessimistic net that also subtracts the discount from saved revenue
stays positive too (6mo +₹3.2k, the thinnest point). The conclusion doesn't hinge on the horizon.

Operating point we quote: 12 months → value ₹1,548 → break-even ≈ 0.323. So we contact paid customers
above ~0.32 calibrated P(churn) — not 0.5, and not the earlier ~0.63 (a flat ₹800 placeholder, now
retired). At this point the rule contacts ~3,491 paid customers at 0.50 precision, net ≈ ₹287k over
12 months. The count is a hard cut at the break-even, so it wobbles ~±1% run-to-run under
multithreaded training; plan on precision@budget, not the integer.

Sensitivity: hold the horizon at 12 months, sweep save_rate 0.15–0.40 → net-positive even at 0.15.
Robust to the softest assumption. Until Phase D measures save_rate, 0.30 is the planning anchor and
the sign doesn't flip across the range.

Probabilities must be calibrated before entering this formula (PROJECT_RULES.md rule #7) — the threshold
reads P(churn) as money. Isotonic-on-val keeps the ranking near-identical (PR-AUC 0.400 → 0.388) and
lines up the reliability curve (top decile predicted 0.305 vs observed 0.303). Watch the mild
test-side under-prediction: cal mean 0.039 vs actual 0.047 — Phase F tier 2 monitors exactly this.

## 4. Unit of analysis & train/val/test split
Training rows are monthly expiry cohorts: for each calendar month, one row per customer whose
membership expires that month. Features are point-in-time as of that expiry (§2); the label is the
next-30-days outcome (§1).

A customer recurs across cohorts — one row per renewal cycle — because that's how a production churn
model works, re-scoring the same base every cycle. The unit is a (customer, expiry-month) prediction
point, not a customer.

The split is temporal by expiry month: earlier months → train, a middle block → val, the latest
labelable months → test. A customer can appear in more than one split.

Why this is leakage-safe without keeping each customer on one side. Leakage is prevented by strict
point-in-time features (every feature uses only data on or before that row's cutoff), not by disjoint
customers. A customer's January row and March row are independent observations made with different
information, so the same `msno` in both train and test isn't future leakage — it's the deployment
reality. One-line defence: *point-in-time features, not disjoint splits, are the correct leakage guard
for a model that re-scores the same base.*

The trade-off, stated. We accept a small risk that the model picks up stable per-customer
idiosyncrasies, in exchange for (a) a design that matches deployment, and (b) avoiding the bias of a
one-row-per-customer design, where the latest cohorts would skew toward still-active users (a churned
user's last expiry is old, so recent test rows would under-represent churn). Checked, not asserted:
test PR-AUC on customers seen in train 0.325 (base 0.036) vs unseen 0.539 (base 0.094) — the model is
no *better* on seen customers, so recurrence isn't inflating the headline. The gap is the
new-vs-tenured base-rate difference.

Rule considered and rejected: "each customer on one side of the split." That answers "how well do I
predict for customers I've never seen?" — the cold-start / fraud question. The churn question is "how
well do I predict for customers I keep re-scoring?", so recurring customers in the test set is the
faithful evaluation. (This revises the earlier non-negotiable rule #5; PROJECT_RULES.md is updated to match.)

## 5. Model scope — paid only; free trials handled by rule
Trained and evaluated on paid prediction points only (`is_free = 0`). Free trials
(`actual_amount_paid = 0`) aren't scored by the model — rule: `is_free = 1` → ~93% churn → route to a
conversion flow, not a retention offer.

Why scope it out instead of one model for both:
- The cost model (§3) assumes a paying customer with revenue to protect. A trial has no ARPU to
  retain, so the EV formula is meaningless for it. "Convert, don't retain" is the right intervention.
- Trials are trivially separable (one `is_free` split cleaves the ~93%-churn block), so a rule
  captures them as well as the model would, and keeps the headline honest: with trials in, topline
  test PR-AUC ≈ 0.59 is mostly an `is_free` detector; on paid only it's ≈ 0.40 (test 0.400) — the
  deployable number.
- Scoping costs no accuracy: the paid-refit model (test 0.400) matches the all-population model scored
  on paid rows (≈0.400). The `is_free` split was free to the tree, not capacity-stealing — so this is
  a call about honesty and economics, not a performance trade-off.

`is_auto_renew` is the analogous dominant binary cleave *within* the paid model (the risky slice — the
whole Phase-C contact list is this off-auto-renew book, §7). It's deliberately NOT scoped out the way
trials are: an `auto_renew = 0` customer is paying, high-risk, and retainable, i.e. the target. A
trial is none of those.

## 6. Model lock — base12, and the features dropped from the scorer
The shipped scorer is **base12**: 12 paid, point-in-time features. Two families were tested and left
out.

`payment_method_id` — dropped, poison. High-cardinality categorical the tree memorises, and its
category mix shifts across the temporal split, so it overfits and degrades the honest test number.

lifecycle (`tenure_days`, `n_prior_cycles`) — built, kept out of the scorer. On a paired bootstrap
it's topline-neutral (PR-AUC −0.0044 [−0.0133, +0.0040], within noise) while sharpening the look-safe
(`auto_renew = 1`) slice by +0.041 [+0.029, +0.053], roughly doubling it (0.038 → 0.079). The
14-feature variant overfits the topline (train 0.66 / test 0.40 — tenure acts as a calendar clock
under the temporal split); base12's train–test gap is ~0, test even ≥ train. The call was made in
rupees, not PR-AUC: topline is a wash, 14-feat overfits, and under the global cost threshold base12
wins the backtest (net ₹287k vs ₹257k at 12mo; precision tied ~0.50; prec@1k 0.70 vs 0.68, prec@3k
0.52 vs 0.51). The safe-slice gain is real, but a single global threshold never reaches that low-P
segment — it only pays off with segment-specific save rates. So lifecycle goes to the Phase-D segment
layer, not the scorer.

This overturned the starting hypothesis that lifecycle would help. The result is the discovery:
lifecycle is topline-neutral, the 14-feat overfits via the tenure clock, and it sharpens the saveable
segment — but the deployed global rule can't use that.

## 7. Saveability ≠ P(churn) — why Phase D is an uplift experiment, not more scoring
Phase C's contact list (calibrated P ≥ break-even) collapses to one regime: it's entirely the
off-auto-renew book. On-auto-renew customers get a negative SHAP push and never clear the threshold.
Within the book, new / first-cycle customers are a high-churn slice (943 of ~3,491, ~27%), not the
bulk. The dominant SHAP drivers are `is_auto_renew` and price, and the price effect is real, not a
collinear artifact (raw churn rises 0.016 → 0.173 across price quartiles). This corrects an earlier
(14-feat) read that called the list "majority new."

The core point: a churn score ranks who will leave, not who an offer can move. Saveability is causal —
the lift in retention from contacting — and the contact list / SHAP give lever *hypotheses* only. So
Phase D doesn't build a bigger classifier; it builds an uplift model on data where the treatment was
randomised. On Hillstrom (a real email RCT) a T-learner → held-out Qini validated the method end to
end: Qini coef 42, the top 30% by predicted uplift captures ~24% more incremental response than
random, a persuadable head is real, there are no true sleeping-dogs, and the scores rank but don't
size the effect. The uplift *tree* was dropped — a direct-method contrast that wouldn't change the
call here.

What transfers to KKBox is the method and discipline (uplift ≠ propensity; a control holdout is
non-negotiable; trust coarse tiers, not per-customer scores), not Hillstrom's numbers. The experiment
that would measure KKBox's own per-slice save rates is specified in **AB_DESIGN.md**. That measured
save_rate then replaces the 0.30 assumption in §3.

## 8. Productionisation (Phase E)
The notebooks are the lab; `src/` is the shipped pipeline. Each call chosen for defensibility.

- **One feature path.** `sql/features.sql` builds training and scoring rows identically, reading from
  a swappable points view (`_feat_points`); features never carry the label. The batch scorer selects a
  cutoff month's points from the validated phase-A `pred_points` table (point construction stays in
  notebook 01, kappa 0.94, not re-derived) and runs the same SQL. One feature code path, no skew.
- **Calibrated bundle in the registry.** Logged as a single MLflow pyfunc bundling {xgboost, isotonic
  calibrator, feature order}, emitting calibrated P directly — the calibrator can't be forgotten
  downstream and the feature order is pinned to the model. Registered by name + alias
  (`churn-base12@prod`), loaded by alias, never a run hash. Local sqlite backend. The registry is thin
  (register + load-by-alias, no stage-promotion workflow) because there's one model and one consumer.
  Tracking logs params/metrics so the base12-vs-14feat selection story is auditable.
- **Threshold derived, never hardcoded.** Cost params (offer ₹150, save_rate 0.30, ARPU ₹129, horizon)
  live in `src/config.py`; break-even and per-row EV are computed from them. `save_rate` and `value`
  are parameters, so Phase-D segment-specific save rates plug into the same rule without a rewrite.
  0.323 is an output, not a constant in the code.
- **Value rule in code.** The scorer applies `contact iff save*value*P - offer >= 0`, ranks by EV, and
  routes `is_free = 1` to the conversion flow (never scored). `evaluate()` reconciles the cost backtest
  against notebook cell 38 (the sampled test set); `score()` produces the full-base ranked contact list
  for a cutoff month (~785k paid/month → ~3.4% contacted). Different denominators, same operating point.
- **Serving has no registry dependency.** `export_model.py` materialises the prod model to
  `api/model/`; the API loads it by path (registry is only a local-dev fallback). The container is
  self-contained and doesn't carry the tracking DB.
- **Thin API.** FastAPI, feature-vector-in → calibrated P + EV decision, same cost math as the batch
  job (imported from `config`, so they can't disagree). This is the on-demand single-customer path
  (demo / CSM lookup); the batch contact list is the product. Churn is slow-moving, so a real-time
  endpoint is a capability demo, not a need. The API casts inputs to the model's own signature dtypes
  before predict — a training-sample int32/int64 artifact would otherwise surface as a 500.
- **Orchestration: a Makefile** (`features → train → export → score`, plus `serve`/`smoke`), with a
  `make.bat` shim for Windows. A 3-step DAG doesn't warrant a flow engine.
- **Reproducibility.** `xgboost==3.3.0` pinned. The contact count is a hard cut at the break-even and
  wobbles ~±1% run-to-run under multithreaded XGBoost (early-stopping tree count shifts → calibrator
  shifts → a few hundred borderline customers cross the line). That's why the notebook's ~3,491 and a
  fresh `src` run's ~3,526 differ — same model, a threshold knife-edge. Plan on precision@budget.

## 9. Monitoring (Phase F)
The model scores at expiry; the label resolves 30 days later. No ground truth at score time, so
monitoring is two-tier: leading signals with no labels (score + contact-rate drift, a data-quality
gate) and a lagging calibration check once labels mature. The retrain trigger sits on the lagging
tier, not on a drift flag — a drift signal says the inputs moved, not that the model decayed.

The calls that matter:
- **Outcome-first hierarchy:** data-quality gate → score/contact drift → calibration on matured labels
  → per-feature drift (attribution only, if the earlier tiers fire). Feature-drift-first misses label
  shift (base rate moving with season/promo/price, AB_DESIGN §11), which is the likely failure here
  and breaks the frozen calibrator while leaving covariates flat.
- **Watch the contacted-band gap and realized precision@budget, not global ECE.** Global ECE on test
  is 0.0080, but ~86% of it is near-zero-P customers who never get contacted. The contacted band is
  mean_pred 0.4768 vs realized 0.4987 → gap −0.022, ~3× the global figure — the §3 under-prediction,
  on exactly the customers the rule contacts.
- **Hand-rolled, no Evidently.** It would only wrap the per-feature tier (the least useful), and its
  0.1/0.25 bands aren't tied to this model. Same reasoning as the hand-rolled uplift (§7). PSI was
  built, run on two cohorts, and dropped — flat both times, arbitrary bands.
- **Demonstrated, not validated.** ~2 labelable months and adjacent cohorts are in-distribution, so no
  tier has caught a real regression; detection is proven only in the synthetic smoke tests. The
  retrain trigger (contacted-band gap past ~2-3× the test reference, or precision@budget under ~0.45)
  is provisional and re-estimated from the first matured production cohorts. Model's locked and no new
  data flows here, so it's a written rule, not a running job.

Full hierarchy, per-tier checks, verified numbers, reference-refresh policy, and how-to-run are in
**MONITORING.md**.