# AB_DESIGN — Retention Offer Experiment

This document specifies the experiment that Phase D points to. The churn model (Phases A–C) ranks
paid customers by calibrated P(churn) and the cost layer (DECISIONS §3) contacts those above a
break-even probability. What neither can supply is **save_rate** — the causal lift in retention
from making an offer — because it is not estimable from historical data: every past customer's
treatment is confounded with their risk. The Hillstrom uplift work (notebook 04) demonstrated the
*method* on a real randomised experiment (Qini-validated: a persuadable head exists and is
rankable, Qini coef 42, top-quintile realised uplift ~2× the rest). This design transfers the
*method*, not Hillstrom's numbers, to get KKBox's own save_rates and reallocate the contact budget
from *who will churn* to *who we can move*.

## 1. Objective & the decision it informs
Estimate the causal save_rate — overall and within the contact list — so the fixed retention
budget goes where contact is positive-EV and is withdrawn where it is not. This is **not** a test
of the churn model (locked in B) nor of offer design (a v2); it measures the single quantity the
model cannot see.

Concretely it answers, for the contact list and its sub-slices: is `save_rate × value > offer_cost`?
At the locked operating point the list is 3,491 paid customers (3.6% of the paid test population,
precision 0.500), and the question is whether a retention offer moves enough of them — and which
slices — to clear the bar.

## 2. Why an experiment, and not observational data
save_rate is causal: `P(retain | contacted) − P(retain | not contacted)` for the same customer.
In historical data, who got contacted is not random — it correlates with risk, channel, and prior
behaviour — so any observed difference confounds the offer's effect with selection. No modelling on
observational data recovers it; this is the uplift-vs-propensity distinction. Only randomised
assignment breaks the confound. Hillstrom (a genuine RCT) validated that the uplift method works;
this design applies that method to KKBox.

## 3. Population & unit of randomisation
**Population:** the model's contact list — paid customers (`is_free = 0`) with calibrated
P(churn) ≥ 0.323 (the 12-month break-even) at a membership expiry. We experiment only on customers
we would actually act on, not the whole base.

A finding from Phase C shapes everything below: at this operating point **the contact list is
entirely the off-auto-renew book** (notebook 03 — on-auto-renew customers give a negative SHAP push
and never clear the threshold). So `is_auto_renew = 0` is not one segment among several; it
describes the whole experimental population. The structure to test lives *within* the off-AR book.

**Unit:** randomise at the **customer (`msno`) level, sticky** for the experiment window. This
deliberately differs from the model's unit (customer × expiry). The model re-scores a customer
every cycle; the experiment cannot, because a customer offered in one cycle and held out the next
would carry the earlier offer's effect into the control outcome (carryover contamination). Sticky
customer-level assignment keeps each customer cleanly always-treated or always-control, so per-cycle
outcomes are independent reads of one assignment. Assignment is stratified (§5).

## 4. Arms, treatment, and the comparison baseline
- **Treatment:** the retention offer (value ≈ ₹150, the `offer_cost` of DECISIONS §3), at expiry.
- **Control:** held out — no proactive contact.

v1 is a single offer. Offer *size* is deliberately held fixed even though Phase C found price to be
a real churn driver (raw churn rises ~10× across price quartiles, not a collinear artifact) — the
price/offer-size lever is the prioritised **v2** arm (§7), kept out of v1 so the per-slice save_rate
is not confounded with offer-size variation.

**Comparison baseline.** "Offers retain people" is not the goal; "targeting beats the policy we
already have" is. The incumbent policy contacts the whole list (all P ≥ 0.323) uniformly. The
readout is the net EV of a **slice-pruned** policy (contact only positive-EV slices) versus this
**contact-all-above-threshold** baseline. That difference is the model's incremental value.

## 5. Strata within the off-AR book — deterministic pre-treatment rules
The contact list is the off-auto-renew book; the levers Phase C surfaced are slices *within* it, not
peer segments. They are re-expressed as deterministic, pre-treatment rules (computable at expiry, no
dependence on the fitted model), used as randomisation strata:

- **new / first-cycle:** `n_prior_cycles = 1` — 943 of 3,491 (27%); the highest-risk slice
  (short-tenure churn ~0.59). An onboarding problem more than a renewal one.
- **dormant:** `has_activity_60d = 0` — no listening in the 60-day pre-expiry window; the
  re-engagement slice.
- **tenured active (the bulk):** off-AR, `n_prior_cycles > 1`, has recent activity — 1,923 (55%)
  are tenured > 1 year (churn ~0.50). The core renewal/payment-method-nudge population.

A customer can satisfy more than one rule; assign by fixed priority (new > dormant > tenured) or
analyse as overlapping strata — decided before launch, not after. Randomisation is **stratified by
slice** so each yields its own powered save_rate. Targeting is at the slice (coarse) level, not the
per-customer uplift score — the Hillstrom tiers showed fine-grained ranking below the top was noise.

## 6. Metrics & estimand
- **Primary outcome:** renewal within 30 days of expiry — the inverse of the churn label
  (DECISIONS §1), measured with the same point-in-time discipline.
- **Estimand:** `save_rate = P(retain | treatment) − P(retain | control)`, overall and per slice.
  This is exactly the `save_rate` DECISIONS §3 currently *assumes* at 0.30; the experiment replaces
  the assumption with a measured number.
- **Guardrails:** net revenue / ARPU retained (the offer must not cost more than it saves); offer
  take-rate among customers who would have renewed anyway (cannibalisation — the dominant cost
  risk); opt-out / complaint rate; downstream ARPU suppression from discounting.

## 7. Hypotheses
Stated to be tested, not assumed — the C/D priors are carried only as priors:
- **whole off-AR list:** save_rate > break-even (~0.19). Prior: the off-AR book is the clear lever
  (paying, high-risk, persuadable in principle).
- **new / first-cycle:** save_rate vs break-even (~0.16). Prior: mixed — highest churn, but largely
  an onboarding gap, so a renewal offer may underperform a product/onboarding fix.
- **dormant:** save_rate vs its break-even. Prior: likely lowest lift (re-engagement is hard once
  silent).
- **Explicitly test for negative save_rate** in every slice. Hillstrom had no true sleeping-dogs; we
  do not presume KKBox is the same — we test for it.
- **v2 (offer size / price):** because price is a real driver, a follow-up arm varies offer size to
  estimate save_rate elasticity — held out of v1 to keep the per-slice estimate clean.

## 8. Power & sample size
The minimum detectable effect is anchored to the decision: the save_rate that flips a slice's EV
sign — its **break-even save_rate**, `offer / (P(churn) × value)` at the 12-month horizon
(value = ARPU ₹129 × 12 = ₹1,548):

| slice | churn on the list (C) | break-even save_rate |
|---|---|---|
| new / first-cycle (≤30d tenure) | ~0.59 | ~0.16 |
| whole off-AR list | ~0.50 | ~0.19 |
| tenured > 1 year | ~0.50 | ~0.19 |

The experiment must estimate each slice's save_rate tightly enough to place it above or below that
line. Two-proportion test (α = 0.05 two-sided, power = 0.80), per arm per slice:

```
n ≈ (z_{α/2} + z_β)² · [ p_c(1−p_c) + p_t(1−p_t) ] / δ²
```

with control retention `p_c` ≈ 0.41–0.56 (= 1 − slice churn) and MDE `δ`:
- δ = 0.10 (a 10pp save_rate): **~400 / arm** (~800 / slice)
- δ = 0.05 (a 5pp save_rate): **~1,550 / arm** (~3,100 / slice)

**Accrual & duration.** Customers reach expiry on monthly cohorts (notebook 01). The list counts
above are sample-scale (3,491 on the ~98k paid test set); at production scale (full base, ~30×) the
tenured-active slice and the list overall accrue enough per monthly cohort to reach a 10pp-MDE
readout within ~1–2 cycles; a 5pp MDE needs roughly 4× longer.

**Underpowered-slice contingency.** new/first-cycle (943 in-sample) and especially the dormant slice
are the small ones and may not reach power in a reasonable window. Pre-specified: if a slice has not
reached target n by the planned horizon, either (a) extend its accrual while the larger slices read
out and stop, or (b) report it **inconclusive** rather than over-interpret a noisy estimate. Slices
are not pooled — different break-even, different mechanism.

## 9. Analysis plan (pre-registered)
- **Intention-to-treat:** analyse by assigned arm regardless of offer delivery/redemption; report a
  complier-adjusted estimate (offer redeemed vs merely sent) as secondary.
- **Test:** two-proportion comparison of retention, per slice and overall; reported as save_rate
  (the uplift), not raw retention.
- **Multiple comparisons:** Holm correction across the slice-level tests.
- **Fixed horizon / no peeking:** the primary analysis runs once, at the pre-set readout. If interim
  looks are needed for governance, use an alpha-spending boundary; ad-hoc peeking is not permitted
  (modest effects accruing over months make it a real false-positive risk).

## 10. Decision rule (pre-committed)
For each slice, after readout: **contact in production iff `save_rate × value > offer_cost`** —
i.e. measured save_rate exceeds that slice's break-even. Slices that clear it stay in the contacted
population; those that don't are dropped, and that budget retires or shifts toward the clearing
slices. The net EV of this slice-pruned policy is compared to the incumbent
contact-all-above-threshold policy (§4); the difference is the realised value of the work. The
measured save_rates feed straight back into the DECISIONS §3 cost layer (ARPU ₹129, value ₹1,548,
break-even 0.323), replacing the 0.30 assumption per slice.

## 11. Threats to validity & guardrails
- **Cannibalisation** (largest risk): customers who would have renewed anyway take the discount —
  pure cost, no benefit. The control arm prices this directly (its renewal rate is the
  counterfactual), and it is a named guardrail (§6).
- **Seasonality:** cohort churn rates are volatile month to month (notebook 01 — e.g. promo spikes).
  Run across multiple cohorts and block the analysis by cohort so one wave can't dominate.
- **Carryover** is handled by sticky assignment (§3). **Interference / SUTVA** is assumed negligible
  for an individual subscription offer (no referral mechanic) — stated, not merely hoped.
- **Novelty / Hawthorne:** monitor effect stability across cohorts; a decaying lift signals novelty,
  not a durable save_rate.
- **Cost of the holdout:** withholding offers from at-risk control customers is a real, accepted cost
  of learning. Keep the control share only as large as power requires, time-box the experiment, and
  cap total offer spend.
