# DECISIONS

## 1. What counts as churn
A customer has churned if they have NOT renewed within 30 days after their
subscription expires. (Expiry date + 30 days, no renewal → churned.)
Chosen because it's a hard yes/no decidable at a fixed date — unlike a fuzzy
rule like "after several notifications," which isn't cleanly measurable.

## 2. Prediction window
We predict AT the expiry date. Everything before expiry is the information
(features) we use. The label is resolved by the 30 days after expiry: a
renewal in that window → stayed, none → churned.
Consequence: we can only label customers whose 30-day window has already
passed — otherwise we don't yet know the answer. So training uses only
customers old enough that their outcome is known.

## 3. Cost model
Contact a customer only when expected gain beats the offer cost:
    P(churn) × save_rate × value_of_keeping > offer_cost
The cutoff is the break-even probability this implies — `offer / (save × value)`
— not a default 0.5.

Terms (fixed where assumed, derived where possible):
    offer_cost = ₹150   — the retention offer extended
    save_rate  = 0.30   — assumed lift from contact; the SOFTEST number, validated in Phase D (uplift)
    value      = ARPU × horizon, ARPU = ₹149/mo DERIVED FROM DATA
                 (median of actual_amount_paid / payment_plan_days × 30 over paid test rows;
                  lands on the standard monthly plan price → grounded, not a guess)

`value` is swept over a 6–24 month retention horizon rather than pinned, because the
"right" horizon is a business judgement and the rule should be shown robust to it.
Backtested on the test set the rule is profitable at EVERY horizon (net rises as the
horizon lengthens), so the conclusion does not hinge on the choice.

Operating point we quote: horizon = 12 months → value = ₹149 × 12 = ₹1,788 →
break-even ≈ 0.28. So we contact paid customers above ~0.28 calibrated P(churn) — not 0.5,
and not the earlier ~0.63 (which came from a flat ₹800 placeholder, now retired).

Probabilities MUST be calibrated before entering this formula (PROJECT_RULES.md rule #7): the
threshold reads P(churn) as real money, so the score has to be an honest probability.

## 4. Unit of analysis & train/val/test split
Training rows are **monthly expiry cohorts**: for each calendar month, one row
per customer whose membership expires that month. Features are point-in-time as
of that expiry (decision §2); the label is the next-30-days outcome (§1).

A customer **recurs across cohorts** — one row per renewal cycle — because that
mirrors a production churn model, which re-scores the same customer base every
cycle. The unit of analysis is therefore a (customer, expiry-month) prediction
point, not a customer.

The split is **temporal by expiry month**: earlier months → train, a middle
block → validation, the latest labelable months → test. A customer may appear in
more than one split.

**Why this is leakage-safe without keeping each customer on one side.**
Leakage is prevented by strict point-in-time features (every feature uses only
data on or before that row's cutoff), not by disjoint customers. A customer's
January row and March row are independent observations made with different
information, so the same `msno` landing in both train and test is not future
leakage — it is the deployment reality.
One-line defence: *"point-in-time features, not disjoint splits, are the correct
leakage guard for a model that re-scores the same base."*

**Trade-off, stated honestly.** We accept a small risk that the model picks up
stable per-customer idiosyncrasies, in exchange for (a) a design that matches how
the model is actually deployed, and (b) avoiding the bias of a one-row-per-customer
design, where the latest cohorts would skew toward still-active users (a churned
user's last expiry is old, so recent test rows would under-represent churn).

**Rule considered and rejected: "each customer on one side of the split."** That
rule answers "how well do I predict for customers I've *never seen*?" — the
cold-start / fraud question. The churn question is "how well do I predict for
customers I keep re-scoring?", so recurring customers in the test set is the more
faithful evaluation here. (This consciously revises the earlier non-negotiable
rule #5; PROJECT_RULES.md is updated to match.)

## 5. Model scope — paid only; free trials handled by rule
The model is trained and evaluated on **paid** prediction points only (`is_free = 0`).
Free trials (`actual_amount_paid = 0`) are NOT scored by the model; they are handled by a
**rule**: `is_free = 1` → ~93% churn → route to a conversion flow, not a retention offer.

Why scope it out rather than let one model do both:
- The cost model (§3) assumes a paying customer with revenue to protect. A trial has no
  ARPU to retain, so the EV formula is meaningless for it — "convert, don't retain" is the
  correct, different intervention.
- Trials are trivially separable (a single `is_free` split cleaves the ~93%-churn block), so
  a rule captures them as well as the model would, and keeps the headline metric honest:
  with trials in, topline PR-AUC ≈ 0.84 is mostly a free-trial detector; on paid only it is
  ≈ 0.30 — the real, deployable number.
- Verified scoping costs no accuracy: refitting on paid-only gave the same ≈0.30 as the
  all-population model evaluated on paid (0.296 vs 0.296). The `is_free` split was "free" to
  the tree, not capacity-stealing — so scoping is a decision about honesty and economics,
  not a performance trade-off.

Note on `is_auto_renew`: it is the analogous dominant binary cleave *within* the paid model
(`auto_renew = 0` ≈ 29% churn). It is deliberately NOT scoped out the way trials are — an
`auto_renew = 0` customer is paying, high-risk, and retainable, i.e. exactly the target,
whereas a trial is none of those.
