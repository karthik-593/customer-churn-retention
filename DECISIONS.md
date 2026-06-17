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
Placeholder numbers (refine on real data):
    value ≈ ₹800, offer cost ≈ ₹150, save_rate ≈ 0.30
→ break-even churn probability ≈ 0.63, so we contact above ~0.63, not 0.5.
The cost model — not a default threshold — sets the cutoff.

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
