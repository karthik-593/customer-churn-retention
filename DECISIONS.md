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