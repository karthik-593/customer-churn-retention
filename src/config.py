"""paths + feature list. one home for the constants that were retyped across notebook cells,
so nothing downstream re-declares the feature set from memory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# the notebooks run from notebooks/ with ../data/churn.duckdb -> that resolves to <repo>/data.
# resolved from this file so src/ modules work regardless of the cwd they're run from.
DB_PATH = REPO_ROOT / "data" / "churn.duckdb"
SQL_DIR = REPO_ROOT / "sql"

# base12 — the 12 paid, point-in-time features the shipped scorer consumes (DECISIONS.md §6).
# deliberately EXCLUDED: payment_method_id (poison — memorised, mix shifts across the split)
# and the lifecycle pair (tenure_days, n_prior_cycles — overfit the topline under the temporal
# split; carried to the phase-D segment layer instead). is_free is built but only feeds the
# routing rule (free -> conversion flow), it is not a scored feature.
BASE12 = [
    "is_auto_renew", "payment_plan_days", "actual_amount_paid", "plan_list_price", "discount",
    "has_activity_60d", "recency_days", "active_days_30", "secs_30", "unq_30",
    "completion_ratio", "activity_trend",
]

# --- MLflow (E2) ----------------------------------------------------------------
# absolute paths from REPO_ROOT so they're immune to cwd changes (db.connect() chdirs
# into notebooks/ to resolve the CSV-backed views). all three live at the repo root and
# are gitignored (mlruns/, mlartifacts/, *.db).
MLFLOW_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_URI = (REPO_ROOT / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT   = "churn-base12"
MODEL_NAME          = "churn-base12"   # registered model; E3 loads models:/churn-base12@prod
MODEL_ALIAS         = "prod"

# --- decision layer (E3) --------------------------------------------------------
# VALUE rule, not a churn cutoff: contact iff the expected saved value clears the offer —
#   EV = SAVE_RATE * value * P(churn) - OFFER  >= 0   <=>   P >= OFFER/(SAVE_RATE*value)
# under a flat SAVE_RATE this collapses to one break-even (0.323); phase D makes SAVE_RATE
# segment-specific and the SAME rule still targets "worth contacting". the threshold is always
# DERIVED from these params, never hardcoded. MEDIAN_MONTHLY_PAID/SAVE_RATE are documented
# business inputs (derived once in notebook cell 38), not refit each run.
OFFER               = 150   # NT$, retention offer cost per contact
SAVE_RATE           = 0.30  # P(retain | contacted), flat global rate (phase D: per-segment)
HORIZON_MONTHS      = 12    # value horizon
MEDIAN_MONTHLY_PAID = 129   # NT$/mo — median (not mean): robustness to high-value tail; understates mean customer value → raises break-even → contacts fewer; conservative, chosen deliberately
