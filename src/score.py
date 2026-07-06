"""E3 step 1 — evaluate(): score the labelled TEST cohort with the registered model, apply the
value rule, and reconcile against notebook cell 38 (cost-based model selection).

value rule (NOT a churn cutoff): contact iff the expected saved value clears the offer cost —
    EV = SAVE_RATE * value * P(churn) - OFFER  >= 0
which is algebraically P >= OFFER/(SAVE_RATE*value) = break-even. under the flat global save rate
this collapses to a single threshold (0.323); phase D varies SAVE_RATE per segment and the SAME
code still targets "worth contacting". the threshold is derived, never a hardcoded 0.323.

run: python -m src.score     (after src.features + src.train)
"""
import numpy as np
import mlflow

from . import config, db, data, features

REPORTS = config.REPO_ROOT / "reports"

# paid test cohort + label (label lives in cohorts_s; model_data never carries it)
TEST_PAID = """
    SELECT m.*, c.is_churn
    FROM model_data m JOIN cohorts_s c USING (msno, expiry)
    WHERE m.is_free = 0 AND c.split = 'test'
    ORDER BY m.msno, m.expiry
"""


def _load_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")


def evaluate():
    con = db.connect()
    df = con.execute(TEST_PAID).df()
    y = df.is_churn.astype(int).to_numpy()

    # the registered pyfunc emits CALIBRATED P directly (calibrator is bundled in)
    p = np.asarray(_load_model().predict(df[config.BASE12]))

    value = config.ARPU * config.HORIZON_MONTHS                # Rs retained per save
    be = config.OFFER / (config.SAVE_RATE * value)             # derived break-even
    ev = config.SAVE_RATE * value * p - config.OFFER           # per-customer EV of contacting
    contact = ev >= 0                                          # == P >= be, by construction

    # reproducibility check: ARPU re-derived from this cohort should match the config default
    pm = df.payment_plan_days > 0
    arpu_chk = (df.loc[pm, "actual_amount_paid"] / df.loc[pm, "payment_plan_days"] * 30).median()

    contacted = int(contact.sum())
    tp = int(y[contact].sum())                                 # churners among contacted
    precision = tp / contacted if contacted else float("nan")
    would_stay = contacted - tp
    net = tp * config.SAVE_RATE * value - contacted * config.OFFER

    order = np.argsort(-p)                                     # rank all test by P for prec@budget
    prec_at = lambda k: float(y[order[:k]].mean())

    print(f"ARPU Rs {config.ARPU} (derived {arpu_chk:.0f}) | value(12mo) Rs {value} | break-even {be:.3f}")
    print(f"\n{'':17} contacted  precision  would_stay   net_Rs  prec@1000  prec@3000")
    print(f"{'base12 (locked)':17} {contacted:9d}  {precision:9.3f}  {would_stay:10d}  {net:7.0f}  "
          f"{prec_at(1000):9.3f}  {prec_at(3000):9.3f}")
    print(f"{'cell 38 target':17} {2772:9d}  {0.538:9.3f}  {1282:10d}  {276156:7d}  {0.704:9.3f}  {0.520:9.3f}")

    if abs(arpu_chk - config.ARPU) > 1:
        print(f"\n[warn] derived ARPU {arpu_chk:.1f} != config {config.ARPU} — reconcile before trusting net")

    # write the scored cohort, ranked by EV (worth), for inspection (gitignored reports/)
    REPORTS.mkdir(exist_ok=True)
    out = df[["msno", "expiry"]].copy()
    out["p_churn"], out["expected_value"], out["contact"] = p, ev, contact
    out = out.sort_values("expected_value", ascending=False)
    out.to_csv(REPORTS / "eval_test.csv", index=False)
    print(f"\nwrote {REPORTS / 'eval_test.csv'} ({len(out)} rows, {contacted} flagged to contact)")
    return dict(contacted=contacted, precision=precision, net=net)


def score(cutoff=None):
    """batch scorer (the product): score the FULL paid base for one cutoff month and write the
    ranked contact list. unlabelled — the label is unknown at scoring time. free trials are routed
    out by rule (is_free=1 -> conversion flow), never scored for retention."""
    con = db.connect()
    if cutoff is None:
        cutoff = data.latest_cutoff(con)

    # build features for this month's points via the SAME path as training (parity); out_table
    # keeps the training model_data intact.
    features.build_features(con, data.points_sql(cutoff), out_table="score_data")
    df = con.execute("SELECT * FROM score_data").df()

    # routing rule: paid scored; free -> conversion flow (counted, not scored)
    paid = df[df.is_free == 0].copy()
    n_free = int((df.is_free == 1).sum())

    p = np.asarray(_load_model().predict(paid[config.BASE12]))
    value = config.ARPU * config.HORIZON_MONTHS
    be = config.OFFER / (config.SAVE_RATE * value)
    paid["p_churn"] = p
    paid["expected_value"] = config.SAVE_RATE * value * p - config.OFFER   # EV of contacting
    paid["contact"] = paid["expected_value"] >= 0                         # == P >= break-even

    cl = (paid[paid.contact]
          .sort_values("expected_value", ascending=False)
          [["msno", "expiry", "p_churn", "expected_value"]]
          .reset_index(drop=True))

    con.register("cl_df", cl)
    con.execute("CREATE OR REPLACE TABLE contact_list AS SELECT * FROM cl_df")
    REPORTS.mkdir(exist_ok=True)
    cl.to_csv(REPORTS / "contact_list.csv", index=False)

    print(f"cutoff {cutoff} | value Rs {value} | break-even {be:.3f}")
    print(f"base this month : {len(df)}  (paid {len(paid)}, free->conversion {n_free})")
    pct = len(cl) / len(paid) if len(paid) else float("nan")
    print(f"contacted       : {len(cl)} ({pct:.1%} of paid), EV>=0, ranked by expected value")
    if len(cl):
        print(f"EV range Rs     : {cl.expected_value.max():.0f} (top) -> {cl.expected_value.min():.0f} (last)")
    print(f"wrote contact_list table + {REPORTS / 'contact_list.csv'}")
    return cl


if __name__ == "__main__":
    import sys
    # optional cutoff: `python -m src.score 2017-02-01`; default = latest cohort_month in pred_points
    score(sys.argv[1] if len(sys.argv) > 1 else None)
