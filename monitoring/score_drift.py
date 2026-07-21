"""phase F step 2 — score-distribution + contact-count drift (the leading, label-free signal).

runs at SCORE time (no labels), vs the step-1 calibration check which waits 30d for labels. the
model has already weighted the 12 inputs, so consequential covariate moves surface here in the score
distribution — which is why score drift outranks per-feature drift in the hierarchy (§9).

NO global distribution-distance metric (PSI / KS) by design: this is a THRESHOLD decision, so only
the region at and above break-even bites — sub-threshold mass moving costs zero rupees (shown in
steps 1-2). a whole-distribution distance is not decision-anchored, and its 0.1/0.25 bands are the
same folklore disowned for calibration (§9). we read three decision-anchored numbers instead:
  - contact_rate     : mass >= break-even == the budget. the triggerable number; a drop is
                       under-contacting (the §3 risk) showing up label-free, before labels mature.
  - mean P|contacted : composition of who we'd contact. catches drift WITHIN the band that a flat
                       contact_rate hides (same count, different risk mix); it is also the label-free
                       lead on step-1's contacted-band gap (tier-1 predicts it now, tier-2 compares
                       it to realized later — one quantity, leading then lagging).
  - upper-tail quantiles : the interpretable shape view; contacts come from q95+.

reference = the verified-good split (test); current = a scoring cohort built via the SAME
features.py path src.score uses (re-scored here so the locked scorer stays untouched).

    python -m monitoring.score_drift                 # cur=latest cutoff vs ref=test
    python -m monitoring.score_drift 2017-01-01      # a specific cutoff month
"""
import numpy as np
import pandas as pd

from src import config, db, data, features

REPORTS = config.REPO_ROOT / "reports"
SPLIT_PAID = """
    SELECT m.* FROM model_data m JOIN cohorts_s c USING (msno, expiry)
    WHERE m.is_free = 0 AND c.split = ?
"""
QS = (0.90, 0.95, 0.99, 1.0)


def _band_stats(p, break_even):
    c = p >= break_even
    return dict(
        contact_rate=float(c.mean()),
        mean_pred_contacted=float(p[c].mean()) if c.any() else float("nan"),
        q=[float(np.quantile(p, x)) for x in QS],
    )


def score_drift(p_ref, p_cur, break_even):
    """pure: reference vs current calibrated-P distributions -> a (metric x [ref,cur,delta]) table.
    no db / no model, so it is unit-testable on a fixture. every row is decision-anchored — no
    global distance metric, no bins, no bands."""
    p_ref = np.asarray(p_ref, dtype=float)
    p_cur = np.asarray(p_cur, dtype=float)
    rs, cs = _band_stats(p_ref, break_even), _band_stats(p_cur, break_even)

    tab = pd.DataFrame(
        {"ref": [rs["contact_rate"], rs["mean_pred_contacted"], *rs["q"]],
         "cur": [cs["contact_rate"], cs["mean_pred_contacted"], *cs["q"]]},
        index=["contact_rate", "mean_P|contacted", "q90", "q95", "q99", "max"],
    )
    tab["delta"] = tab["cur"] - tab["ref"]
    summary = dict(n_ref=len(p_ref), n_cur=len(p_cur), ref=rs, cur=cs)
    return summary, tab.round(4)


def _load_model():
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")


def report(cutoff=None, ref_split="test"):
    con = db.connect()
    model = _load_model()

    ref = con.execute(SPLIT_PAID, [ref_split]).df()
    p_ref = np.asarray(model.predict(ref[config.BASE12]))

    if cutoff is None:
        cutoff = data.latest_cutoff(con)
    # normalise to 'YYYY-MM-DD': latest_cutoff returns a Timestamp whose ' 00:00:00' tail clutters
    # the DATE literal and is an illegal Windows filename char (':'). one clean token throughout.
    cutoff = str(pd.Timestamp(cutoff).date())
    # build the cutoff cohort via the SAME path src.score uses; out_table keeps training data intact
    features.build_features(con, data.points_sql(cutoff), out_table="score_data")
    cur = con.execute("SELECT * FROM score_data WHERE is_free = 0").df()
    p_cur = np.asarray(model.predict(cur[config.BASE12]))

    be = config.OFFER / (config.SAVE_RATE * config.MEDIAN_MONTHLY_PAID * config.HORIZON_MONTHS)
    s, tab = score_drift(p_ref, p_cur, be)

    print(f"score-drift report — ref={ref_split} vs cur={cutoff}, paid (is_free=0)")
    print(f"n_ref={s['n_ref']}  n_cur={s['n_cur']}  break-even={be:.3f}")
    print(f"\n{tab.to_string()}")
    print("\ncontact_rate = the budget (triggerable); mean_P|contacted = band composition")
    print("(label-free lead on step-1's contacted-band gap); q* = tail shape, contacts from q95+.")

    REPORTS.mkdir(exist_ok=True)
    tab.to_csv(REPORTS / f"score_drift_{cutoff}.csv")
    print(f"\nwrote {REPORTS / f'score_drift_{cutoff}.csv'}")
    return s, tab


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else None)
