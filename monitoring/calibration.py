"""phase F step 1 — calibration / realized-rate check (the monitoring spine).

the system has no ground truth at score time; the only honest "is the rupee math still right?"
signal is predicted calibrated P vs realized churn, 30d later. this measures exactly that on a
labelled cohort: overall (mean_pred vs realized base_rate), per-decile reliability (reproduces
train.py's cell-46 table), and IN THE CONTACTED BAND (P >= break-even) where the decision actually
spends. label-shift (a moving base rate, §3 / AB_DESIGN §11) breaks the frozen calibrator here
first, and shows up in `gap` even when covariates barely move.

run on the locked test split, these numbers ARE the reference a future matured cohort is judged
against — no pass/fail band is hardcoded here; that belongs to the cohort-comparison step, defined
off these figures, not off folklore.

    python -m monitoring.calibration            # split=test (default)
    python -m monitoring.calibration val        # the calibrator's own fit set (sanity: ~exact)
"""
import numpy as np
import pandas as pd

from src import config, db

REPORTS = config.REPO_ROOT / "reports"

# same cohort + label join as src.score (label lives in cohorts_s; model_data never carries it),
# but parametrised by split so the SAME monitor later points at a matured production cohort.
COHORT = """
    SELECT m.*, c.is_churn
    FROM model_data m JOIN cohorts_s c USING (msno, expiry)
    WHERE m.is_free = 0 AND c.split = ?
    ORDER BY m.msno, m.expiry
"""


def calibration_metrics(p, y, break_even, n_bins=10):
    """pure: calibrated P vs realized label -> (summary, reliability df). no db / no model, so it
    is unit-testable on a fixture. the reliability table mirrors train.py cell 46 exactly (qcut(10),
    pred/actual/n) so a test-split run reconciles against the logged reliability.csv."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    rel = (df.groupby("bin", observed=True)
             .agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
             .round(4))

    # ECE: n-weighted mean |pred - actual| across bins. one honest scalar to band on later —
    # anchored to this model's own reliability, not the PSI 0.1/0.25 folklore.
    ece = float((rel["n"] * (rel["pred"] - rel["actual"]).abs()).sum() / rel["n"].sum())

    contacted = p >= break_even   # the decision: contact iff P >= break-even (== EV >= 0)
    summary = dict(
        n=len(y),
        base_rate=float(y.mean()),                 # realized churn on the cohort
        mean_pred=float(p.mean()),                 # mean calibrated P
        gap=float(p.mean() - y.mean()),            # <0 => model UNDER-predicts (the §3 risk)
        ece=ece,
        n_contacted=int(contacted.sum()),
        # realized churn among contacted == backtest precision; cross-checks src.score.evaluate()
        precision=float(y[contacted].mean()) if contacted.any() else float("nan"),
        mean_pred_contacted=float(p[contacted].mean()) if contacted.any() else float("nan"),
    )
    return summary, rel


def _load_model():
    import mlflow   # lazy: keeps the metric math (calibration_metrics) importable without the
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)   # serving/registry stack installed.
    return mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")


def report(split="test"):
    con = db.connect()
    df = con.execute(COHORT, [split]).df()
    y = df.is_churn.astype(int).to_numpy()
    p = np.asarray(_load_model().predict(df[config.BASE12]))   # pyfunc emits CALIBRATED P

    be = config.OFFER / (config.SAVE_RATE * config.MEDIAN_MONTHLY_PAID * config.HORIZON_MONTHS)
    s, rel = calibration_metrics(p, y, be)

    arrow = "under-predicts" if s["gap"] < 0 else "over-predicts"
    print(f"calibration report — split={split}, paid (is_free=0)")
    print(f"n={s['n']}  realized base_rate={s['base_rate']:.4f}  mean_pred(cal)={s['mean_pred']:.4f}"
          f"  gap={s['gap']:+.4f} (model {arrow})")
    print(f"ECE(10-bin)={s['ece']:.4f}")
    print(f"\ncontacted band (P >= {be:.3f}):")
    print(f"  n_contacted={s['n_contacted']}  realized_churn(precision)={s['precision']:.4f}"
          f"  mean_pred={s['mean_pred_contacted']:.4f}")
    print("\nreliability (calibrated, 10 bins) — pred should track actual:")
    print(rel.to_string())

    REPORTS.mkdir(exist_ok=True)
    rel.to_csv(REPORTS / f"calibration_{split}.csv")
    print(f"\nwrote {REPORTS / f'calibration_{split}.csv'}")
    return s, rel


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else "test")
