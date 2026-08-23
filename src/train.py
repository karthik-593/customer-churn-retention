"""E2 — train the LOCKED base12 scorer, calibrate, log to MLflow, register.

reproduces notebook 02:
  - cell 32: fit base12 on paid rows (is_free=0), temporal split, exact XGB config
  - cell 46: isotonic calibration (raw VAL scores -> val labels), applied to test; PR-AUC + Brier

run (after `python -m src.features` has built model_data):
    python -m src.train
"""
import json
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from xgboost import XGBClassifier
import mlflow
from mlflow.tracking import MlflowClient

from . import config, db

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))   # import the wrapper as a TOP-LEVEL module so mlflow's
import churn_pyfunc                 # noqa: E402  code_paths re-import resolves the same name

# exact config from notebook cell 32's fit_xgb (the locked scorer we defend).
XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", early_stopping_rounds=50, tree_method="hist",
    n_jobs=1,  # determinism: n_jobs=-1 relocates the isotonic steps and moves the contact count
               # ~17% (2,984 -> 3,491) at cut 0.323; single-thread reproduces the canonical number.
    random_state=42,
)

# labels live in cohorts_s (features never carry them, per E1). join back, paid only, stable order.
LABELLED_PAID = """
    SELECT m.*, c.split, c.is_churn
    FROM model_data m JOIN cohorts_s c USING (msno, expiry)
    WHERE m.is_free = 0
    ORDER BY m.msno, m.expiry
"""


def _ensure_experiment(name, artifact_uri):
    if MlflowClient().get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_uri)


def _register_alias(name, alias):
    # point the alias at the version we just logged (the highest version for this name).
    # alias API requires mlflow >= 2.3; guard so an older install still completes training.
    try:
        c = MlflowClient()
        latest = max(int(v.version) for v in c.search_model_versions(f"name='{name}'"))
        c.set_registered_model_alias(name, alias, latest)
        return f"{name}@{alias} -> v{latest}"
    except Exception as e:
        return f"[warn] alias not set ({e}); load by version instead"


def train():
    con = db.connect()
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='model_data'"
    ).fetchone()[0]
    if not exists:
        raise SystemExit("model_data not found — run `python -m src.features` first.")

    df = con.execute(LABELLED_PAID).df()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ytr, yva, yte = (x.is_churn.astype(int) for x in (tr, va, te))
    feats = config.BASE12

    # 1) fit locked base12
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], verbose=False)
    p_va = xgb.predict_proba(va[feats])[:, 1]
    p_te = xgb.predict_proba(te[feats])[:, 1]

    # 2) calibrate: isotonic on raw VAL scores -> val labels, then map test
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
    p_te_cal = iso.predict(p_te)

    # 3) metrics (raw is the reported number; cal should ~match on PR-AUC, improve Brier)
    metrics = {
        "test_pr_auc_raw": float(average_precision_score(yte, p_te)),
        "test_pr_auc_cal": float(average_precision_score(yte, p_te_cal)),
        "val_pr_auc_raw":  float(average_precision_score(yva, p_va)),
        "brier_raw":       float(brier_score_loss(yte, p_te)),
        "brier_cal":       float(brier_score_loss(yte, p_te_cal)),
        "best_iteration":  float(xgb.best_iteration),
        "cal_max_prob":    float(p_te_cal.max()),
        "base_test":       float(yte.mean()),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }

    # reliability table (notebook cell 46) -> logged so the calibration story is auditable
    chk = pd.DataFrame({"p": p_te_cal, "y": yte.values})
    chk["bin"] = pd.qcut(chk["p"], 10, duplicates="drop")
    reliability = chk.groupby("bin", observed=True).agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(4)

    # 4) log + register the bundled (calibrated) model
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    _ensure_experiment(config.MLFLOW_EXPERIMENT, config.MLFLOW_ARTIFACT_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        joblib.dump(xgb, tmp / "xgb.joblib")
        joblib.dump(iso, tmp / "calibrator.joblib")
        (tmp / "features.json").write_text(json.dumps(feats))
        reliability.to_csv(tmp / "reliability.csv")

        mlflow.log_params({**XGB_PARAMS, "n_features": len(feats),
                           "calibration": "isotonic_on_val", "scope": "is_free=0", "split": "temporal"})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(tmp / "reliability.csv"))

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=churn_pyfunc.CalibratedChurn(),
            artifacts={"xgb": str(tmp / "xgb.joblib"),
                       "calibrator": str(tmp / "calibrator.joblib"),
                       "features": str(tmp / "features.json")},
            code_paths=[str(SRC_DIR / "churn_pyfunc.py")],
            input_example=va[feats].head(3),
            registered_model_name=config.MODEL_NAME,
        )
        alias_msg = _register_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        run_id = run.info.run_id

    print(f"\nrun_id          : {run_id}")
    print(f"test PR-AUC raw : {metrics['test_pr_auc_raw']:.4f}   (notebook target ~0.402)")
    print(f"test PR-AUC cal : {metrics['test_pr_auc_cal']:.4f}   (should ~match raw)")
    print(f"Brier raw->cal  : {metrics['brier_raw']:.5f} -> {metrics['brier_cal']:.5f}   (cal lower)")
    print(f"trees / cal max : {int(metrics['best_iteration'])} / {metrics['cal_max_prob']:.4f}")
    print(f"registered      : {alias_msg}")
    print("\nreliability (calibrated, 10 bins) — pred should track actual:")
    print(reliability.to_string())
    return run_id


if __name__ == "__main__":
    train()
