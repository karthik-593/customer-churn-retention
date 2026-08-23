"""Hyperparameter search for the LOCKED base12 scorer — a correctness check, not a re-tune.

Runs Optuna/TPE around train.py's XGB_PARAMS on a single temporal split (train/val), no
cross-validation, objective = val PR-AUC. TEST is untouched during the search — it is
evaluated exactly once, at the end, only to compare the search winner against the locked
baseline on equal footing (same isotonic-on-val calibration, same score.py decision rule).

This script does NOT touch train.py/config.py/score.py, does NOT register a model, and does
NOT write to the "churn-base12" experiment or model registry — trials log to a separate
MLflow experiment, "churn-base12-hpo". Even if the tuned config wins on test, this script
does not adopt it: the locked config only changes via a deliberate human edit to train.py
after reviewing this printout. The expected outcome is a NULL result — tuned lands inside
the documented ±1% contact-wobble band and the locked config stays.

Search trials use n_jobs=-1 for speed (XGBoost's multithreaded histogram build is a known
source of run-to-run wobble, but it doesn't matter mid-search). Everything downstream of the
search — the stability check, the pipeline fingerprint, and the locked-vs-tuned test
comparison — forces n_jobs=1 + random_state=42 so those numbers are reproducible.

run (after `python -m src.features` has built model_data):
    python -m src.hpo
"""
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import mlflow
from mlflow.tracking import MlflowClient

from . import config, db

SEED = 42
N_TRIALS = 40
HPO_EXPERIMENT = "churn-base12-hpo"   # separate from config.MLFLOW_EXPERIMENT ("churn-base12")

# exact config from train.py's XGB_PARAMS — the locked scorer this script checks, not re-derives.
LOCKED_XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", early_stopping_rounds=50, tree_method="hist", n_jobs=-1, random_state=42,
)

# fixed for every trial (not searched): early stopping IS the tree-count tuner, so n_estimators
# is a ceiling not a knob. learning_rate is held at the locked value too — it wasn't named in the
# search space, so it's treated as fixed rather than silently searched. scale_pos_weight stays OFF
# (deliberate: it would distort the probability level the cost rule needs calibrated).
FIXED_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50,
    eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42,
)

# labels live in cohorts_s (features never carry them, per E1) — identical query to train.py.
LABELLED_PAID = """
    SELECT m.*, c.split, c.is_churn
    FROM model_data m JOIN cohorts_s c USING (msno, expiry)
    WHERE m.is_free = 0
    ORDER BY m.msno, m.expiry
"""


def _ensure_experiment(name, artifact_uri):
    if MlflowClient().get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_uri)


def _load_split():
    con = db.connect()
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='model_data'"
    ).fetchone()[0]
    if not exists:
        raise SystemExit("model_data not found — run `python -m src.features` first.")
    df = con.execute(LABELLED_PAID).df()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ytr, yva, yte = (x.is_churn.astype(int) for x in (tr, va, te))
    return tr, va, te, ytr, yva, yte


def _fit(params, tr, va, ytr, yva, feats):
    xgb = XGBClassifier(**params)
    xgb.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], verbose=False)
    return xgb


def _run_pipeline(params, tr, va, te, ytr, yva, yte_arr, feats, value, be):
    """the one pipeline every locked/tuned/stability fit runs through: fit on train w/ early
    stopping on val -> isotonic on val -> apply to test -> score.py's decision layer at `be`."""
    model = _fit(params, tr, va, ytr, yva, feats)
    p_va = model.predict_proba(va[feats])[:, 1]
    p_te = model.predict_proba(te[feats])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
    p_te_cal = iso.predict(p_te)

    # same decision layer as score.py: EV = SAVE_RATE*value*P - OFFER >= 0  <=>  P >= be
    ev = config.SAVE_RATE * value * p_te_cal - config.OFFER
    contact = ev >= 0
    contacted = int(contact.sum())
    tp = int(yte_arr[contact].sum())
    precision = tp / contacted if contacted else float("nan")
    net = tp * config.SAVE_RATE * value - contacted * config.OFFER

    return dict(
        model=model, p_te_cal=p_te_cal,
        val_pr_auc=float(average_precision_score(yva, p_va)),
        pr_auc=float(average_precision_score(yte_arr, p_te_cal)),
        contacted=contacted, precision=precision, net=net,
        best_iteration=int(model.best_iteration), cal_max_prob=float(p_te_cal.max()),
    )


def _topk_metrics(p_te_cal, yte_arr, k, value):
    """precision/net for the top-k TEST rows by calibrated P — a fixed budget, so it compares
    pure ranking with the threshold difference removed."""
    order = np.argsort(-p_te_cal)[:k]
    tp = int(yte_arr[order].sum())
    precision = tp / k if k else float("nan")
    net = tp * config.SAVE_RATE * value - k * config.OFFER
    return precision, net


def _locked_cut_to_match(p_te_cal, yte_arr, target_k, value):
    """raise the cut on locked's calibrated TEST P until contact count is as close as possible
    to target_k. diagnostic only — this does NOT change the deployed 0.323 cut anywhere else."""
    order = np.argsort(-p_te_cal)
    k = min(max(target_k, 1), len(p_te_cal))
    cut = float(p_te_cal[order[k - 1]])
    contact = p_te_cal >= cut          # ties at the boundary can push achieved > k
    achieved = int(contact.sum())
    tp = int(yte_arr[contact].sum())
    precision = tp / achieved if achieved else float("nan")
    net = tp * config.SAVE_RATE * value - achieved * config.OFFER
    return cut, achieved, precision, net


def _spread_report(label, runs):
    contacts = [r["contacted"] for r in runs]
    precisions = [r["precision"] for r in runs]
    nets = [r["net"] for r in runs]
    spread = max(contacts) - min(contacts)
    print(f"{label:28} contacts={contacts}  min={min(contacts)} max={max(contacts)} spread={spread}")
    print(f"{'':28} precision range=[{min(precisions):.3f}, {max(precisions):.3f}]  "
          f"net range=[{min(nets):,.0f}, {max(nets):,.0f}]")
    return spread


def _suggest_params(trial):
    return dict(
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # reg_lambda=0 isn't representable on a log scale; 1e-3 stands in for "~none".
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 2.0),
    )


def _make_objective(tr, va, ytr, yva, feats):
    def objective(trial):
        params = {**FIXED_PARAMS, **_suggest_params(trial)}
        xgb = _fit(params, tr, va, ytr, yva, feats)
        p_va = xgb.predict_proba(va[feats])[:, 1]
        val_pr_auc = float(average_precision_score(yva, p_va))

        with mlflow.start_run(run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(params)
            mlflow.log_metric("val_pr_auc", val_pr_auc)
            mlflow.log_metric("best_iteration", float(xgb.best_iteration))
            mlflow.set_tags({"trial_number": trial.number, "scope": "is_free=0", "split": "temporal"})

        return val_pr_auc
    return objective


def hpo():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tr, va, te, ytr, yva, yte = _load_split()
    feats = config.BASE12
    yte_arr = yte.to_numpy()

    value = config.MEDIAN_MONTHLY_PAID * config.HORIZON_MONTHS   # NT$ retained per save
    be = config.OFFER / (config.SAVE_RATE * value)                # same break-even as score.py

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    _ensure_experiment(HPO_EXPERIMENT, config.MLFLOW_ARTIFACT_URI)
    mlflow.set_experiment(HPO_EXPERIMENT)

    # === search: TRAIN/VAL only, no cross-validation, no test leakage. trials keep n_jobs=-1
    # (via FIXED_PARAMS) for speed — only the final refit->calibrate->eval below needs to be
    # deterministic, so determinism is enforced there, not here. ===
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="base12-hpo")
    study.optimize(_make_objective(tr, va, ytr, yva, feats), n_trials=N_TRIALS)

    tuned_params = {**FIXED_PARAMS, **study.best_params}

    # === stability check: fit the LOCKED config N=5 times through the FULL pipeline (fit -> val
    # calibration -> test -> decision layer) under two regimes differing ONLY in n_jobs. isolates
    # threading as a variable, to explain a prior run where hpo.py's locked baseline returned
    # 3,491 test contacts against the canonical (notebook, n_jobs=1) 2,984. regime A also serves
    # as THE deterministic locked fit reused below (fingerprint + clean comparison). ===
    N_STABILITY = 5
    regime_a = [_run_pipeline({**LOCKED_XGB_PARAMS, "n_jobs": 1}, tr, va, te, ytr, yva, yte_arr, feats, value, be)
                for _ in range(N_STABILITY)]
    regime_b = [_run_pipeline({**LOCKED_XGB_PARAMS, "n_jobs": -1}, tr, va, te, ytr, yva, yte_arr, feats, value, be)
                for _ in range(N_STABILITY)]

    print(f"\n=== stability check: LOCKED config x{N_STABILITY}, full pipeline, cut {be:.3f} ===")
    spread_a = _spread_report("regime A (n_jobs=1,  seed=42, deterministic)", regime_a)
    spread_b = _spread_report("regime B (n_jobs=-1, seed=42, multithreaded)", regime_b)

    if spread_a == 0 and spread_b > spread_a:
        stability_verdict = (
            f"THREADING CONFIRMED: regime A is flat (spread 0) while regime B swings {spread_b} "
            f"contacts ({min(r['contacted'] for r in regime_b)}-{max(r['contacted'] for r in regime_b)}). "
            f"The documented '±1% run-to-run' wobble claim UNDERSTATES the true swing seen here — flag it for correction."
        )
    elif spread_a > 0:
        stability_verdict = (
            f"NOT (only) threading: the deterministic regime A ALSO swings (spread {spread_a} contacts) "
            f"under a fixed seed and n_jobs=1. Something in this pipeline differs from the notebook's — "
            f"do not conclude threading is the cause until that mismatch is found."
        )
    else:
        stability_verdict = (
            "Both regimes are flat (spread 0) — this run doesn't reproduce a threading-driven swing. "
            "The 3,491-vs-2,984 gap needs another explanation; check the pipeline fingerprint below first."
        )
    print(f"\nstability verdict: {stability_verdict}")

    # === pipeline fingerprint: the deterministic (n_jobs=1) locked fit, to reconcile hpo.py's
    # split/features/calibration against train.py's known outputs and rule out a mismatch there. ===
    fp = regime_a[0]
    print(f"\n=== pipeline fingerprint (locked, n_jobs=1, seed=42) ===")
    print(f"n_train={len(tr)}  n_val={len(va)}  n_test={len(te)}  base_test={yte_arr.mean():.4f}  "
          f"best_iteration={fp['best_iteration']}  cal_max_prob={fp['cal_max_prob']:.4f}")

    # === clean comparison: LOCKED (regime A's deterministic fit, reused — it's the same params,
    # n_jobs=1, seed=42) vs TUNED refit deterministically the same way. test touched once for this. ===
    locked_det = regime_a[0]
    tuned_det = _run_pipeline({**tuned_params, "n_jobs": 1}, tr, va, te, ytr, yva, yte_arr, feats, value, be)
    results = {"locked": locked_det, "tuned": tuned_det}

    # summary run in the SAME hpo experiment (no registration, no touching churn-base12)
    with mlflow.start_run(run_name="locked_vs_tuned_summary"):
        mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})
        mlflow.log_metrics({"search_best_val_pr_auc": study.best_value,
                             "stability_spread_regime_a": spread_a, "stability_spread_regime_b": spread_b})
        for name, r in results.items():
            mlflow.log_metrics({f"{name}_test_{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        mlflow.set_tag("break_even", f"{be:.3f}")

    # --- report ---
    print(f"\noptuna trials   : {N_TRIALS} (TPE, seed={SEED}) — logged to MLflow experiment '{HPO_EXPERIMENT}'")
    print(f"best params     : { {k: round(v, 4) if isinstance(v, float) else v for k, v in study.best_params.items()} }")
    print(f"search best val PR-AUC (n_jobs=-1 during trials) : {study.best_value:.4f}")
    print(f"tuned  val PR-AUC (deterministic n_jobs=1 refit) : {tuned_det['val_pr_auc']:.4f}")
    print(f"locked val PR-AUC (deterministic n_jobs=1 refit) : {locked_det['val_pr_auc']:.4f}")

    print(f"\nbreak-even cut  : {be:.3f}   (config.OFFER / (SAVE_RATE * MEDIAN_MONTHLY_PAID * HORIZON_MONTHS), same as score.py)")
    print(f"\n=== clean comparison (both n_jobs=1, seed=42) ===")
    print(f"{'':17} PR-AUC   contacts  precision   net_NT$")
    for name in ("locked", "tuned"):
        r = results[name]
        print(f"{name:17} {r['pr_auc']:7.4f}  {r['contacted']:8d}  {r['precision']:9.3f}  {r['net']:9.0f}")

    d_pr_auc = results["tuned"]["pr_auc"] - results["locked"]["pr_auc"]
    d_contacts = results["tuned"]["contacted"] - results["locked"]["contacted"]
    d_precision = results["tuned"]["precision"] - results["locked"]["precision"]
    d_net = results["tuned"]["net"] - results["locked"]["net"]
    rel_contacts = d_contacts / results["locked"]["contacted"] if results["locked"]["contacted"] else float("nan")
    print(f"{'Δ (tuned-locked)':17} {d_pr_auc:+7.4f}  {d_contacts:+8d}  {d_precision:+9.3f}  {d_net:+9.0f}   "
          f"(Δcontacts {rel_contacts:+.2%})")

    anchor_contacts = 2984
    print(f"\n[anchor — canonical locked (notebook, n_jobs=1), for your own check, not hardcoded into any logic]")
    print(f"  test {anchor_contacts:,} contacts / 0.526 precision / NT$281,972 at cut 0.323")
    print(f"  deterministic locked row above {'MATCHES' if results['locked']['contacted'] == anchor_contacts else 'DIFFERS FROM'} "
          f"the anchor by {results['locked']['contacted'] - anchor_contacts:+d} contacts.")

    wobble_band = 0.01
    if abs(rel_contacts) <= wobble_band:
        verdict = (f"NULL RESULT — tuned contacts Δ {rel_contacts:+.2%} is within the documented "
                    f"±{wobble_band:.0%} contact-wobble band. KEEP THE LOCKED CONFIG; no file changed.")
    else:
        verdict = (f"tuned contacts Δ {rel_contacts:+.2%} is OUTSIDE the ±{wobble_band:.0%} wobble band — "
                    f"worth a closer look, but this script still does not adopt the tuned config or "
                    f"touch any locked file. That's a deliberate human decision, made elsewhere.")
    print(f"\nVERDICT: {verdict}")

    # === diagnostic: is the tuned advantage a better ranking frontier, or the same frontier at a
    # different operating point? locked/tuned contact counts differ but ΔPR-AUC is ~flat on test
    # and negative on val — ranking didn't improve, which suggests the tuned "gain" may just be a
    # threshold-placement artifact reproducible by moving locked's cut. purely diagnostic: this
    # does NOT change the deployed 0.323 cut anywhere else in the codebase. ===
    p_te_cal_locked = locked_det["p_te_cal"]
    p_te_cal_tuned = tuned_det["p_te_cal"]
    k_locked, k_tuned = locked_det["contacted"], tuned_det["contacted"]

    print(f"\n=== diagnostic: better frontier, or same frontier at a different cut? ===")
    print(f"fixed-budget precision/net — same k for both models removes the threshold difference:")
    print(f"{'':17} k        precision   net_NT$")
    for k in sorted({k_locked, k_tuned}):
        p_l, n_l = _topk_metrics(p_te_cal_locked, yte_arr, k, value)
        p_t, n_t = _topk_metrics(p_te_cal_tuned, yte_arr, k, value)
        print(f"{'locked @k=' + str(k):17} {k:6d}  {p_l:9.3f}  {n_l:9.0f}")
        print(f"{'tuned  @k=' + str(k):17} {k:6d}  {p_t:9.3f}  {n_t:9.0f}")

    cut_match, achieved, prec_match, net_match = _locked_cut_to_match(p_te_cal_locked, yte_arr, k_tuned, value)
    print(f"\nlocked-cut-to-match : locked cut raised to {cut_match:.4f} (off the deployed 0.323 — "
          f"DIAGNOSTIC ONLY, not a re-selected operating point) to hit tuned's {k_tuned} contacts")
    print(f"  achieved {achieved} contacts  precision {prec_match:.3f}  net NT${net_match:.0f}")
    print(f"  tuned (native cut)  : {k_tuned} contacts  precision {tuned_det['precision']:.3f}  "
          f"net NT${tuned_det['net']:.0f}")

    d_net_match = tuned_det["net"] - net_match
    d_prec_match = tuned_det["precision"] - prec_match
    d_val_pr_auc = tuned_det["val_pr_auc"] - locked_det["val_pr_auc"]
    d_test_pr_auc = results["tuned"]["pr_auc"] - results["locked"]["pr_auc"]

    if abs(d_net_match) < 3000 and abs(d_prec_match) < 0.01:
        frontier_verdict = (
            f"ARTIFACT: tuned gain is threshold placement; locked reaches the same frontier by moving "
            f"the cut (matched precision {prec_match:.3f} vs tuned {tuned_det['precision']:.3f}, net "
            f"NT${net_match:.0f} vs NT${tuned_det['net']:.0f}). Ranking equivalent — "
            f"ΔPR-AUC val={d_val_pr_auc:+.4f}, test={d_test_pr_auc:+.4f}."
        )
    else:
        frontier_verdict = (
            f"REAL: tuned frontier dominates near the operating region by Δprecision={d_prec_match:+.3f}, "
            f"Δnet=NT${d_net_match:+.0f} even after matching locked's cut to the same {k_tuned}-contact "
            f"budget — ΔPR-AUC val={d_val_pr_auc:+.4f}, test={d_test_pr_auc:+.4f}."
        )
    print(f"\nFRONTIER VERDICT: {frontier_verdict}")

    return dict(study=study, results=results, regime_a=regime_a, regime_b=regime_b,
                tuned_params=tuned_params, verdict=verdict, stability_verdict=stability_verdict,
                frontier_verdict=frontier_verdict, locked_cut_match=cut_match)


if __name__ == "__main__":
    hpo()
