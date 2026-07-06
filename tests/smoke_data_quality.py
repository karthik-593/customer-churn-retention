"""smoke: quality_checks on a valid table + deliberately corrupted copies, each tripping one check.
no db, no mlflow. run: python -m tests.smoke_data_quality
"""
import numpy as np
import pandas as pd

from monitoring.data_quality import quality_checks


def _valid(n=100):
    rng = np.random.default_rng(0)
    n_silent = int(n * 0.20)                                   # ~ baseline 0.213, under the warn ceiling
    return pd.DataFrame({
        "is_auto_renew": rng.integers(0, 2, n),
        "payment_plan_days": [30] * n,
        "actual_amount_paid": rng.integers(99, 199, n),
        "plan_list_price": [149] * n,
        "discount": rng.integers(0, 50, n),                    # all >= 0
        "has_activity_60d": [0] * n_silent + [1] * (n - n_silent),
        "recency_days": rng.integers(0, 61, n),
        "active_days_30": rng.integers(0, 31, n),
        "secs_30": rng.random(n) * 1000,
        "unq_30": rng.random(n) * 100,
        "completion_ratio": rng.random(n),
        "activity_trend": rng.random(n) * 2,
        "is_free": [0] * n,
    })


def _status(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_valid_all_pass():
    df = _valid()
    checks = quality_checks(df, n_points=len(df))
    assert all(c.status == "pass" for c in checks), [c for c in checks if c.status != "pass"]
    print("[1] valid table: every check pass  OK")


def test_each_fault_trips_its_check():
    cases = [
        ("schema/columns",       lambda d: d.drop(columns=["secs_30"]),                "fail"),
        ("schema/numeric",       lambda d: d.assign(unq_30=d["unq_30"].astype(str)),    "fail"),
        ("values/non_null",      lambda d: d.assign(recency_days=d["recency_days"].mask(d.index < 3)), "fail"),
        ("values/binary",        lambda d: d.assign(is_auto_renew=d["is_auto_renew"].mask(d.index < 1, 2)), "fail"),
        ("values/non_negative",  lambda d: d.assign(secs_30=d["secs_30"].mask(d.index < 1, -5.0)), "fail"),
        ("values/completion∈[0,1]", lambda d: d.assign(completion_ratio=d["completion_ratio"].mask(d.index < 1, 1.5)), "fail"),
        ("values/recency∈[0,60]", lambda d: d.assign(recency_days=d["recency_days"].mask(d.index < 1, 70)), "fail"),
    ]
    for name, mutate, want in cases:
        checks = quality_checks(mutate(_valid()), n_points=len(_valid()))
        got = _status(checks, name)
        assert got == want, f"{name}: got {got}, want {want}"
    print("[2] each corrupted column trips exactly its check (fail)  OK")


def test_row_parity_fail():
    df = _valid(100)
    checks = quality_checks(df, n_points=120)                  # 20 points dropped by a join break
    assert _status(checks, "structure/row_parity") == "fail"
    print("[3] row_parity: rows_out 100 vs points_in 120 -> fail  OK")


def test_silent_spike_warns_not_fails():
    # an engagement-join break repaints rows as dormant: silent share jumps past 2x baseline.
    df = _valid(100)
    df["has_activity_60d"] = [0] * 60 + [1] * 40                # 0.60 >> 0.213*2 = 0.426
    checks = quality_checks(df, n_points=len(df))
    assert _status(checks, "structure/silent_share") == "warn"
    assert all(c.status != "fail" for c in checks)             # a join break is a WARN, not a stop
    print("[4] silent-share spike 0.60: warn (not fail)  OK")


def test_discount_negative_surge_warns():
    df = _valid(100)
    df.loc[df.index < 20, "discount"] = -10                    # 0.20 share paid>list
    checks = quality_checks(df, n_points=len(df))
    assert _status(checks, "values/discount_negative") == "warn"
    print("[5] discount-negative surge 0.20: warn  OK")


if __name__ == "__main__":
    test_valid_all_pass()
    test_each_fault_trips_its_check()
    test_row_parity_fail()
    test_silent_spike_warns_not_fails()
    test_discount_negative_surge_warns()
    print("\nall data-quality smoke tests passed")
