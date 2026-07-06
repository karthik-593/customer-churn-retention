"""smoke: calibration_metrics on synthetic fixtures with HAND-COMPUTED expected values.
no db, no mlflow — exercises only the metric math (the part I can verify without the real data).
run: python -m tests.smoke_calibration
"""
import numpy as np

from monitoring.calibration import calibration_metrics


def test_perfectly_calibrated():
    # 4 bins, each bin's predicted prob == its realized rate -> ECE == 0, gap == 0.
    # bin a: p=0.05, 5% churn (1/20); b: 0.20, 20% (4/20); c: 0.50, 50% (10/20); d: 0.80, 80% (16/20)
    p, y = [], []
    for prob, k, n in [(0.05, 1, 20), (0.20, 4, 20), (0.50, 10, 20), (0.80, 16, 20)]:
        p += [prob] * n
        y += [1] * k + [0] * (n - k)
    s, rel = calibration_metrics(p, y, break_even=0.323, n_bins=4)

    assert abs(s["ece"]) < 1e-9, s["ece"]                      # perfectly calibrated
    assert abs(s["gap"]) < 1e-9, s["gap"]                      # mean_pred == base_rate
    # base rate = (1+4+10+16)/80 = 31/80 = 0.3875
    assert abs(s["base_rate"] - 0.3875) < 1e-9, s["base_rate"]
    # contacted (p>=0.323): bins c(0.50) and d(0.80) -> 40 rows, churn (10+16)/40 = 0.65
    assert s["n_contacted"] == 40, s["n_contacted"]
    assert abs(s["precision"] - 0.65) < 1e-9, s["precision"]
    print("[1] perfectly-calibrated fixture: ECE=0, gap=0, precision=0.65  OK")


def test_under_prediction():
    # two prediction levels, both below the realized 60% -> model UNDER-predicts (the §3 risk).
    # A: p=0.20, 30/50 churn; B: p=0.40, 30/50 churn.
    p = [0.20] * 50 + [0.40] * 50
    y = ([1] * 30 + [0] * 20) + ([1] * 30 + [0] * 20)
    s, rel = calibration_metrics(p, y, break_even=0.323, n_bins=2)

    assert abs(s["mean_pred"] - 0.30) < 1e-9, s["mean_pred"]   # (0.20*50 + 0.40*50)/100
    assert abs(s["base_rate"] - 0.60) < 1e-9, s["base_rate"]
    assert abs(s["gap"] + 0.30) < 1e-9, s["gap"]               # 0.30 - 0.60 = -0.30, under-predict
    # ECE = (50*|0.20-0.60| + 50*|0.40-0.60|)/100 = (20 + 10)/100 = 0.30
    assert abs(s["ece"] - 0.30) < 1e-9, s["ece"]
    # contacted (p>=0.323): only bin B -> 50 rows, churn 30/50 = 0.60
    assert s["n_contacted"] == 50, s["n_contacted"]
    assert abs(s["precision"] - 0.60) < 1e-9, s["precision"]
    print("[2] under-prediction fixture: gap=-0.30, ECE=0.30, precision=0.60  OK")


def test_empty_contacted_band():
    # a cohort where nothing clears break-even -> empty band must give nan, not crash.
    # (varied p so qcut still bins; all below 0.323.)
    p = [0.05] * 40 + [0.15] * 40
    y = [1] * 4 + [0] * 36 + [1] * 8 + [0] * 32
    s, _ = calibration_metrics(p, y, break_even=0.323, n_bins=2)

    assert s["n_contacted"] == 0, s["n_contacted"]
    assert np.isnan(s["precision"]) and np.isnan(s["mean_pred_contacted"])
    print("[3] empty contacted band: n_contacted=0, precision=nan, no crash  OK")


if __name__ == "__main__":
    test_perfectly_calibrated()
    test_under_prediction()
    test_empty_contacted_band()
    print("\nall calibration smoke tests passed")
