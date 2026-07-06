"""smoke: score_drift on synthetic fixtures with HAND-COMPUTED expected values.
no db, no mlflow — exercises only the drift math. run: python -m tests.smoke_score_drift
"""
import numpy as np

from monitoring.score_drift import score_drift


def _row(tab, name):
    return tab.loc[name, "ref"], tab.loc[name, "cur"], tab.loc[name, "delta"]


def test_identical_no_drift():
    p = [0.01] * 50 + [0.05] * 20 + [0.10] * 10 + [0.50] * 20
    s, tab = score_drift(p, p, break_even=0.323)
    assert (tab["delta"] == 0).all(), tab
    print("[1] identical ref==cur: every delta 0  OK")


def test_tail_thins_under_contacting():
    # mass leaves the high-P tail -> fewer clear break-even (the §3 under-contact signal).
    ref = [0.01] * 50 + [0.05] * 20 + [0.10] * 10 + [0.50] * 20   # 20/100 >= 0.323
    cur = [0.01] * 60 + [0.05] * 20 + [0.10] * 10 + [0.50] * 10   # 10/100 >= 0.323
    s, tab = score_drift(ref, cur, break_even=0.323)
    cr_r, cr_c, _ = _row(tab, "contact_rate")
    mc_r, mc_c, _ = _row(tab, "mean_P|contacted")
    q_r, q_c, _ = _row(tab, "q90")
    assert abs(cr_r - 0.20) < 1e-9 and abs(cr_c - 0.10) < 1e-9, (cr_r, cr_c)
    assert abs(mc_r - 0.50) < 1e-9 and abs(mc_c - 0.50) < 1e-9   # both contact only the 0.50s
    assert q_c < q_r                                             # tail thinned
    print("[2] tail thins: contact_rate 0.20->0.10, mean_P|contacted 0.50 (flat), q90 down  OK")


def test_composition_shift_at_constant_budget():
    # the case contact_rate ALONE misses: same number contacted, higher-risk mix.
    ref = [0.01] * 90 + [0.50] * 10     # contact_rate 0.10, mean_P|contacted 0.50
    cur = [0.01] * 90 + [0.80] * 10     # contact_rate 0.10, mean_P|contacted 0.80
    s, tab = score_drift(ref, cur, break_even=0.323)
    cr_r, cr_c, _ = _row(tab, "contact_rate")
    mc_r, mc_c, _ = _row(tab, "mean_P|contacted")
    assert abs(cr_r - 0.10) < 1e-9 and abs(cr_c - 0.10) < 1e-9   # budget UNCHANGED
    assert abs(mc_r - 0.50) < 1e-9 and abs(mc_c - 0.80) < 1e-9   # composition moved
    print("[3] composition shift at flat budget: contact_rate 0.10 both, mean_P|contacted 0.50->0.80  OK")


def test_empty_contacted_band():
    p = [0.05] * 40 + [0.15] * 40   # nothing clears break-even
    s, tab = score_drift(p, p, break_even=0.323)
    cr_r, _, _ = _row(tab, "contact_rate")
    mc_r, _, _ = _row(tab, "mean_P|contacted")
    assert cr_r == 0.0
    assert np.isnan(mc_r)
    print("[4] empty contacted band: contact_rate 0, mean_P|contacted nan, no crash  OK")


if __name__ == "__main__":
    test_identical_no_drift()
    test_tail_thins_under_contacting()
    test_composition_shift_at_constant_budget()
    test_empty_contacted_band()
    print("\nall score-drift smoke tests passed")
