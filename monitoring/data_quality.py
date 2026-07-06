"""phase F tier 0 — the data-quality / schema gate (monitor #0 in the §9 hierarchy).

runs BEFORE any statistic, because the common, catastrophic failure is a broken pipeline (a join
change, a CSV-view path that didn't resolve, a units flip) — more frequent than subtle drift, and
cheaper to catch. the subtlety this gate is built around: features.sql coalesces missing engagement
to neutral defaults (has_activity_60d=0, recency=60, ...), so a null-rate check on the assembled
table is BLIND to an engagement-join break — the coalesce repaints the dropped rows as dormant
customers. so the gate checks STRUCTURE (row-count parity, silent-share) and impossible values, not
just nulls.

hard-fail (stops the pipeline): schema, dtype, un-coalesced nulls, impossible values, row-count
parity. warn (look closer, does not stop): silent-share spike, negative-discount surge — both have
legitimate-variation explanations, so a gross deviation is the signal, not any deviation.

scope: catches impossible / structural faults. a units flip that stays in-range but wrong-magnitude
is NOT tier 0's job — it shifts the score distribution and is caught by step 2.

    python -m monitoring.data_quality               # gate the latest cutoff cohort
    python -m monitoring.data_quality 2017-01-01
"""
from collections import namedtuple

import pandas as pd

from src import config, db, data, features

Check = namedtuple("Check", "name status detail")   # status in {pass, warn, fail}

BINARY = ["is_auto_renew", "has_activity_60d"]                       # ∈ {0,1}
NONNEG = ["payment_plan_days", "actual_amount_paid", "plan_list_price",
          "active_days_30", "secs_30", "unq_30", "activity_trend"]   # counts / amounts / ratio ≥ 0
# signed by design: discount = plan_list_price - actual_amount_paid CAN be < 0 (paid > list, a
# promo/tax edge) — so it is a warn on surge, never a hard fail. completion_ratio ∈ [0,1]
# (completed ⊆ plays); recency_days ∈ [0,60] (structurally capped at the 60-day window).


def quality_checks(df, n_points, silent_baseline=0.213, silent_tol=2.0):
    """pure: a built feature table + the count of prediction points fed in -> list[Check].
    no db / no model, so it is unit-testable by handing it deliberately corrupted DataFrames."""
    out = []
    def add(name, ok, detail, soft=False):
        out.append(Check(name, "pass" if ok else ("warn" if soft else "fail"), detail))

    missing = [c for c in config.BASE12 + ["is_free"] if c not in df.columns]
    add("schema/columns", not missing, f"missing {missing}" if missing else f"{len(config.BASE12)}+is_free present")
    present = [c for c in config.BASE12 if c in df.columns]

    nonnum = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    add("schema/numeric", not nonnum, f"non-numeric {nonnum}" if nonnum else "all base12 numeric")
    # value checks below operate only on numeric columns: a non-numeric column is already a hard
    # fail above, and `str < 0` would otherwise crash the gate instead of reporting it.
    num = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]

    # un-coalesced nulls: features.sql step 3 coalesces engagement, so a remaining null means a
    # column slipped past coalesce (a code-path change) — NOT a join break (those hide as silent).
    nullc = {c: int(df[c].isna().sum()) for c in present if df[c].isna().any()}
    add("values/non_null", not nullc, f"nulls {nullc}" if nullc else "no nulls in base12")

    badbin = {c: sorted({int(v) for v in df[c].dropna().unique()} - {0, 1})
              for c in BINARY if c in num and {int(v) for v in df[c].dropna().unique()} - {0, 1}}
    add("values/binary", not badbin, f"non-{{0,1}} {badbin}" if badbin else "flags ∈ {0,1}")

    neg = {c: int((df[c] < 0).sum()) for c in NONNEG if c in num and (df[c] < 0).any()}
    add("values/non_negative", not neg, f"negative {neg}" if neg else "counts/amounts ≥ 0")

    bad01 = "completion_ratio" in num and \
        bool(((df["completion_ratio"] < 0) | (df["completion_ratio"] > 1)).any())
    add("values/completion∈[0,1]", not bad01, "out of [0,1]" if bad01 else "in [0,1]")
    badrec = "recency_days" in num and \
        bool(((df["recency_days"] < 0) | (df["recency_days"] > 60)).any())
    add("values/recency∈[0,60]", not badrec, "out of [0,60]" if badrec else "in [0,60]")

    # row-count parity: every prediction point -> exactly one feature row. a shortfall = the
    # commitment INNER JOIN dropped points (date-format / CSV-view break); an excess = the
    # (msno,expiry) dedup failed. either is a hard structural failure the coalesce can't mask.
    add("structure/row_parity", len(df) == n_points, f"rows_out {len(df)} vs points_in {n_points}")

    # silent-share (WARN): has_activity_60d=0 is the engagement LEFT JOIN missing. the coalesce
    # masks a join break as dormancy, so a gross SPIKE (not subtle drift — that's legit cohort
    # variation) is the only label-free tell. soft + wide on purpose.
    if "has_activity_60d" in df.columns:
        silent = float((df["has_activity_60d"] == 0).mean())
        ceiling = silent_baseline * silent_tol
        add("structure/silent_share", silent <= ceiling,
            f"{silent:.3f} (baseline ~{silent_baseline:.3f}; warn if >{ceiling:.2f})", soft=True)

    # discount sign (WARN, informational): paid > list is a documented rare edge; flag a surge.
    if "discount" in num and (df["discount"] < 0).any():
        share = float((df["discount"] < 0).mean())
        add("values/discount_negative", share < 0.05,
            f"{share:.3f} share paid>list (rare edge; warn if >0.05)", soft=True)

    return out


def gate(cutoff=None):
    con = db.connect()
    if cutoff is None:
        cutoff = data.latest_cutoff(con)
    cutoff = str(pd.Timestamp(cutoff).date())
    features.build_features(con, data.points_sql(cutoff), out_table="score_data")
    n_points = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT msno, expiry FROM _feat_points)").fetchone()[0]
    df = con.execute("SELECT * FROM score_data").df()

    checks = quality_checks(df, n_points)
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]

    print(f"data-quality gate — cutoff {cutoff}, score_data ({len(df)} rows, {n_points} points)")
    for c in checks:
        mark = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}[c.status]
        print(f"  [{mark}] {c.name:24} {c.detail}")
    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    print(f"\ngate: {verdict}  ({len(fails)} fail, {len(warns)} warn)")
    if fails:
        raise SystemExit(1)   # non-zero so the gate can stop the pipeline before scoring
    return checks


if __name__ == "__main__":
    import sys
    gate(sys.argv[1] if len(sys.argv) > 1 else None)
