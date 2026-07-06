"""build prediction points for a scoring cutoff from phase A's validated `pred_points` table.

the scorer does NOT re-derive the operative-expiry / genuine-paid logic — that lives in notebook 01
(validated row-level to kappa 0.94). it selects the cutoff month's points from `pred_points` and
feeds them through the SAME features.py path the training model used -> train/serve parity on
features. point construction is upstream and shared; only the month filter differs.
"""


def latest_cutoff(con):
    return con.execute("SELECT max(cohort_month) FROM pred_points").fetchone()[0]


def available_cutoffs(con):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT cohort_month FROM pred_points ORDER BY 1").fetchall()]


def points_sql(cutoff):
    """SQL selecting (msno, expiry, gov_txn) for one monthly cohort.
    cutoff: a date / 'YYYY-MM-01' string (first of the month, as stored in cohort_month)."""
    return (f"SELECT msno, expiry, gov_txn FROM pred_points "
            f"WHERE cohort_month = DATE '{cutoff}'")
