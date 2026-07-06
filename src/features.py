"""build the feature table (model_data) from a set of prediction points.

the SAME entry point serves both sides:
  - training : points = the labelled phase-A cohorts (cohorts_s) -> default below
  - scoring  : points = the unlabelled base at a cutoff (built in src/score, phase E3)
features carry NO label — it is joined back only where needed (training / verification),
so there is exactly one feature code path and no train/serve skew.
"""
from . import config, db


def _load_sql(path):
    # strip -- line comments first, else a ';' inside a comment would split a statement mid-way.
    # (features.sql has no '--' inside string literals, so line-level stripping is safe here.)
    return "\n".join(line.split("--", 1)[0] for line in path.read_text().splitlines())


def build_features(con, points_sql="SELECT msno, expiry, gov_txn FROM cohorts_s", out_table="model_data"):
    """create feat_commitment, feat_engagement, and the assembled feature table in `con`.
    points_sql must yield (msno, expiry, gov_txn); default is the phase-A training cohorts.
    out_table routes the final assembly elsewhere (e.g. score_data) so a scoring run never
    clobbers the training model_data. feat_commitment/feat_engagement are ephemeral scratch."""
    # _feat_points: internal scratch view the SQL reads from. leading underscore so it never
    # collides with the phase-A `pred_points` TABLE already in the db.
    con.execute(f"CREATE OR REPLACE VIEW _feat_points AS {points_sql}")
    code = _load_sql(config.SQL_DIR / "features.sql")
    if out_table != "model_data":
        code = code.replace("CREATE OR REPLACE TABLE model_data", f"CREATE OR REPLACE TABLE {out_table}")
    for stmt in (s for s in code.split(";") if s.strip()):
        con.execute(stmt)
    return con


def verify(con):
    """reproduce the notebook's E1 sanity checks. expected against the real db (notebook 02):
      - cohorts_s splits : train 300000/0.1192, val 100000/0.1368, test 100000/0.0595
      - model_data       : n=500000, silent (has_activity_60d=0)=106651
      - model_data base rates per split must equal the cohorts_s rates (label join is lossless)
    """
    q = lambda s: con.execute(s).df()

    print("[1] input cohorts (split sizes + base rate) — should match the phase-A lock")
    print(q("""SELECT split, count(*) AS n, count(DISTINCT msno) AS users,
                      round(avg(is_churn::int),4) AS churn_rate
               FROM cohorts_s GROUP BY split ORDER BY min(cohort_month)"""))

    print("\n[2] model_data row count + silent share (engagement miss -> has_activity_60d=0)")
    print(q("""SELECT count(*) AS n,
                      count(*) FILTER (WHERE has_activity_60d=0) AS silent
               FROM model_data"""))

    print("\n[3] per-split base rate after joining the label back (must equal [1])")
    print(q("""SELECT c.split, count(*) AS n, round(avg(c.is_churn::int),4) AS churn_rate
               FROM model_data m JOIN cohorts_s c USING (msno, expiry)
               GROUP BY c.split ORDER BY min(c.cohort_month)"""))


if __name__ == "__main__":
    con = db.connect()
    build_features(con)
    verify(con)
