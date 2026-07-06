-- point-in-time feature pipeline (lifted verbatim from notebook 02, only the source changed).
-- reads from the view `_feat_points` (msno, expiry, gov_txn) so the SAME code serves
-- training (labelled phase-A cohorts) and scoring (the unlabelled base at a cutoff) — one
-- feature path, no train/serve skew. features NEVER carry the label; it is joined back
-- downstream. every aggregate filters date <= expiry => strictly point-in-time (PROJECT_RULES.md rule #3).

-- 1) commitment: the governing renewal transaction for each prediction point.
--    QUALIFY collapses same-day / same-expiry duplicates to the highest-paid row
--    (latest-/max-wins, matching the label construction in DECISIONS.md §1).
CREATE OR REPLACE TABLE feat_commitment AS
SELECT p.msno, p.expiry,
       t.is_auto_renew, t.payment_plan_days, t.actual_amount_paid, t.plan_list_price,
       (t.actual_amount_paid = 0)::int            AS is_free,    -- routing flag, NOT a scored feature
       (t.plan_list_price - t.actual_amount_paid) AS discount,
       t.payment_method_id                                       -- carried for parity; base12 drops it (§6)
FROM _feat_points p
JOIN transactions t
  ON t.msno = p.msno
 AND t.transaction_date       = strftime(p.gov_txn,'%Y%m%d')::INT
 AND t.membership_expire_date = strftime(p.expiry,'%Y%m%d')::INT
QUALIFY row_number() OVER (PARTITION BY p.msno, p.expiry ORDER BY t.actual_amount_paid DESC) = 1;

-- 2) engagement: listening behaviour in the 60d window <= expiry, split last-30 vs prior-30.
CREATE OR REPLACE TABLE feat_engagement AS
WITH pts AS (
    SELECT msno, expiry,
           strftime(expiry,    '%Y%m%d')::INT AS e_int,
           strftime(expiry-30, '%Y%m%d')::INT AS e_30,
           strftime(expiry-60, '%Y%m%d')::INT AS e_60
    FROM _feat_points),
j AS (
    SELECT p.msno, p.expiry, p.e_int, p.e_30, l.date,
           l.num_25, l.num_50, l.num_75, l.num_985, l.num_100, l.num_unq, l.total_secs
    FROM pts p JOIN user_logs l
      ON l.msno = p.msno AND l.date > p.e_60 AND l.date <= p.e_int),
agg AS (
    SELECT msno, expiry,
           date_diff('day', strptime(max(date)::VARCHAR,'%Y%m%d')::DATE, expiry) AS recency_days,
           count(DISTINCT CASE WHEN date >  e_30 THEN date END) AS active_days_30,
           count(DISTINCT CASE WHEN date <= e_30 THEN date END) AS active_days_prior,
           sum(CASE WHEN date > e_30 THEN total_secs ELSE 0 END) AS secs_30,
           sum(CASE WHEN date > e_30 THEN num_unq    ELSE 0 END) AS unq_30,
           sum(CASE WHEN date > e_30 THEN num_100    ELSE 0 END) AS completed_30,
           sum(CASE WHEN date > e_30 THEN num_25+num_50+num_75+num_985+num_100 ELSE 0 END) AS plays_30
    FROM j GROUP BY msno, expiry)
SELECT *,
       completed_30   / nullif(plays_30, 0)          AS completion_ratio,
       active_days_30 / nullif(active_days_prior, 0) AS activity_trend
FROM agg;

-- 3) assemble: commitment LEFT JOIN engagement; silent customers (no logs in window) get
--    has_activity_60d=0 and neutral defaults (recency capped at the 60d window edge).
CREATE OR REPLACE TABLE model_data AS
SELECT c.*,
       (e.msno IS NOT NULL)::int       AS has_activity_60d,
       coalesce(e.recency_days, 60)    AS recency_days,
       coalesce(e.active_days_30, 0)   AS active_days_30,
       coalesce(e.secs_30, 0)          AS secs_30,
       coalesce(e.unq_30, 0)           AS unq_30,
       coalesce(e.completion_ratio, 0) AS completion_ratio,
       coalesce(e.activity_trend, 0)   AS activity_trend
FROM feat_commitment c
LEFT JOIN feat_engagement e USING (msno, expiry);
