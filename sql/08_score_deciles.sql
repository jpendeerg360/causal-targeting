-- Treatment-specific rates are diagnostic only. Effect is an assignment contrast.
WITH ranked AS (
    SELECT *, ntile(10) OVER(ORDER BY score DESC,tie_key,row_id) AS score_decile
    FROM predictions
), agg AS (
    SELECT policy, score_decile, count(*) AS n,
      count(*) FILTER(WHERE treatment=1) AS n1,
      count(*) FILTER(WHERE treatment=0) AS n0,
      avg(score) AS average_score, avg(m1) AS predicted_treated_rate,
      avg(conversion) FILTER(WHERE treatment=1) AS treated_rate,
      avg(conversion) FILTER(WHERE treatment=0) AS control_rate,
      avg(conversion*treatment/{propensity} - conversion*(1-treatment)/(1-{propensity})) AS ipw_effect
    FROM ranked GROUP BY policy,score_decile
)
SELECT *, treated_rate-control_rate AS difference_in_means FROM agg ORDER BY score_decile;
