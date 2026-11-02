-- Numeric SMDs are meaningful for continuous variables. For anonymized
-- categorical tokens, also compute total variation across complete level tables.
WITH long AS (
    UNPIVOT development ON f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11
    INTO NAME feature VALUE value
), by_arm AS (
    SELECT feature, treatment, count(*) AS n, avg(value) AS mean,
      var_samp(value) AS variance, approx_quantile(value, 0.1) AS q10,
      approx_quantile(value, 0.5) AS median, approx_quantile(value, 0.9) AS q90
    FROM long GROUP BY feature, treatment
)
SELECT a.feature, a.n AS n_control, b.n AS n_treatment,
  a.mean AS mean_control, b.mean AS mean_treatment,
  (b.mean-a.mean)/nullif(sqrt((a.variance+b.variance)/2),0) AS smd,
  a.q10 AS q10_control, b.q10 AS q10_treatment,
  a.median AS median_control, b.median AS median_treatment,
  a.q90 AS q90_control, b.q90 AS q90_treatment
FROM by_arm a JOIN by_arm b USING(feature)
WHERE a.treatment=0 AND b.treatment=1 ORDER BY feature;
