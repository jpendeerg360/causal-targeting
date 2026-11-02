-- UNPIVOT INCLUDE NULLS avoids silently dropping missing values.
WITH long AS (
  SELECT variable,value FROM raw
  UNPIVOT INCLUDE NULLS (value FOR variable IN
    (f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure))
)
SELECT variable, count(*) AS n, count(*) FILTER (WHERE value IS NULL) AS missing,
  count(*) FILTER (WHERE NOT isfinite(value)) AS nonfinite,
  min(value) AS minimum, max(value) AS maximum, count(DISTINCT value) AS distinct_values
FROM long GROUP BY variable ORDER BY variable;
