WITH long AS (
    UNPIVOT development ON f1,f3,f4,f5,f6,f8,f9,f11
    INTO NAME feature VALUE value
), counts AS (
    SELECT feature, value, count(*) FILTER (WHERE treatment=0) AS n0,
      count(*) FILTER (WHERE treatment=1) AS n1
    FROM long GROUP BY feature,value
), frequencies AS (
    SELECT *, n0::DOUBLE/sum(n0) OVER(PARTITION BY feature) AS p0,
      n1::DOUBLE/sum(n1) OVER(PARTITION BY feature) AS p1
    FROM counts
)
SELECT feature, count(*) AS levels, sum(abs(p1-p0))/2 AS total_variation,
  max(abs(p1-p0)) AS maximum_level_difference
FROM frequencies GROUP BY feature ORDER BY feature;
