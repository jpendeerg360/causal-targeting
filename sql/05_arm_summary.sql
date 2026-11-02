SELECT split, treatment, count(*) AS n,
  sum(conversion) AS conversions, avg(conversion) AS conversion_rate,
  sum(visit) AS visits, avg(visit) AS visit_rate,
  sum(exposure) AS exposed, avg(exposure) AS exposure_rate
FROM analysis GROUP BY split, treatment ORDER BY split, treatment;
