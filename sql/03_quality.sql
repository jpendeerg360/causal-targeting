SELECT count(*) AS n_rows, count(DISTINCT row_id) AS unique_row_ids,
  count(*) FILTER (WHERE treatment NOT IN (0,1) OR treatment IS NULL) AS invalid_treatment,
  count(*) FILTER (WHERE conversion NOT IN (0,1) OR conversion IS NULL) AS invalid_conversion,
  count(*) FILTER (WHERE visit NOT IN (0,1) OR visit IS NULL) AS invalid_visit,
  count(*) FILTER (WHERE exposure NOT IN (0,1) OR exposure IS NULL) AS invalid_exposure,
  count(*) FILTER (WHERE exposure=1 AND treatment=0) AS exposed_controls,
  count(*) FILTER (WHERE conversion=1 AND visit=0) AS conversions_without_visits
FROM raw;
