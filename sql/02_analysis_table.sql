-- All split/sample randomness uses identifiers generated from source position,
-- independent of treatment, exposure, covariates and outcomes. MD5 is portable.
COPY (
    WITH keyed AS (
        SELECT *,
          md5_number_lower(row_id::VARCHAR || ':{salt}:split') % 100 AS split_bucket,
          md5_number_lower(row_id::VARCHAR || ':{salt}:sample:{seed}') AS sample_key,
          md5_number_lower(row_id::VARCHAR || ':{salt}:tie') AS tie_key
        FROM raw
    )
    SELECT * EXCLUDE(split_bucket),
      CASE WHEN split_bucket < 60 THEN 'train'
           WHEN split_bucket < 70 THEN 'validation' ELSE 'test' END AS split
    FROM keyed
) TO '{processed_path}' (FORMAT PARQUET, PARTITION_BY(split), COMPRESSION ZSTD,
                         ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE);
