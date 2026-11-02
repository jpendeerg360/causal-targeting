-- Single-threaded CSV reading preserves source order for an ingestion key.
-- row_id is NOT a released user identifier and is NEVER a model covariate.
CREATE OR REPLACE TABLE raw AS
SELECT row_number() OVER ()::BIGINT AS row_id, *
FROM read_csv('{raw_path}', header=true, parallel=false, auto_detect=false,
    columns={{'f0':'DOUBLE','f1':'DOUBLE','f2':'DOUBLE','f3':'DOUBLE',
    'f4':'DOUBLE','f5':'DOUBLE','f6':'DOUBLE','f7':'DOUBLE','f8':'DOUBLE',
    'f9':'DOUBLE','f10':'DOUBLE','f11':'DOUBLE','treatment':'TINYINT',
    'conversion':'TINYINT','visit':'TINYINT','exposure':'TINYINT'}});
