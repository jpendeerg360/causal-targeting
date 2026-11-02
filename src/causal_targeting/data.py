"""Verified download, substantive DuckDB ingestion, and label-blind sampling."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FEATURES = [f'f{i}' for i in range(12)]
CONTINUOUS = ['f0', 'f2', 'f7', 'f10']
CATEGORICAL = [x for x in FEATURES if x not in CONTINUOUS]
LABELS = ['treatment', 'conversion', 'visit', 'exposure']
SCHEMA = FEATURES + LABELS
EXPECTED_ROWS = 13_979_592
SHA256 = '2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc'
REVISION = '2424920019e49d52d72c13ac1143ec5d53af276b'
URL = f'https://huggingface.co/datasets/criteo/criteo-uplift/resolve/{REVISION}/criteo-research-uplift-v2.1.csv.gz'


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def sql(root, name, **kwargs):
    # Values are local paths or controlled configuration, not arbitrary SQL.
    escaped = {k: str(v).replace("'", "''") for k, v in kwargs.items()}
    return (root / 'sql' / name).read_text().format(**escaped)


def connect(root, config):
    con = duckdb.connect()
    con.execute('SET enable_progress_bar=false')
    con.execute(f"SET threads={int(config['threads'])}")
    con.execute(f"SET memory_limit='{config['memory_limit']}'")
    temp = root / 'artifacts' / 'duckdb_temp'
    temp.mkdir(parents=True, exist_ok=True)
    con.execute("SET temp_directory='" + str(temp).replace("'", "''") + "'")
    return con


def download(root, local_file=None):
    target = root / 'data/raw/criteo-research-uplift-v2.1.csv.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    if local_file and Path(local_file).resolve() != target.resolve():
        shutil.copyfile(local_file, target)
    if not target.exists():
        part = target.with_suffix('.partial')
        print('Downloading publisher mirror:', URL, flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, part.open('wb') as out:
            shutil.copyfileobj(response, out, length=8 * 1024**2)
        part.replace(target)
    actual = checksum(target)
    if actual != SHA256:
        raise ValueError(f'Corrected release SHA256 mismatch: {actual}; expected {SHA256}')
    with gzip.open(target, 'rt') as stream:
        header = next(csv.reader(stream))
    if header != SCHEMA:
        raise ValueError(f'Unexpected CSV schema: {header}')
    return target


def validate_frame(frame):
    """Used for tests and sampled tables; full-file checks live in SQL."""
    if list(frame.columns) != SCHEMA:
        raise ValueError('Schema must be precisely the 12 baseline features and 4 indicators')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Missing or nonfinite input')
    if not frame[LABELS].isin([0, 1]).all().all():
        raise ValueError('Indicators must be coded 0/1')
    if ((frame.exposure > frame.treatment) | (frame.conversion > frame.visit)).any():
        raise ValueError('Exposure/visit structural constraint violated')


def attach(con, root):
    glob = str(root / 'data/processed/split=*/*.parquet').replace("'", "''")
    con.execute(f"CREATE VIEW analysis AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)")


def prepare(root, config, local_file=None):
    raw = download(root, local_file)
    manifest_path = root / 'results/data_manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior['split_salt'] != config['split_salt'] or prior['seed'] != config['seed']:
            raise ValueError('Split config changed; use a fresh output directory')
        if list((root / 'data/processed').glob('split=*/*.parquet')):
            print('Verified existing input and analysis table.', flush=True)
            if not (root / 'results/tables/development_arms.csv').exists():
                development_diagnostics(root, config)
            return
    con = connect(root, config)
    con.execute('SET threads=1')
    print('Ingesting full corrected release with DuckDB...', flush=True)
    con.execute(sql(root, '01_ingest.sql', raw_path=raw))
    con.execute(f"SET threads={config['threads']}")
    qc = con.execute(sql(root, '03_quality.sql')).df().iloc[0].to_dict()
    if qc['n_rows'] != EXPECTED_ROWS or qc['unique_row_ids'] != EXPECTED_ROWS:
        raise ValueError(f'Unexpected corrected-release row count: {qc}')
    if any(qc[k] for k in qc if k not in ['n_rows', 'unique_row_ids']):
        raise ValueError(f'Quality constraints failed: {qc}')
    missing = con.execute(sql(root, '04_missingness.sql')).df()
    if missing[['missing', 'nonfinite']].to_numpy().sum():
        raise ValueError('Missing or nonfinite inputs; review before modeling')
    missing.to_csv(root / 'results/tables/missingness.csv', index=False)
    schema = con.execute('DESCRIBE raw').df().to_dict(orient='records')
    con.execute(sql(root, '02_analysis_table.sql', raw_path=raw,
                    processed_path=root / 'data/processed', salt=config['split_salt'], seed=config['seed']))
    attach(con, root)
    splits = con.execute('SELECT split,count(*) AS n FROM analysis GROUP BY split ORDER BY split').df()
    splits.to_csv(root / 'results/tables/split_counts.csv', index=False)
    write_json(manifest_path, {
        'dataset': 'Corrected Criteo Uplift v2.1', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'official_page': 'https://ailab.criteo.com/criteo-uplift-prediction-dataset/',
        'source_url': URL, 'publisher_repository': 'criteo/criteo-uplift', 'revision': REVISION,
        'sha256': SHA256, 'bytes': raw.stat().st_size, 'quality': qc,
        'original_header': SCHEMA, 'duckdb_schema': schema,
        'split_salt': config['split_salt'], 'seed': config['seed'],
        'split_counts': splits.to_dict(orient='records'),
        'license': 'CC BY-NC-SA 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'raw_git_excluded': True, 'platform': platform.platform()
    })
    con.close()
    development_diagnostics(root, config)
    print('Prepared verified data. Final-test outcomes not inspected.', flush=True)


def development_diagnostics(root, config):
    # A separate connection releases the raw table before full-development summaries.
    con = connect(root, config)
    attach(con, root)
    con.execute("CREATE VIEW development AS SELECT * FROM analysis WHERE split!='test'")
    con.execute(sql(root, '06_balance.sql')).df().to_csv(root / 'results/tables/balance.csv', index=False)
    con.execute(sql(root, '07_category_balance.sql')).df().to_csv(root / 'results/tables/category_balance.csv', index=False)
    con.execute(sql(root, '05_arm_summary.sql').replace('FROM analysis', 'FROM development')).df().to_csv(
        root / 'results/tables/development_arms.csv', index=False)
    con.close()


def load_split(root, config, split, n=None, seed=None):
    if split not in ['train', 'validation', 'test']:
        raise ValueError('Invalid split')
    con = connect(root, config)
    attach(con, root)
    order = 'sample_key, row_id'
    if seed is not None and seed != config['seed']:
        order = f"md5_number_lower(row_id::VARCHAR || ':{config['split_salt']}:sample:{int(seed)}'), row_id"
    query = f"SELECT * FROM analysis WHERE split='{split}' ORDER BY {order}"
    if n is not None:
        query += f' LIMIT {int(n)}'
    frame = con.execute(query).df()
    con.close()
    return frame


def baseline_features(frame):
    """Explicit allowlist prevents treatment/exposure/label/id leakage."""
    return frame.loc[:, FEATURES].copy()
