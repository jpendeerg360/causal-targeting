"""Cross-artifact checks and an independent SQL audit of the primary estimate."""
import json
import math
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from .data import EXPECTED_ROWS, FEATURES, SHA256, checksum, write_json


def verify(root, config):
    passed, skipped = [], []
    def check(condition, message):
        if not condition:
            raise AssertionError(message)
        passed.append(message)
    read = lambda name: pd.read_csv(root / f'results/tables/{name}.csv')
    manifest = json.loads((root / 'results/data_manifest.json').read_text())
    freeze = json.loads((root / 'results/freeze.json').read_text())
    summary = json.loads((root / 'results/summary.json').read_text())
    evaluation = json.loads((root / 'results/evaluation.json').read_text())
    check(manifest['sha256'] == SHA256 and manifest['quality']['n_rows'] == EXPECTED_ROWS,
          'Publisher fingerprint and corrected release row count agree')
    splits = read('split_counts').set_index('split').n
    check(splits.sum() == EXPECTED_ROWS and splits['test'] == summary['test_n'] == evaluation['n_test'],
          'Split, evaluation and report sample sizes agree')
    check(freeze['features'] == FEATURES and freeze['test_outcomes_used'] is False,
          'Frozen feature allowlist and holdout declaration are intact')
    check(freeze['config_sha256'] == checksum(root / 'config/analysis.json') and
          freeze['plan_sha256'] == checksum(root / 'docs/research_plan.md'), 'Plan and configuration hashes match freeze')
    check(evaluation['freeze_sha256'] == checksum(root / 'results/freeze.json'),
          'Evaluation references the frozen model selection')
    missing = read('missingness')
    check(missing['n'].eq(EXPECTED_ROWS).all() and missing[['missing', 'nonfinite']].sum().sum() == 0,
          'Full-file missingness checks reconcile')
    arms = read('all_arms')
    check(arms.n.sum() == EXPECTED_ROWS and arms.query("split == 'test'").conversions.sum() == evaluation['test_conversions'],
          'SQL outcome counts reconcile with evaluation')
    values, diffs, curves = read('policy_values'), read('paired_differences'), read('policy_curves')
    check(np.isfinite(values[['estimate', 'se', 'lower', 'upper']].to_numpy()).all(), 'Policy estimates are finite')
    check((values.lower <= values.estimate).all() and (values.estimate <= values.upper).all() and (values.se >= 0).all(),
          'All policy confidence intervals are ordered')
    exact = values.query("policy != 'none'")
    expected_fractions = np.floor(exact.budget*evaluation['n_test']+1e-10)/evaluation['n_test']
    check(np.allclose(exact.actual_fraction, expected_fractions, atol=1e-14), 'Every exact policy respects its budget')
    check((values.query("policy == 'none'")[['actual_fraction', 'estimate', 'se', 'lower', 'upper']] == 0).all().all(),
          'No-targeting is exactly zero')
    for estimator in ['IPW', 'DR']:
        endpoints = curves.query('estimator == @estimator and budget == 1')
        check(np.ptp(endpoints.estimate) < 1e-7, f'{estimator}: full-budget policies coincide')
        full = float(endpoints.estimate.iloc[0])
        random = values.query("outcome == 'conversion' and estimator == @estimator and policy == 'random'")
        check(np.allclose(random.estimate, random.actual_fraction*full, atol=1e-7),
              f'{estimator}: random value equals audience fraction times full value')
    for _, row in diffs.iterrows():
        table = values[(values.outcome == row.outcome) & (values.estimator == row.estimator) &
                       (values.budget == row.budget)].set_index('policy')
        if not np.isclose(row.estimate, table.loc[row.policy_a, 'estimate']-table.loc[row.policy_b, 'estimate'], atol=1e-9):
            raise AssertionError('Paired point estimate does not equal policy difference')
    passed.append('All paired point differences reconcile')
    q = config['primary_budget']
    primary = diffs.query("budget == @q and estimator == 'IPW' and policy_a == 'tree_uplift' and policy_b == 'conversion'").iloc[0]
    check(all(np.isclose(primary[k], summary['primary_contrast'][k], atol=1e-10) for k in ['estimate', 'se', 'lower', 'upper']),
          'Primary summary equals machine-readable paired result')
    token = f"{primary.estimate:.1f} [{primary.lower:.1f}, {primary.upper:.1f}]"
    check(all(token in (root / filename).read_text() for filename in ['README.md', 'docs/research_report.md', 'docs/resume_bullets.md']),
          'README, report and resume numbers agree with saved primary result')
    for name in ['01_balance_and_effect', '02_policy_value_curves', '03_primary_budget', '04_prediction_vs_targeting', '05_development_robustness']:
        check(all((root / f'results/figures/{name}.{ext}').stat().st_size > 1000 for ext in ['png', 'svg', 'pdf']),
              f'{name}: PNG, SVG and PDF outputs exist')
    raw = root / 'data/raw/criteo-research-uplift-v2.1.csv.gz'
    if raw.exists():
        check(checksum(raw) == SHA256, 'Local raw file checksum matches publisher metadata')
    else:
        skipped.append('Raw checksum recomputation: data intentionally excluded from source distribution')
    predictions = root / 'artifacts/test_predictions.parquet'
    if predictions.exists():
        con = duckdb.connect()
        con.execute("SET threads=4; SET memory_limit='4GB'; SET enable_progress_bar=false")
        con.execute("CREATE VIEW predictions AS SELECT * FROM read_parquet('" + str(predictions).replace("'", "''") + "')")
        n = con.execute('SELECT count(*) FROM predictions').fetchone()[0]
        k = math.floor(q*n)
        e = config['propensity']
        # Prediction score name `conversion` must not shadow the outcome column.
        # The saved file stores the observed outcome as outcome_conversion.
        independent = con.execute(f'''
            WITH ranked AS (
              SELECT *, row_number() OVER(ORDER BY tree_uplift DESC,tie_key,row_id) AS tree_rank,
                row_number() OVER(ORDER BY conversion DESC,tie_key,row_id) AS conversion_rank
              FROM predictions
            ), contributions AS (
              SELECT ((tree_rank <= {k})::INTEGER-(conversion_rank <= {k})::INTEGER)*
                (treatment*outcome_conversion/{e}-(1-treatment)*outcome_conversion/(1-{e})) AS d
              FROM ranked
            ) SELECT avg(d)*100000, stddev_samp(d)/sqrt(count(*))*100000 FROM contributions
        ''').fetchone()
        check(n == evaluation['n_test'] and np.allclose(independent, [primary.estimate, primary.se], atol=1e-8),
              'Independent SQL audit reproduces primary IPW point estimate and paired SE from row predictions')
        con.close()
    else:
        skipped.append('Row-level SQL audit: local predictions intentionally excluded from source distribution')
    result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'checks_passed': passed,
              'checks_skipped': skipped, 'status': 'PASS'}
    write_json(root / 'results/verification.json', result)
    print(f'PASS: {len(passed)} artifact checks; {len(skipped)} optional local-data checks skipped.', flush=True)
