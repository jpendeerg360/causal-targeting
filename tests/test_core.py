import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from causal_targeting.data import FEATURES, SCHEMA, baseline_features, sql, validate_frame
from causal_targeting.evaluation import (curve, dr_score, ipw_score, paired_difference,
                                         policy, value)
from causal_targeting.models import TreeEncoder
from causal_targeting.pipeline import prediction_artifact

ROOT = Path(__file__).resolve().parents[1]


def fixture_frame(n=100):
    frame = pd.DataFrame(np.zeros((n, 16)), columns=SCHEMA)
    frame['treatment'] = np.arange(n) % 2
    return frame


def test_schema_coding_and_structural_constraints():
    frame = fixture_frame()
    validate_frame(frame)
    for column, bad in [('treatment', 2), ('conversion', -1), ('visit', np.nan)]:
        invalid = frame.copy()
        invalid.loc[0, column] = bad
        with pytest.raises(ValueError):
            validate_frame(invalid)
    for column in ['exposure', 'conversion']:
        invalid = frame.copy()
        invalid.loc[0, column] = 1
        with pytest.raises(ValueError):
            validate_frame(invalid)


def test_leakage_allowlist():
    frame = fixture_frame()
    frame['row_id'] = np.arange(len(frame))
    frame['experiment_id_invented'] = 999
    assert list(baseline_features(frame)) == FEATURES
    changed = frame.copy()
    changed[['treatment', 'conversion', 'visit', 'exposure', 'row_id']] = 42
    pd.testing.assert_frame_equal(baseline_features(frame), baseline_features(changed))


def test_prediction_export_preserves_outcome_and_distinguishes_score():
    frame = fixture_frame(6)
    frame['row_id'] = np.arange(6)
    frame['tie_key'] = np.arange(6)
    frame['conversion'] = [0, 1, 0, 0, 1, 0]
    scores = {'conversion': np.full(6, .1), 'tree_uplift': np.full(6, .05)}
    result = prediction_artifact(frame, scores, (np.zeros(6), np.ones(6)))
    assert np.array_equal(result.outcome_conversion, frame.conversion)
    assert np.array_equal(result.conversion, scores['conversion'])


def test_split_is_portable_reproducible_and_label_blind(tmp_path):
    con = duckdb.connect()
    frame = fixture_frame(3000)
    frame.insert(0, 'row_id', np.arange(1, len(frame)+1))
    con.register('raw', frame)
    for suffix in ['a', 'b']:
        destination = tmp_path / suffix
        con.execute(sql(ROOT, '02_analysis_table.sql', processed_path=destination,
                        salt='unit-test', seed=42))
        if suffix == 'a':
            frame[['treatment', 'conversion', 'visit', 'exposure']] = 1
            con.unregister('raw')
            con.register('raw', frame)
    a = con.execute(f"SELECT row_id,split,sample_key,tie_key FROM read_parquet('{tmp_path}/a/*/*.parquet') ORDER BY row_id").df()
    b = con.execute(f"SELECT row_id,split,sample_key,tie_key FROM read_parquet('{tmp_path}/b/*/*.parquet') ORDER BY row_id").df()
    pd.testing.assert_frame_equal(a, b)
    assert a.row_id.is_unique and len(a) == len(frame)
    assert set(a.split) == {'train', 'validation', 'test'}
    groups = {s: set(a[a.split == s].row_id) for s in a.split.unique()}
    assert not (groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    # DuckDB md5_number_lower means the low 64 bits in its documented little-endian layout.
    assert con.execute("SELECT md5('1:unit-test:split')").fetchone()[0] == hashlib.md5(b'1:unit-test:split').hexdigest()
    con.close()


def test_train_only_category_pooling():
    train = fixture_frame(400)
    train['f1'] = np.arange(400)
    encoder = TreeEncoder().fit(train)
    frozen = dict(encoder.levels['f1'])
    new = fixture_frame(5)
    new['f1'] = 99999
    transformed = encoder.transform(new)
    assert np.all(transformed[:, 1] == 0)
    assert encoder.levels['f1'] == frozen
    assert transformed.max() <= 254


@pytest.mark.parametrize('q', [0, .05, .1, .2, .5, 1])
def test_exact_policy_capacity_and_ties(q):
    n = 101
    score = np.zeros(n)
    row_ids = np.arange(n)
    ties = row_ids[::-1]
    p = policy(score, q, ties, row_ids)
    assert p.sum() == np.floor(n*q)
    assert np.array_equal(p, policy(score, q, ties, row_ids))
    permutation = np.random.default_rng(1).permutation(n)
    shuffled = policy(score[permutation], q, ties[permutation], row_ids[permutation])
    assert set(row_ids[p == 1]) == set(row_ids[permutation][shuffled == 1])


def test_up_to_excludes_nonpositive_but_exact_keeps_them():
    score = np.array([.1, 0, -.1, -.2])
    keys = np.arange(4)
    assert policy(score, 1, keys, keys).sum() == 4
    assert policy(score, 1, keys, keys, up_to=True).sum() == 1
    with pytest.raises(ValueError):
        policy(score, 1.1, keys, keys)


def test_unequal_assignment_exact_known_effect():
    # Finite population 80% assigned: all treated convert, controls never convert.
    # Known effect is one, so a 25% stochastic policy is worth 25,000 / 100,000.
    t = np.array([1]*80+[0]*20)
    y = t.copy()
    psi = ipw_score(t, y, .8)
    assert psi.mean() == pytest.approx(1)
    assert value(np.full(100, .25), psi)['estimate'] == pytest.approx(25000)
    assert ipw_score(t, y, .5).mean() == pytest.approx(1.6)
    assert value(np.zeros(100), psi)['estimate'] == 0


def test_dr_consistency_with_correct_outcome_models_despite_wrong_e():
    t = np.array([1]*80+[0]*20)
    y = t.copy()
    score = dr_score(t, y, .5, np.zeros(100), np.ones(100))
    assert np.array_equal(score, np.ones(100))


def test_dr_reduces_to_ipw_with_zero_nuisances():
    t = np.array([1, 1, 1, 0, 0])
    y = np.array([1, 0, 1, 1, 0])
    assert np.allclose(dr_score(t, y, .7, np.zeros(5), np.zeros(5)), ipw_score(t, y, .7))


def test_binary_simulation_recovers_heterogeneous_policy_value():
    # Synthetic test only, never a reported empirical result.
    rng = np.random.default_rng(19)
    n = 400000
    x = rng.binomial(1, .4, n)
    e = np.where(x == 1, .8, .6)
    t = rng.binomial(1, e)
    m0 = .1 + .05*x
    m1 = m0 + .2*x
    y = rng.binomial(1, np.where(t == 1, m1, m0))
    target = float(np.mean(x*(m1-m0)))*100000
    for score in [ipw_score(t, y, e), dr_score(t, y, e, m0, m1)]:
        estimate = value(x, score)
        assert abs(estimate['estimate']-target) < 4*estimate['se']


def test_paired_intervals_use_covariance_and_identical_policies_cancel():
    score = np.array([0, 3, -2, 1, 0.])
    p = np.array([1, 1, 0, 0, 0.])
    result = paired_difference(p, p, score)
    assert result['estimate'] == result['se'] == result['lower'] == result['upper'] == 0
    assert paired_difference(p, np.zeros(5), score) == value(p, score)


def test_curve_units_and_qini_definition():
    n = 20
    scores = np.arange(n)
    rows, area = curve(scores, np.full(n, .2), scores, scores, [0, .5, 1])
    assert [r['estimate'] for r in rows] == pytest.approx([0, 10000, 20000])
    assert area['auuc_probability'] == pytest.approx(.1)
    assert area['qini_probability'] == pytest.approx(0)


@pytest.mark.parametrize('e', [0, 1, -.1, np.nan])
def test_invalid_propensity_rejected(e):
    with pytest.raises(ValueError):
        ipw_score([0, 1], [0, 1], e)
