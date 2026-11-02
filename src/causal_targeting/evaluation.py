"""Assignment-based estimators; policy contrasts are paired on the same rows."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

Z975 = 1.959963984540054
SCALE = 100_000


def _inputs(t, y, e):
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    e = np.broadcast_to(np.asarray(e, dtype=float), t.shape)
    if t.ndim != 1 or y.shape != t.shape or not np.isfinite(e).all():
        raise ValueError('Inputs must be finite aligned vectors')
    if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Treatment/outcome must be binary, with no missing values')
    if ((e <= 0) | (e >= 1)).any():
        raise ValueError('Propensities must lie strictly between zero and one')
    return t, y, e


def ipw_score(t, y, e):
    t, y, e = _inputs(t, y, e)
    return t*y/e - (1-t)*y/(1-e)


def dr_score(t, y, e, m0, m1):
    t, y, e = _inputs(t, y, e)
    m0, m1 = np.asarray(m0), np.asarray(m1)
    if m0.shape != y.shape or m1.shape != y.shape or not np.isfinite([m0, m1]).all():
        raise ValueError('Nuisance predictions must be finite aligned vectors')
    if ((m0 < 0) | (m0 > 1) | (m1 < 0) | (m1 > 1)).any():
        raise ValueError('Outcome predictions must be probabilities')
    return m1-m0 + t*(y-m1)/e - (1-t)*(y-m0)/(1-e)


def ranking(score, tie_key, row_id):
    score = np.asarray(score)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError('Scores must be finite')
    if len(score) != len(tie_key) or len(score) != len(row_id):
        raise ValueError('Ranking keys are not aligned')
    return np.lexsort((np.asarray(row_id), np.asarray(tie_key), -score))


def policy(score, fraction, tie_key, row_id, up_to=False):
    if not 0 <= fraction <= 1:
        raise ValueError('Budget must be in [0,1]')
    n = len(score)
    k = int(np.floor(fraction*n + 1e-10))
    selected = ranking(score, tie_key, row_id)[:k]
    if up_to:
        selected = selected[np.asarray(score)[selected] > 0]
    pi = np.zeros(n, dtype=float)
    pi[selected] = 1
    return pi


def mean_ci(contributions):
    a = np.asarray(contributions, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('At least two finite observations required')
    estimate = float(a.mean())
    se = float(a.std(ddof=1)/np.sqrt(len(a)))
    return {'estimate': estimate*SCALE, 'se': se*SCALE,
            'lower': (estimate-Z975*se)*SCALE, 'upper': (estimate+Z975*se)*SCALE}


def value(pi, score):
    pi, score = np.asarray(pi), np.asarray(score)
    if pi.shape != score.shape or ((pi < 0) | (pi > 1)).any():
        raise ValueError('Invalid policy weights')
    return mean_ci(pi*score)


def paired_difference(pi_a, pi_b, score):
    return mean_ci((np.asarray(pi_a)-np.asarray(pi_b))*np.asarray(score))


def curve(score, pseudo_outcome, tie_key, row_id, fractions):
    order = ranking(score, tie_key, row_id)
    a = np.asarray(pseudo_outcome)[order]
    sums, squares = np.r_[0., np.cumsum(a)], np.r_[0., np.cumsum(a*a)]
    n = len(a)
    result = []
    for q in fractions:
        k = min(n, int(np.floor(q*n+1e-10)))
        mean = sums[k]/n
        variance = max(0, (squares[k] - n*mean*mean)/(n-1))
        se = np.sqrt(variance/n)
        result.append({'budget': float(q), 'actual_fraction': k/n,
                       'estimate': mean*SCALE, 'se': se*SCALE,
                       'lower': (mean-Z975*se)*SCALE, 'upper': (mean+Z975*se)*SCALE})
    auuc = (sums[1:].sum() - 0.5*sums[-1])/n**2
    qini = auuc - 0.5*sums[-1]/n
    return result, {'auuc_probability': float(auuc), 'qini_probability': float(qini),
                    'auuc_per_100k': float(auuc*SCALE), 'qini_per_100k': float(qini*SCALE)}


def prediction_metrics(y, prediction):
    return {'roc_auc': float(roc_auc_score(y, prediction)),
            'average_precision': float(average_precision_score(y, prediction)),
            'log_loss': float(log_loss(y, prediction, labels=[0, 1])),
            'brier': float(brier_score_loss(y, prediction)), 'n': len(y),
            'positives': int(np.sum(y)), 'prevalence': float(np.mean(y))}


def difference_in_means(t, y):
    t, y, _ = _inputs(t, y, .85)
    a, b = y[t == 1], y[t == 0]
    estimate = a.mean()-b.mean()
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    return {'estimate': float(estimate*SCALE), 'se': float(se*SCALE),
            'lower': float((estimate-Z975*se)*SCALE), 'upper': float((estimate+Z975*se)*SCALE)}
