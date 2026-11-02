"""Development, model freeze, and one final held-out evaluation."""
from __future__ import annotations

import gc
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from .data import (FEATURES, attach, checksum, connect, load_split, sql, write_json)
from .evaluation import (curve, difference_in_means, dr_score, ipw_score, mean_ci,
                         paired_difference, policy, prediction_metrics, value)
from .models import LogisticInteractions, TLearner, fit_assignment_model


def log(message):
    print(datetime.now(timezone.utc).strftime('%H:%M:%S'), message, flush=True)


def predict_batches(model, frame, batch_size=200_000):
    m0, m1 = [], []
    for start in range(0, len(frame), batch_size):
        a, b = model.predict(frame.iloc[start:start+batch_size])
        m0.append(a)
        m1.append(b)
    return np.concatenate(m0), np.concatenate(m1)


def prediction_artifact(frame, scores, nuisance):
    saved = frame[['row_id', 'tie_key', 'treatment', 'conversion', 'visit']].copy()
    saved = saved.rename(columns={'conversion': 'outcome_conversion'})
    for name, score in scores.items():
        saved[name] = score
    saved['nuisance_m0'], saved['nuisance_m1'] = nuisance
    return saved


def validation_row(model, validation, config, **metadata):
    m0, m1 = predict_batches(model, validation)
    psi = ipw_score(validation.treatment, validation.conversion, config['propensity'])
    p = policy(m1-m0, config['primary_budget'], validation.tie_key, validation.row_id)
    conversion_pi = policy(m1, config['primary_budget'], validation.tie_key, validation.row_id)
    treated = validation.treatment.to_numpy() == 1
    return {**metadata, 'uplift_value': value(p, psi)['estimate'],
            'uplift_se': value(p, psi)['se'],
            'conversion_value': value(conversion_pi, psi)['estimate'],
            'uplift_minus_conversion': paired_difference(p, conversion_pi, psi)['estimate'],
            'treated_log_loss': float(log_loss(validation.conversion.to_numpy()[treated], m1[treated])),
            'n_validation': len(validation)}


def develop(root, config):
    if (root / 'results/freeze.json').exists():
        raise RuntimeError('Models already frozen. Use evaluate/report or a separate fresh copy to retrain.')
    started = time.monotonic()
    validation = load_split(root, config, 'validation', config['validation_n'])
    train = load_split(root, config, 'train', config['tune_n'])
    assert not np.intersect1d(train.row_id, validation.row_id).size
    rows, stability = [], []
    with threadpool_limits(limits=config['threads']):
        log(f'Pilot T-learner: {config["pilot_n"]:,} train rows')
        pilot = TLearner(config['tree_candidates'][0], config['seed']).fit(train.iloc[:config['pilot_n']])
        pilot_result = validation_row(pilot, validation, config, family='tree',
                                     train_n=config['pilot_n'], seed=config['seed'], candidate=0,
                                     check='pilot')
        stability.append(pilot_result)
        del pilot
        tree_models = []
        for i, params in enumerate(config['tree_candidates']):
            log(f'Tuning tree candidate {i+1}/{len(config["tree_candidates"])}')
            model = TLearner(params, config['seed']).fit(train)
            rows.append(validation_row(model, validation, config, family='tree', candidate=i))
            tree_models.append(model)
        tree_rows = [r for r in rows if r['family'] == 'tree']
        chosen_tree = max(tree_rows, key=lambda r: r['uplift_value'])['candidate']
        chosen_conversion = min(tree_rows, key=lambda r: r['treated_log_loss'])['candidate']
        logistic_models = []
        for i, c in enumerate(config['logistic_c']):
            log(f'Tuning logistic interactions C={c}')
            model = LogisticInteractions(c).fit(train)
            rows.append(validation_row(model, validation, config, family='logistic', candidate=i,
                                       c=c, solver_iterations=model.n_iter))
            logistic_models.append(model)
        chosen_logistic = max([r for r in rows if r['family'] == 'logistic'],
                              key=lambda r: r['uplift_value'])['candidate']
        selection = {'tree_candidate': chosen_tree, 'conversion_candidate': chosen_conversion,
                     'logistic_candidate': chosen_logistic,
                     'tree_params': config['tree_candidates'][chosen_tree],
                     'conversion_params': config['tree_candidates'][chosen_conversion],
                     'logistic_c': config['logistic_c'][chosen_logistic]}
        pd.DataFrame(rows).to_csv(root / 'results/tables/tuning.csv', index=False)
        write_json(root / 'results/model_selection.json', selection)
        log('Selected settings using validation only: ' + str(selection))
        base = validation_row(tree_models[chosen_tree], validation, config, family='tree',
                              train_n=len(train), seed=config['seed'], candidate=chosen_tree,
                              check='size_and_seed')
        stability.append(base)
        del tree_models, logistic_models, model
        gc.collect()
        # Same settings across sizes, including a second pilot if tuning changed them.
        if chosen_tree != 0:
            model = TLearner(selection['tree_params'], config['seed']).fit(train.iloc[:config['pilot_n']])
            stability.append(validation_row(model, validation, config, family='tree',
                                            train_n=config['pilot_n'], seed=config['seed'],
                                            candidate=chosen_tree, check='size_and_seed'))
            del model
        for seed in config['stability_seeds'][1:]:
            log(f'Training-subset stability seed {seed}')
            other = load_split(root, config, 'train', config['tune_n'], seed=seed)
            model = TLearner(selection['tree_params'], seed).fit(other)
            stability.append(validation_row(model, validation, config, family='tree',
                                            train_n=len(other), seed=seed, candidate=chosen_tree,
                                            check='size_and_seed'))
            del other, model
        log('Fitting assignment-predictability diagnostic on development data')
        assignment_encoder, assignment_model = fit_assignment_model(train, config['seed'])
        assignment_p = assignment_model.predict_proba(assignment_encoder.transform(validation))[:, 1]
        assignment_diagnostic = {
            'validation_roc_auc': float(roc_auc_score(validation.treatment, assignment_p)),
            'validation_log_loss': float(log_loss(validation.treatment, assignment_p)),
            'constant_log_loss': float(log_loss(validation.treatment,
                                                 np.full(len(validation), config['propensity']))),
            'p01': float(np.quantile(assignment_p, .01)), 'p99': float(np.quantile(assignment_p, .99)),
            'interpretation': 'A diagnostic, not a proof of exchangeability or a design reconstruction.'}
        write_json(root / 'results/assignment_diagnostic.json', assignment_diagnostic)
        joblib.dump((assignment_encoder, assignment_model), root / 'artifacts/assignment.joblib')
        del train
        gc.collect()
        train = load_split(root, config, 'train', config['train_n'])
        log(f'Final T-learner fit: {len(train):,} train rows')
        tree = TLearner(selection['tree_params'], config['seed']).fit(train)
        stability.append(validation_row(tree, validation, config, family='tree', train_n=len(train),
                                        seed=config['seed'], candidate=chosen_tree, check='size_and_seed'))
        joblib.dump(tree, root / 'artifacts/tree.joblib')
        if chosen_conversion == chosen_tree:
            conversion = tree
        else:
            log('Final conversion model fit (different validation-selected tree setting)')
            conversion = TLearner(selection['conversion_params'], config['seed']).fit(train)
        joblib.dump(conversion, root / 'artifacts/conversion.joblib')
        log('Final logistic interaction fit')
        logistic = LogisticInteractions(selection['logistic_c']).fit(train)
        joblib.dump(logistic, root / 'artifacts/logistic.joblib')
        logistic.coefficients().to_csv(root / 'results/tables/logistic_coefficients.csv', index=False)
        final_validation = [validation_row(model, validation, config, family=name)
                            for name, model in [('tree', tree), ('conversion', conversion), ('logistic', logistic)]]
        pd.DataFrame(final_validation).to_csv(root / 'results/tables/final_validation.csv', index=False)
        pd.DataFrame(stability).to_csv(root / 'results/tables/robustness_validation.csv', index=False)
        con = connect(root, config)
        attach(con, root)
        dev_e, dev_n = con.execute("SELECT avg(treatment),count(*) FROM analysis WHERE split!='test'").fetchone()
        con.close()
        sample_counts = train.groupby('treatment').agg(n=('row_id', 'size'), conversions=('conversion', 'sum'),
                                                       visits=('visit', 'sum')).reset_index()
        sample_counts.to_csv(root / 'results/tables/training_arms.csv', index=False)
        freeze = {
            'frozen_utc': datetime.now(timezone.utc).isoformat(), 'primary_budget': config['primary_budget'],
            'primary_contrast': 'tree_uplift minus conversion; IPW; conversion outcome',
            'training_n': len(train), 'validation_n': len(validation), 'development_n': dev_n,
            'development_propensity': dev_e, 'nominal_propensity': config['propensity'],
            'features': FEATURES, 'plan_sha256': checksum(root / 'docs/research_plan.md'),
            'config_sha256': checksum(root / 'config/analysis.json'),
            'model_sha256': {name: checksum(root / f'artifacts/{name}.joblib')
                             for name in ['tree', 'conversion', 'logistic', 'assignment']},
            'selection': selection, 'logistic_solver_iterations': logistic.n_iter,
            'development_seconds': time.monotonic()-started,
            'python': platform.python_version(), 'platform': platform.platform(),
            'test_outcomes_used': False}
        write_json(root / 'results/freeze.json', freeze)
        log('Model freeze written. Final-test evaluation may now run.')


def evaluate(root, config):
    started = time.monotonic()
    freeze = json.loads((root / 'results/freeze.json').read_text())
    if freeze['config_sha256'] != checksum(root / 'config/analysis.json'):
        raise ValueError('Configuration changed after freeze')
    for name, digest in freeze['model_sha256'].items():
        if checksum(root / f'artifacts/{name}.joblib') != digest:
            raise ValueError(f'Model {name} changed after freeze')
    if freeze['plan_sha256'] != checksum(root / 'docs/research_plan.md'):
        raise ValueError('Research plan changed after freeze')
    log('Opening final-test set with frozen models and comparison')
    test = load_split(root, config, 'test')
    n = len(test)
    models = {name: joblib.load(root / f'artifacts/{name}.joblib')
              for name in ['tree', 'conversion', 'logistic']}
    with threadpool_limits(limits=config['threads']):
        preds = {name: predict_batches(model, test) for name, model in models.items()}
        encoder, assignment = joblib.load(root / 'artifacts/assignment.joblib')
        e_x = np.concatenate([assignment.predict_proba(encoder.transform(test.iloc[i:i+200_000]))[:, 1]
                              for i in range(0, n, 200_000)])
    scores = {'conversion': preds['conversion'][1],
              'logistic_uplift': preds['logistic'][1]-preds['logistic'][0],
              'tree_uplift': preds['tree'][1]-preds['tree'][0]}
    t, y = test.treatment.to_numpy(), test.conversion.to_numpy()
    psi = {
        'IPW': ipw_score(t, y, config['propensity']),
        'DR': dr_score(t, y, config['propensity'], *preds['tree']),
        'IPW_empirical_e': ipw_score(t, y, freeze['development_propensity']),
        'DR_feature_e': dr_score(t, y, np.clip(e_x, .05, .95), *preds['tree'])}
    values, differences, curves, metrics, up_to, diagnostics = [], [], [], [], [], []
    fractions = np.unique(np.r_[np.linspace(0, 1, 41), config['budgets']])
    for estimator, pseudo in psi.items():
        for q in config['budgets']:
            q_actual = np.floor(q*n+1e-10)/n
            policies = {name: policy(s, q, test.tie_key, test.row_id) for name, s in scores.items()}
            policies['random'] = np.full(n, q_actual)
            policies['none'] = np.zeros(n)
            for name, p in policies.items():
                values.append({'outcome': 'conversion', 'estimator': estimator, 'policy': name,
                               'budget': q, 'actual_fraction': float(p.mean()), 'n_test': n, **value(p, pseudo)})
            pairs = [(a, 'random') for a in scores] + [('tree_uplift', 'conversion'),
                    ('logistic_uplift', 'conversion'), ('tree_uplift', 'logistic_uplift')]
            for a, b in pairs:
                differences.append({'outcome': 'conversion', 'estimator': estimator, 'budget': q,
                                    'policy_a': a, 'policy_b': b,
                                    **paired_difference(policies[a], policies[b], pseudo)})
            if estimator in ['IPW', 'DR']:
                for name in ['logistic_uplift', 'tree_uplift']:
                    p = policy(scores[name], q, test.tie_key, test.row_id, up_to=True)
                    up_to.append({'estimator': estimator, 'budget': q, 'policy': name,
                                  'actual_fraction': float(p.mean()), **value(p, pseudo)})
            if estimator == 'IPW':
                for name, p in policies.items():
                    if name in scores:
                        chosen = p == 1
                        diagnostics.append({'policy': name, 'budget': q, 'selected_n': int(chosen.sum()),
                            'selected_treated_n': int(np.sum(chosen & (t == 1))),
                            'selected_control_n': int(np.sum(chosen & (t == 0))),
                            'treated_conversions': int(np.sum(y[chosen & (t == 1)])),
                            'control_conversions': int(np.sum(y[chosen & (t == 0)])),
                            'minimum_selected_score': float(scores[name][chosen].min()),
                            'negative_score_fraction': float(np.mean(scores[name] < 0))})
        if estimator in ['IPW', 'DR']:
            for name, s in scores.items():
                c, m = curve(s, pseudo, test.tie_key, test.row_id, fractions)
                curves.extend([{'policy': name, 'estimator': estimator, **v} for v in c])
                metrics.append({'policy': name, 'estimator': estimator, **m})
            for q in fractions:
                q_actual = np.floor(q*n+1e-10)/n
                curves.append({'policy': 'random', 'estimator': estimator, 'budget': q,
                               'actual_fraction': q_actual, **value(np.full(n, q_actual), pseudo)})
    # Clearly separated secondary endpoint: same frozen conversion-trained policies.
    visit_psi = ipw_score(t, test.visit.to_numpy(), config['propensity'])
    for q in config['budgets']:
        random_pi = np.full(n, np.floor(q*n+1e-10)/n)
        for name in list(scores) + ['random', 'none']:
            p = (policy(scores[name], q, test.tie_key, test.row_id) if name in scores else
                 random_pi if name == 'random' else np.zeros(n))
            values.append({'outcome': 'visit', 'estimator': 'IPW', 'policy': name, 'budget': q,
                           'actual_fraction': float(p.mean()), 'n_test': n, **value(p, visit_psi)})
    tables = root / 'results/tables'
    for filename, records in [('policy_values', values), ('paired_differences', differences),
                              ('policy_curves', curves), ('uplift_metrics', metrics),
                              ('up_to_policies', up_to), ('audience_diagnostics', diagnostics)]:
        pd.DataFrame(records).to_csv(tables / f'{filename}.csv', index=False)
    prediction_rows = []
    for name, (m0, m1) in preds.items():
        prediction_rows.append({'model': name, 'arm': 1, **prediction_metrics(y[t == 1], m1[t == 1])})
        prediction_rows.append({'model': name, 'arm': 0, **prediction_metrics(y[t == 0], m0[t == 0])})
    pd.DataFrame(prediction_rows).to_csv(tables / 'prediction_metrics.csv', index=False)
    con = connect(root, config)
    attach(con, root)
    con.execute(sql(root, '05_arm_summary.sql')).df().to_csv(tables / 'all_arms.csv', index=False)
    deciles = []
    for name, s in scores.items():
        model_name = {'conversion': 'conversion', 'logistic_uplift': 'logistic', 'tree_uplift': 'tree'}[name]
        predictions = pd.DataFrame({'row_id': test.row_id, 'tie_key': test.tie_key, 'score': s,
                                    'm1': preds[model_name][1], 'policy': name,
                                    'treatment': t, 'conversion': y})
        con.register('predictions', predictions)
        deciles.append(con.execute(sql(root, '08_score_deciles.sql', propensity=config['propensity'])).df())
        con.unregister('predictions')
    pd.concat(deciles).to_csv(tables / 'score_deciles.csv', index=False)
    con.close()
    effects = []
    for outcome, arr in [('conversion', y), ('visit', test.visit.to_numpy())]:
        effects.append({'outcome': outcome, 'estimator': 'difference_in_means',
                        **difference_in_means(t, arr)})
        effects.append({'outcome': outcome, 'estimator': 'IPW',
                        **mean_ci(ipw_score(t, arr, config['propensity']))})
    effects.append({'outcome': 'conversion', 'estimator': 'DR', **mean_ci(psi['DR'])})
    pd.DataFrame(effects).to_csv(tables / 'overall_effects.csv', index=False)
    # Hypothetical economics only; identical assignment costs at equal capacity.
    economics = []
    for row in values:
        if row['outcome'] == 'conversion' and row['estimator'] == 'IPW' and row['budget'] == config['primary_budget']:
            for conversion_value in [10, 50, 100]:
                for cost in [.01, .05, .10]:
                    economics.append({'policy': row['policy'], 'budget': row['budget'],
                        'hypothetical_value_per_conversion': conversion_value,
                        'hypothetical_cost_per_assignment': cost,
                        'net_value_per_100k': conversion_value*row['estimate']-cost*row['actual_fraction']*100000,
                        'lower': conversion_value*row['lower']-cost*row['actual_fraction']*100000,
                        'upper': conversion_value*row['upper']-cost*row['actual_fraction']*100000})
    pd.DataFrame(economics).to_csv(tables / 'hypothetical_economics.csv', index=False)
    saved = prediction_artifact(test, scores, preds['tree'])
    saved.to_parquet(root / 'artifacts/test_predictions.parquet', index=False)
    write_json(root / 'results/evaluation.json', {
        'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'n_test': n,
        'treatment_fraction': float(t.mean()), 'conversion_rate': float(y.mean()),
        'test_conversions': int(y.sum()), 'test_control_conversions': int(y[t == 0].sum()),
        'evaluation_seconds': time.monotonic()-started,
        'feature_propensity_min': float(e_x.min()), 'feature_propensity_max': float(e_x.max()),
        'feature_propensity_clipped_n': int(np.sum((e_x < .05) | (e_x > .95))),
        'intervals': 'Pointwise normal row-level score intervals, conditional on frozen fits and realized audience cutoffs',
        'freeze_sha256': checksum(root / 'results/freeze.json')})
    log('Held-out evaluation tables saved.')
