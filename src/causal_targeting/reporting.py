"""Generate numerical claims directly from saved tables to avoid stale prose."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .data import write_json
from .figures import LABELS, ORDER, make_figures


def ci(row):
    return f"{row['estimate']:.1f} [{row['lower']:.1f}, {row['upper']:.1f}]"


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def report(root: Path, config):
    read = lambda name: pd.read_csv(root / f'results/tables/{name}.csv')
    manifest = json.loads((root / 'results/data_manifest.json').read_text())
    freeze = json.loads((root / 'results/freeze.json').read_text())
    evaluation = json.loads((root / 'results/evaluation.json').read_text())
    assignment = json.loads((root / 'results/assignment_diagnostic.json').read_text())
    values, diffs, effects, pred = [read(n) for n in ['policy_values', 'paired_differences', 'overall_effects', 'prediction_metrics']]
    q = config['primary_budget']
    primary_values = values.query("outcome == 'conversion' and budget == @q and estimator == 'IPW'").set_index('policy')
    primary_diffs = diffs.query("budget == @q and estimator == 'IPW'")
    primary = primary_diffs.query("policy_a == 'tree_uplift' and policy_b == 'conversion'").iloc[0]
    dr_primary = diffs.query("budget == @q and estimator == 'DR' and policy_a == 'tree_uplift' and policy_b == 'conversion'").iloc[0]
    simple = primary_diffs.query("policy_a == 'tree_uplift' and policy_b == 'logistic_uplift'").iloc[0]
    tree_random = primary_diffs.query("policy_a == 'tree_uplift' and policy_b == 'random'").iloc[0]
    ate = effects.query("outcome == 'conversion' and estimator == 'difference_in_means'").iloc[0]
    arms = read('all_arms')
    total = arms.groupby('treatment')[['n', 'conversions', 'visits', 'exposed']].sum()
    treatment_rate = total.loc[1, 'n']/total.n.sum()
    outcome_rate = total.conversions.sum()/total.n.sum()
    n_test = evaluation['n_test']
    split_counts = {r['split']: r['n'] for r in manifest['split_counts']}
    robust = read('robustness_validation')
    seed_rows = robust[robust.train_n == config['tune_n']]
    conclusions = (
        'The flexible uplift policy outperformed conversion targeting in the prespecified comparison.' if primary.lower > 0 else
        'The flexible uplift policy underperformed conversion targeting in the prespecified comparison.' if primary.upper < 0 else
        'The prespecified comparison is inconclusive: the interval includes zero. More flexible causal modeling did not establish a targeting advantage over conversion prediction.')
    summary = {
        'title': 'Causal Targeting: Who Actually Benefits from Advertising?',
        'full_rows': int(total.n.sum()), 'train_n': freeze['training_n'],
        'validation_n': freeze['validation_n'], 'test_n': n_test,
        'primary_budget': q, 'primary_estimator': 'IPW',
        'primary_contrast': {k: float(primary[k]) for k in ['estimate', 'se', 'lower', 'upper']},
        'tree_vs_random': {k: float(tree_random[k]) for k in ['estimate', 'se', 'lower', 'upper']},
        'conclusion': conclusions,
        'units': 'incremental conversions per 100,000 eligible benchmark users'}
    write_json(root / 'results/summary.json', summary)
    main_table = md_table(['Policy at 20%', 'IPW estimate [95% CI]', 'Difference vs random [95% CI]'], [
        [LABELS[name], ci(primary_values.loc[name]),
         'Reference' if name == 'random' else ci(primary_diffs.query('policy_a == @name and policy_b == "random"').iloc[0])]
        for name in ORDER])
    dr_values = values.query("outcome == 'conversion' and budget == @q and estimator == 'DR'").set_index('policy')
    feature_values = values.query("outcome == 'conversion' and budget == @q and estimator == 'DR_feature_e'").set_index('policy')
    compare_table = md_table(['Policy', 'IPW', 'Doubly robust'], [
        [LABELS[name], ci(primary_values.loc[name]), ci(dr_values.loc[name])] for name in ORDER])
    metric_table = md_table(['Model / treated outcome predictions', 'ROC AUC', 'Average precision', 'Log loss'], [
        [name, f'{r.roc_auc:.3f}', f'{r.average_precision:.4f}', f'{r.log_loss:.5f}']
        for name in ['conversion', 'logistic', 'tree']
        for _, r in pred.query('model == @name and arm == 1').iterrows()])
    arm_table = md_table(['Assignment', 'Rows', 'Conversions', 'Conversion rate', 'Visit rate', 'Exposure rate'], [
        ['Control' if arm == 0 else 'Treatment', f'{int(row.n):,}', f'{int(row.conversions):,}',
         f'{row.conversions/row.n:.4%}', f'{row.visits/row.n:.3%}', f'{row.exposed/row.n:.3%}']
        for arm, row in total.iterrows()])
    # Mechanical development ATE from SQL aggregates, with independent-arm SE.
    dev = read('development_arms').groupby('treatment')[['n', 'conversions', 'visits']].sum()
    dev_effects = []
    for outcome, column in [('conversion', 'conversions'), ('visit', 'visits')]:
        p0, p1 = dev.loc[0, column]/dev.loc[0, 'n'], dev.loc[1, column]/dev.loc[1, 'n']
        se = math.sqrt(p0*(1-p0)/(dev.loc[0, 'n']-1)+p1*(1-p1)/(dev.loc[1, 'n']-1))*100000
        est = (p1-p0)*100000
        dev_effects.append({'outcome': outcome, 'estimate': est, 'se': se, 'lower': est-1.95996398454*se, 'upper': est+1.95996398454*se})
    pd.DataFrame(dev_effects).to_csv(root / 'results/tables/development_effects.csv', index=False)
    readme = f'''# Causal Targeting: Who Actually Benefits from Advertising?

A completed Python + DuckDB research project comparing conversion prediction with causal targeting on the **corrected Criteo Uplift v2.1** benchmark.

**Main finding.** At the prespecified 20% audience budget, boosted-tree uplift targeting versus conversion targeting produced **{ci(primary)}** additional conversions per 100,000 eligible users. {conclusions}

The same tree policy beat expected random targeting by **{ci(tree_random)}**. These are benchmark assignment effects, not estimates of actual advertiser ROI.

![Held-out policy comparisons](results/figures/03_primary_budget.png)

## Results

{main_table}

All values are incremental conversions per **100,000 eligible users**, relative to targeting nobody; nobody has value zero. Intervals are pointwise and condition on fitted models and realized audience cutoffs. Only tree uplift minus conversion at 20% was the primary comparison. Random averages over uniform fixed-size audiences.

- Full verified release: **{int(total.n.sum()):,} rows**, 12 features, {treatment_rate:.2%} assigned treatment, {outcome_rate:.3%} conversion prevalence.
- Final fit: **{freeze['training_n']:,} training rows**; tuning: {config['tune_n']:,} training and **{freeze['validation_n']:,} validation rows**.
- Untouched test: **{n_test:,} rows**, including {evaluation['test_conversions']:,} conversions and {evaluation['test_control_conversions']:,} control conversions.
- Test ATE (difference in means): **{ci(ate)}** conversions per 100,000.

## Reproduce

Python 3.12 recommended; run commands from this repository directory. Network access is needed for packages and the 311 MB publisher-hosted dataset. Allow roughly 4 GB of disk space plus the environment and up to several GB of RAM. The observed model-development stage took {freeze['development_seconds']:.0f} seconds; evaluation took {evaluation['evaluation_seconds']:.0f} seconds on the recorded Apple Silicon environment with four worker threads. Timing varies by hardware; ingestion and figure rendering are additional.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
python -m causal_targeting run --rebuild
python -m causal_targeting verify
```

`--rebuild` archives prior aggregate results under ignored `artifacts/` and repeats the fixed plan, including the same test set. This is reproduction, not a new independent experiment. Download, checksum verification, ingestion, training, evaluation, figures and Markdown are included. Small floating-point differences across platforms are possible. Tests use synthetic fixtures; all reported results use the real corrected release.

For an existing downloaded file:

```bash
python -m causal_targeting run --rebuild --data-file /absolute/path/criteo-research-uplift-v2.1.csv.gz
```

Separate stages: `download`, `prepare`, `develop`, `evaluate`, `report`, `verify`. `report` regenerates figures and prose from aggregate results without raw data or model fitting. `evaluate` requires the frozen local model artifacts. `verify` checks the distributable results and, when present, local row-level artifacts. To recover after an interrupted stage, rerun that stage; never change tuning decisions after seeing test results.

## Design

The fixed [research plan](docs/research_plan.md) predates modeling. Hash-based 60/10/30 train/validation/test splits use source row positions; they do not invent user identities or time order. Training and validation caps are independent hash samples. All 12 features are pretreatment; exposure, outcomes, treatment and generated IDs cannot enter baseline covariates.

Three rankings: treated-arm boosted conversion probability; regularized logistic treatment interactions; boosted-tree T-learner probability difference. Two settings per causal family keep tuning modest. Continuous features are scaled for logistic regression; anonymized categorical tokens use train-only pooling, one-hot encoding or native categorical tree splits. The models never verify a user's individual treatment effect.

Policies select exactly 5%, 10%, 20% or 50% of each evaluation batch (rounded down). Ties use independent deterministic keys. An additional analysis caps causal targeting at positive estimated effects. Evaluation uses treatment assignment and nominal propensity 0.85, with paired IPW intervals, independent-training outcome nuisances for DR, and propensity sensitivity checks. Full formulas and caveats are in the report.

## Read the project

| Artifact | Purpose |
|---|---|
| [Research report](docs/research_report.md) | Identification, results, uncertainty and business decision |
| [Interview guide](docs/interview_guide.md) | Defend each major design choice |
| [Résumé bullets](docs/resume_bullets.md) | Two claims grounded in completed work |
| [Data sources](docs/data_sources.md) | Official erratum, papers, license, mirror and checksum |
| [Progress log](PROGRESS.md) | Decisions, execution and limitations |
| `sql/` | Ingestion, analysis table, quality, missingness, balance and score-decile queries |
| `src/causal_targeting/` | Models, estimators, CLI and generated reporting |
| `tests/` | Split, leakage, coding, budget and unequal-allocation estimator tests |
| `results/tables/`, `results/figures/` | Machine-readable estimates and PNG/SVG/PDF figures |
| `results/freeze.json` | Preresult plan/config/model hashes and chosen comparison |

## Limits and license

This is a resampled, anonymized benchmark. Undisclosed negative-label sampling prevents recovering source-population effects. The corrected paper supports a common assignment ratio but does not provide every sampling probability; retained-sample exchangeability remains an assumption. Feature-based propensity sensitivity changes absolute policy values substantially while preserving the primary comparison's direction; effect magnitudes are less stable than the ranking conclusion. There are no cluster/experiment IDs or timestamps, so intervals use an independent-row approximation and random splitting does not establish temporal generalization. Intervals exclude retraining variability; three validation seeds are a stability check, not a full uncertainty analysis. Rare control conversions constrain precision. Anonymous categories do not support demographic fairness claims.

Prospective policy randomization, actual costs, business outcomes and operational safeguards are needed before deployment. Hypothetical economics are labeled and never presented as real ROI.

Data and derived research outputs: [CC BY-NC-SA 4.0](DATA_LICENSE.md). Original software: [MIT](LICENSE). Cite Diemert et al. (2018, 2021), as listed in [data sources](docs/data_sources.md). No Criteo endorsement is implied. Raw data, fitted models and individual predictions are excluded from Git and the source archive.
'''
    (root / 'README.md').write_text(readme)
    visit_values = values.query("outcome == 'visit' and estimator == 'IPW' and budget == @q").set_index('policy')
    visit_table = md_table(['Frozen conversion policy', 'Incremental visits / 100,000 [95% CI]'],
                           [[LABELS[p], ci(visit_values.loc[p])] for p in ORDER])
    up = read('up_to_policies').query("budget == @q and estimator == 'IPW'")
    up_table = md_table(['Policy', 'Actual positive-score fraction', 'Incremental conversions [95% CI]'],
                        [[LABELS[r.policy], f'{r.actual_fraction:.2%}', ci(r)] for _, r in up.iterrows()])
    propensity_table = md_table(['Estimator', 'Tree minus conversion [95% CI]'], [
        [name, ci(diffs.query('budget == @q and estimator == @name and policy_a == "tree_uplift" and policy_b == "conversion"').iloc[0])]
        for name in ['IPW', 'DR', 'IPW_empirical_e', 'DR_feature_e']])
    report_text = f'''# Causal Targeting: Who Actually Benefits from Advertising?

## Finding and decision

At 20% capacity, tree uplift targeting achieved {ci(primary_values.loc['tree_uplift'])} incremental conversions per 100,000 eligible benchmark users. Its paired advantage over conversion ranking was **{ci(primary)}**, versus {ci(dr_primary)} with doubly robust evaluation. {conclusions}

Its difference from the interpretable logistic uplift policy was {ci(simple)}. These comparisons do not justify choosing a model by predictive accuracy alone, and they do not establish a universal advantage for a model family. The practical next step is a prospective randomized comparison of candidate targeting policies with business-specific outcomes and costs, rather than deployment based on one offline benchmark.

## Dataset and estimand

We verified the corrected 13,979,592-row file against Criteo's pinned publisher repository SHA-256, schema and erratum. The old download endpoint was unavailable; provenance, exact URL, license and source citations are in [data_sources.md](data_sources.md) and [the manifest](../results/data_manifest.json). No synthetic observations enter the empirical analysis.

The unit is a released record, described as a browser user in an advertiser experiment. Treatment means assignment to advertising eligibility; exposure is a downstream event, not the intervention used here. The primary outcome is conversion; visits are secondary. Only f0–f11 are covariates. Anonymous feature tokens have no inferred business meaning.

For an audience rule pi(X), the target is

```
V(pi) = E[pi(X) * (Y(1) - Y(0))].
```

This is incremental value relative to nobody being assigned advertising. Total conversions under a policy instead equal E[Y(0)] + V(pi). Conversions observed only among selected treated users also contain purchases that would occur without assignment, and cannot identify V. For a random audience of fraction q, expected incremental value is q times ATE.

The corrected-release paper supports common-ratio treatment allocation at 0.85, because experiments were resampled to remove advertiser-related assignment leakage. It does **not** disclose the label-sampling probabilities. Identification for the released benchmark therefore assumes retained-sample exchangeability, consistency, positivity and no interference. Without those assumptions, these are evaluations of a standardized released-data contrast, E[pi(X)(E[Y|T=1,X]−E[Y|T=0,X])], rather than a recovered population causal effect. DR does not repair unknown sampling or hidden confounding. Neither insignificant balance diagnostics nor a chance-level classifier proves the design.

## Experimental description

Full-release summaries (computed mechanically after model freeze for reporting):

{arm_table}

No values were missing/nonfinite; all four indicators were binary. No control row had exposure and no conversion lacked a visit. Development-only conversion ATE was {ci(dev_effects[0])}; the untouched-test difference-in-means ATE was {ci(ate)}, both per 100,000. Conversion is rare, and only {evaluation['test_control_conversions']:,} positive control outcomes are available in the full test set. A small selected audience has fewer still; see `audience_diagnostics.csv` for exact counts. Normal intervals are approximate; they should not be extrapolated to tiny segments.

![Balance and overall effect](../results/figures/01_balance_and_effect.png)

Balance uses standardized mean differences for continuous features and total variation distance for categorical distributions. Categorical numeric SMDs are retained only as mechanical summaries, not interpreted as distances between ordered feature values. Quantiles use DuckDB's bounded-memory approximation. The assignment classifier's validation AUC was {assignment['validation_roc_auc']:.4f}; its log loss {assignment['validation_log_loss']:.5f} was compared with {assignment['constant_log_loss']:.5f} for constant 0.85. These are diagnostics, not a recovered randomization audit.

## Splits and models

Deterministic label-blind hashes created {split_counts['train']:,} train-pool, {split_counts['validation']:,} validation-pool and {n_test:,} test rows. We used nested train samples of {config['pilot_n']:,}, {config['tune_n']:,}, and {freeze['training_n']:,}; validation tuning used {freeze['validation_n']:,} hash-selected rows. Remaining development rows contributed descriptive balance/allocation summaries, not model training or tuning. All test rows were evaluated. Source row positions provide reproducibility only; no genuine user, advertiser, experiment, or timestamp key is available. Repeated users across advertisers and dependencies cannot be identified or ruled out.

1. **Conversion ranking:** fit a boosted classifier in T=1; rank m1(X).
2. **Logistic uplift:** fit a regularized logistic model with baseline covariates, assignment and every covariate-by-assignment interaction; score sigmoid(eta(X,1))−sigmoid(eta(X,0)). Coefficients describe anonymous log-odds terms, not direct probability effects.
3. **Tree uplift:** fit separate boosted classifiers in T=0 and T=1; score m1(X)−m0(X).

The logistic model scales continuous inputs and one-hot encodes up to 128 categories per categorical feature. The trees use native categorical splits after retaining the 254 most frequent training categories plus an other category. Pooling is a resource/variance tradeoff that can discard useful rare categories. Unseen categories map to other. Preprocessing is fit exclusively on the training subset. No class weighting or outcome oversampling distorts the probability scale.

We evaluated two tree settings and C in {{0.1, 1.0}} for logistic regression. Causal settings were chosen by validation IPW value at 20%; the conversion setting by treated-arm validation log loss. Validation policy values are noisy, so selection optimism is expected and addressed by final holdout. Final choices: tree {freeze['selection']['tree_params']}; logistic C={freeze['selection']['logistic_c']}. The same tree setting won both selections, so the conversion model and T-learner share m1, isolating their ranking-score difference. Logistic optimization terminated in {freeze['logistic_solver_iterations']} iterations out of the 300 cap. No optional causal forest, neural network or DR learner was added; DR here is an evaluator.

## Policies and estimation

The 5%, 10%, 20% and 50% budgets were fixed in advance. Exact-capacity policies select floor(q*n) users by descending score, with independent deterministic hash keys and row ordinal for ties. Negative estimates remain eligible when capacity must be filled. This is a batch allocation rule; score cutoffs adapt to the batch's pretreatment features, never outcomes. Random is integrated over all uniform fixed-size audiences, represented by constant selection probabilities floor(q*n)/n. It avoids adding lottery noise and uses the same expected budget. No-targeting has exactly zero incremental value and zero interval width by definition.

With e=0.85, the primary score and independent-training DR check are:

```
psi_IPW = T*Y/e - (1-T)*Y/(1-e)
psi_DR  = m1(X)-m0(X) + T*(Y-m1(X))/e - (1-T)*(Y-m0(X))/(1-e)
V_hat(pi) = mean(pi * psi)
Delta_hat = mean((pi_A - pi_B) * psi)
SE(Delta_hat) = sd((pi_A - pi_B) * psi, ddof=1) / sqrt(n)
```

Outcome nuisances are the final T-learner fit only on training rows. No test cross-fitting is necessary because fitting and evaluation are separate. The 95% intervals use ±1.96 SE and preserve pairing; subtracting separate CI endpoints would be wrong. They condition on fitted models and realized label-blind audience cutoffs and do not include model selection, training variation, cutoff-estimation variation, clustering or transport uncertainty. The independent-row score method is an approximation to the benchmark experiment. No experiment IDs exist to support cluster-robust inference; uncertainty could be understated if residual dependence is material. Curves have pointwise, not simultaneous, bands. Only the prespecified primary contrast is treated as confirmatory; all other contrasts are descriptive without multiplicity correction.

## Held-out results

{compare_table}

![Value curves](../results/figures/02_policy_value_curves.png)

![Primary comparison](../results/figures/03_primary_budget.png)

All numbers use per-100,000 eligible users, not per-100,000 targets. The tree versus random contrast was {ci(tree_random)}. At 20%, multiply a selected-population treatment effect by 0.20 to obtain the eligible-population value; failing to do so would overstate reach-adjusted impact fivefold.

## Prediction diagnostic

{metric_table}

Average precision is useful with this low outcome prevalence, but it rewards identifying outcomes under treatment, not incremental outcomes. The conversion model can correctly rank likely buyers who would also buy without advertising. The score-decile plot places treated and control conversion rates beside one another: their difference, not the treated curve alone, supplies assignment-effect evidence. Because individual effects are unobserved, there is no individual-effect RMSE or claim of verified personal benefit.

![Prediction versus targeting](../results/figures/04_prediction_vs_targeting.png)

## Focused robustness

At {config['tune_n']:,} training rows, three independently hash-selected training subsets gave validation tree-policy values from {seed_rows.uplift_value.min():.1f} to {seed_rows.uplift_value.max():.1f} per 100,000; paired differences from their own conversion rankings ranged from {seed_rows.uplift_minus_conversion.min():.1f} to {seed_rows.uplift_minus_conversion.max():.1f}. These samples overlap; this is a limited stability diagnostic, not three independent replications. Larger training samples and the simple/flexible comparison are shown below. All stability checks stay in development data.

![Development robustness](../results/figures/05_development_robustness.png)

The empirical development treatment allocation was {freeze['development_propensity']:.8f}, versus nominal 0.85. A separate feature-based assignment model was fit on development data and used only as a sensitivity check, clipped to [0.05,0.95] ({evaluation['feature_propensity_clipped_n']} test predictions clipped):

{propensity_table}

The empirical-propensity intervals condition on that estimated nuisance; they omit its estimation uncertainty. The feature-propensity result also depends on its model. Absolute values are materially sensitive: tree uplift changes from {ci(dr_values.loc['tree_uplift'])} under constant-propensity DR to {ci(feature_values.loc['tree_uplift'])} under feature-propensity DR; conversion ranking changes from {ci(dr_values.loc['conversion'])} to {ci(feature_values.loc['conversion'])}. The assignment model did not improve validation log loss over the constant benchmark, so it is not an established replacement for the documented design probability. Its extreme predictions can nevertheless alter weighted residuals. The primary difference remains negative. This supports the relative comparison more strongly than a precise absolute-effect magnitude. These checks cannot undo unknown label sampling.

Up-to-budget policies require positive estimated uplift as well as being in the top budget:

{up_table}

They can spend less than the exact-capacity policy. A zero estimated effect threshold is a model decision, not proof that all omitted users have nonpositive causal effects.

Secondary visits use the **same conversion-trained policies**, with no separate visit tuning:

{visit_table}

Visit improvements cannot replace the primary conversion result. They answer a different business question.

## Secondary curve summaries and business sensitivity

We define AUUC = integral_0^1 V(q)dq and Qini area = integral_0^1 [V(q)−q V(1)]dq. Every empirical prefix is integrated by trapezoids, with deterministic tie breaking. `uplift_metrics.csv` contains probability units and values multiplied by 100,000. There is no oracle/perfect-model normalization and these numbers need not match packages with differently normalized Qini coefficients. They are secondary descriptive summaries with no significance ranking.

`hypothetical_economics.csv` assumes values of $10/$50/$100 per incremental conversion and costs of $0.01/$0.05/$0.10 per assignment: net value = value_per_conversion*V − cost_per_assignment*q. These are illustrative dollars, not measured revenue, actual spend or advertiser ROI. Equal-budget costs cancel in pairwise value differences. Assignment is not an impression, so this is a stylized cost model; actual billing may differ.

## What can be defended

The executed analysis supports an out-of-sample comparison of fixed assignment-targeting policies on this corrected benchmark, under the stated sampling and independence assumptions. It does not identify original customer-population ROI, effects for particular users, fair treatment across demographics, later-time performance or cluster-robust precision. A prospective experiment should randomize eligible users among the proposed policy and the best practical comparator at equal capacity, define actual cost/profit and guardrail outcomes, preserve assignment logging and holdout integrity, and estimate uncertainty at the real randomization unit.

The repository includes deterministic ingestion, substantive SQL, frozen metadata, machine-readable tables, vector figures, dependency pins and targeted tests. See the README for exact commands and the interview guide for a concise defense of the design.
'''
    (root / 'docs/research_report.md').write_text(report_text)
    (root / 'docs/resume_bullets.md').write_text(f'''# Résumé bullets

- Built and executed a reproducible Python/DuckDB causal-targeting pipeline on **{int(total.n.sum()):,} corrected Criteo records**, with validated Parquet ingestion, leakage-safe model fitting on **{freeze['training_n']:,} rows**, and evaluation on **{n_test:,} held-out rows**.
- Compared conversion ranking, logistic treatment interactions and boosted-tree uplift policies using unequal-propensity IPW, doubly robust checks and paired confidence intervals; at 20% targeting, estimated tree uplift versus conversion ranking at **{ci(primary)} incremental conversions per 100,000 eligible users**, and versus random at **{ci(tree_random)}**.

Intervals are 95% CIs. These describe a research benchmark, not realized business revenue or a deployed campaign. If space is limited, retain the comparison and its uncertainty rather than replacing it with an unsupported uplift percentage.
''')
    make_figures(root, config)
    print('Generated README, report, résumé bullets and five figure sets.', flush=True)
