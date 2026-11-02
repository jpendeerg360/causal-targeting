"""Publication-style figures from saved aggregate tables (PNG, SVG and PDF)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CONTINUOUS

COLORS = {'random': '#74828e', 'conversion': '#2864a1',
          'logistic_uplift': '#c18425', 'tree_uplift': '#168577', 'none': '#a5abb0'}
LABELS = {'random': 'Random targeting', 'conversion': 'Conversion ranking',
          'logistic_uplift': 'Logistic uplift', 'tree_uplift': 'Tree uplift', 'none': 'No targeting'}
ORDER = ['random', 'conversion', 'logistic_uplift', 'tree_uplift']


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.titleweight': 'bold', 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#bcc5cc', 'axes.labelcolor': '#243340',
        'text.color': '#243340', 'xtick.color': '#485965', 'ytick.color': '#485965',
        'grid.color': '#e5e9ec', 'grid.linewidth': .7, 'savefig.dpi': 300,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})


def save(fig, root, name, caption):
    fig.text(.02, .015, caption, fontsize=8, color='#5b6873', va='bottom')
    fig.subplots_adjust(bottom=.19, top=.82, wspace=.4)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(root / f'results/figures/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def make_figures(root, config):
    style()
    tables = root / 'results/tables'
    read = lambda name: pd.read_csv(tables / f'{name}.csv')
    balance, categories, effect = read('balance'), read('category_balance'), read('overall_effects')
    fig, ax = plt.subplots(1, 3, figsize=(13.7, 4.7), gridspec_kw={'width_ratios': [1, 1.2, 1.1]})
    fig.suptitle('Assignment balance and the overall conversion effect', x=.02, ha='left', fontsize=17, fontweight='bold')
    b = balance.set_index('feature').loc[CONTINUOUS]
    ax[0].scatter(b.smd, np.arange(len(b)), color='#2864a1', s=48, zorder=3)
    ax[0].set_yticks(np.arange(len(b)), b.index)
    ax[0].axvline(0, color='#74828e', lw=1)
    ax[0].set(xlim=(-.05, .05), xlabel='Standardized mean difference', title='Continuous features')
    ax[0].grid(axis='x')
    categories = categories.sort_values('total_variation')
    ax[1].barh(categories.feature, categories.total_variation*100, color='#168577', height=.6)
    ax[1].set(xlabel='Total variation between arms (%)', title='Categorical distributions')
    ax[1].grid(axis='x')
    ax[1].set_axisbelow(True)
    d = effect.query("outcome == 'conversion'").set_index('estimator').loc[['difference_in_means', 'IPW', 'DR']]
    ax[2].errorbar(d.estimate, [2, 1, 0], xerr=[d.estimate-d.lower, d.upper-d.estimate],
                   fmt='o', color='#2864a1', capsize=4)
    ax[2].set_yticks([2, 1, 0], ['Mean difference', 'IPW', 'DR'])
    ax[2].tick_params(axis='y', labelsize=8)
    ax[2].axvline(0, color='#74828e', lw=1)
    ax[2].set(title='Targeting everybody vs nobody', xlabel='Incremental conversions / 100,000')
    ax[2].grid(axis='x')
    save(fig, root, '01_balance_and_effect',
         'Balance uses development rows; effects use final-test rows. 95% intervals assume independent released rows.\nSmall imbalances do not establish exchangeability; original-population effects are not recovered.')

    curves = read('policy_curves')
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    fig.suptitle('How many additional conversions does each audience budget produce?', x=.02, ha='left', fontsize=16, fontweight='bold')
    for a, estimator in zip(ax, ['IPW', 'DR']):
        for name in ORDER:
            c = curves.query('policy == @name and estimator == @estimator').sort_values('budget')
            a.plot(c.budget*100, c.estimate, label=LABELS[name], color=COLORS[name], lw=2,
                   linestyle='--' if name == 'random' else '-')
            a.fill_between(c.budget*100, c.lower, c.upper, color=COLORS[name], alpha=.09)
        a.axhline(0, color='#74828e', lw=.8)
        a.axvline(config['primary_budget']*100, color='#74828e', lw=.8, ls=':')
        a.set(xlabel='Eligible users targeted (%)', title='Primary: IPW' if estimator == 'IPW' else 'Robustness: doubly robust')
        a.grid(alpha=.6)
    ax[0].set_ylabel('Incremental conversions / 100,000 eligible users')
    ax[1].legend(frameon=False, loc='lower right', bbox_to_anchor=(1, .06), fontsize=9)
    save(fig, root, '02_policy_value_curves',
         'Shading: pointwise 95% intervals conditional on fitted policies and realized audience cutoffs; not simultaneous bands.\nExact-capacity policies include negative scores if needed. Random is averaged over uniformly selected audiences.')

    values, differences = read('policy_values'), read('paired_differences')
    q = config['primary_budget']
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.suptitle(f'Primary audience budget: {q:.0%} of eligible users', x=.02, ha='left', fontsize=17, fontweight='bold')
    for j, estimator in enumerate(['IPW', 'DR']):
        d = values.query("outcome == 'conversion' and budget == @q and estimator == @estimator").set_index('policy').loc[ORDER]
        ypos = np.arange(4)[::-1] + (.09 if j == 0 else -.09)
        for i, name in enumerate(ORDER):
            r = d.loc[name]
            ax[0].errorbar(r.estimate, ypos[i], xerr=[[r.estimate-r.lower], [r.upper-r.estimate]],
                           fmt='o' if j == 0 else 's', color=COLORS[name], capsize=3,
                           markerfacecolor=COLORS[name] if j == 0 else 'white')
    ax[0].set_yticks(np.arange(4)[::-1], [LABELS[x] for x in ORDER])
    ax[0].set(title='Incremental conversions', xlabel='Conversions / 100,000 eligible users')
    pairs = [('tree_uplift', 'conversion'), ('tree_uplift', 'logistic_uplift'), ('logistic_uplift', 'conversion')]
    for j, estimator in enumerate(['IPW', 'DR']):
        for i, (a, b) in enumerate(pairs):
            r = differences.query('budget == @q and estimator == @estimator and policy_a == @a and policy_b == @b').iloc[0]
            ax[1].errorbar(r.estimate, 2-i+(.09 if j == 0 else -.09),
                           xerr=[[r.estimate-r.lower], [r.upper-r.estimate]],
                           fmt='o' if j == 0 else 's', color='#243340', capsize=3,
                           markerfacecolor='#243340' if j == 0 else 'white',
                           label=estimator if i == 0 else None)
    ax[1].set_yticks([2, 1, 0], ['Tree − conversion (primary)', 'Tree − logistic', 'Logistic − conversion'])
    ax[1].set(title='Paired policy differences', xlabel='Additional conversions / 100,000')
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.axvline(0, color='#74828e', lw=1, ls='--')
        a.grid(axis='x')
    save(fig, root, '03_primary_budget',
         'Filled circles: IPW. Open squares: DR. Whiskers: 95% confidence intervals.\nPaired comparisons use shared test rows; secondary contrasts are descriptive and unadjusted for multiplicity.')

    pred, deciles = read('prediction_metrics'), read('score_deciles')
    pred = pred.query('arm == 1').set_index('model')
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.9), gridspec_kw={'width_ratios': [1, 1, 1.45]})
    fig.suptitle('Predicting a conversion and changing a conversion are different tasks', x=.02, ha='left', fontsize=16, fontweight='bold')
    policies = ['conversion', 'logistic_uplift', 'tree_uplift']
    names = ['Conversion', 'Logistic uplift', 'Tree uplift']
    model_names = ['conversion', 'logistic', 'tree']
    ax[0].barh(names, [pred.loc[m, 'average_precision'] for m in model_names], color=[COLORS[p] for p in policies])
    ax[0].set(title='Predictive performance', xlabel='Treated-arm average precision')
    ax[0].invert_yaxis()
    d = values.query("outcome == 'conversion' and budget == @q and estimator == 'IPW'").set_index('policy').loc[policies]
    ax[1].barh(names, d.estimate, color=[COLORS[p] for p in policies])
    ax[1].errorbar(d.estimate, np.arange(3), xerr=[d.estimate-d.lower, d.upper-d.estimate], fmt='none', color='#243340', capsize=3)
    ax[1].set(title=f'Targeting value at {q:.0%}', xlabel='Incremental conversions / 100,000')
    ax[1].invert_yaxis()
    d = deciles.query("policy == 'conversion'").sort_values('score_decile')
    ax[2].plot(d.score_decile, d.treated_rate*100, '-o', color='#2864a1', markersize=4, label='Observed treated')
    ax[2].plot(d.score_decile, d.control_rate*100, '-s', color='#74828e', markersize=4, label='Observed control')
    ax[2].fill_between(d.score_decile, d.control_rate*100, d.treated_rate*100, color='#168577', alpha=.16)
    ax[2].set(title='Conversion-ranking diagnostic', xlabel='Score decile (1 = highest)', ylabel='Conversion rate (%)')
    ax[2].set_xticks([1, 2, 4, 6, 8, 10])
    ax[2].legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(axis='x' if a != ax[2] else 'y', alpha=.6)
        a.set_axisbelow(True)
    save(fig, root, '04_prediction_vs_targeting',
         'Predictive metrics use each model’s m1; causal policies rank m1 − m0. Conversion and tree uplift share m1 in this run.\nHigh-response audiences may also convert without assignment. Only the assignment contrast measures incrementality.')

    robustness = read('robustness_validation')
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7))
    fig.suptitle('Development robustness: training size and sampling seed', x=.02, ha='left', fontsize=16, fontweight='bold')
    selected = robustness[robustness.seed == config['seed']]
    selected = selected.sort_values('train_n').drop_duplicates('train_n', keep='last')
    ax[0].errorbar(selected.train_n/1e6, selected.uplift_value, yerr=1.96*selected.uplift_se,
                   fmt='o-', color=COLORS['tree_uplift'], capsize=4)
    ax[0].set(xlabel='Training rows (millions)', ylabel='Incremental conversions / 100,000', title='Nested training samples')
    ax[0].set_xticks([.2, .5, 1, 2])
    selected = robustness[robustness.train_n == config['tune_n']].sort_values('seed')
    ax[1].bar(['Seed '+str(s) for s in selected.seed], selected.uplift_minus_conversion,
              color=COLORS['tree_uplift'])
    ax[1].axhline(0, color='#74828e', lw=1)
    ax[1].set(ylabel='Tree uplift − its conversion ranking', title='Fixed 500,000-row training size')
    ax[1].tick_params(axis='x', labelsize=8)
    for a in ax:
        a.grid(axis='y', alpha=.6)
        a.set_axisbelow(True)
    save(fig, root, '05_development_robustness',
         'All panels use the same 500,000 validation rows. Seed changes the training subset, not the held-out test set.\nError bars capture validation evaluation noise only; three seeds do not estimate total training uncertainty.')
