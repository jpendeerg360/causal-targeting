# Causal Targeting: Who Actually Benefits from Advertising?

A completed Python + DuckDB research project comparing conversion prediction with causal targeting on the **corrected Criteo Uplift v2.1** benchmark.

**Main finding.** At the prespecified 20% audience budget, boosted-tree uplift targeting versus conversion targeting produced **-15.5 [-20.5, -10.6]** additional conversions per 100,000 eligible users. The flexible uplift policy underperformed conversion targeting in the prespecified comparison.

The same tree policy beat expected random targeting by **72.2 [63.4, 81.1]**. These are benchmark assignment effects, not estimates of actual advertiser ROI.

![Held-out policy comparisons](results/figures/03_primary_budget.png)

## Results

| Policy at 20% | IPW estimate [95% CI] | Difference vs random [95% CI] |
| --- | --- | --- |
| Random targeting | 23.1 [20.6, 25.6] | Reference |
| Conversion ranking | 110.9 [98.9, 122.9] | 87.8 [78.2, 97.4] |
| Logistic uplift | 104.4 [92.8, 116.0] | 81.3 [71.9, 90.6] |
| Tree uplift | 95.4 [84.4, 106.4] | 72.2 [63.4, 81.1] |

All values are incremental conversions per **100,000 eligible users**, relative to targeting nobody; nobody has value zero. Intervals are pointwise and condition on fitted models and realized audience cutoffs. Only tree uplift minus conversion at 20% was the primary comparison. Random averages over uniform fixed-size audiences.

- Full verified release: **13,979,592 rows**, 12 features, 85.00% assigned treatment, 0.292% conversion prevalence.
- Final fit: **2,000,000 training rows**; tuning: 500,000 training and **500,000 validation rows**.
- Untouched test: **4,193,056 rows**, including 12,300 conversions and 1,227 control conversions.
- Test ATE (difference in means): **115.8 [103.5, 128.2]** conversions per 100,000.

## Reproduce

Python 3.12 recommended; run commands from this repository directory. Network access is needed for packages and the 311 MB publisher-hosted dataset. Allow roughly 4 GB of disk space plus the environment and up to several GB of RAM. The observed model-development stage took 41 seconds; evaluation took 245 seconds on the recorded Apple Silicon environment with four worker threads. Timing varies by hardware; ingestion and figure rendering are additional.

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
