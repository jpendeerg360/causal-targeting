# Interview guide

## A 60-second explanation

“I asked who should receive an advertising assignment when capacity is limited. I used the corrected Criteo randomized benchmark, verified its version and license, and built a Python/DuckDB pipeline with an untouched final holdout. I compared buying-propensity ranking, a logistic model with treatment interactions, and a boosted-tree T-learner. The target was incremental conversions relative to assigning nobody, evaluated using both treatment and control outcomes. At the prespecified 20% budget, conversion targeting beat tree uplift by 15.5 conversions per 100,000 eligible users, with a paired 95% interval from 10.6 to 20.5. DR evaluation agreed. The interesting result is that a more elaborate causal score did not make the better targeting decision. Unknown release sampling and unavailable experiment IDs limit causal interpretation and precision.”

Use the generated [report](research_report.md) and [résumé bullets](resume_bullets.md) for exact numbers. Do not call this a deployed system or realized revenue lift.

## What is the estimand?

`V(pi) = E[pi(X) {Y(1)-Y(0)}]`, per eligible user, with assignment as treatment. Multiply by 100,000 to get incremental conversions at that eligible-population scale. `Y(1)` and `Y(0)` are never both observed for one record. The model estimates conditional average differences, not verified individual effects.

At a 20% audience limit, a 0.005 average effect within the selected audience would mean 0.001 additional conversions per eligible user, or 100 per 100,000. This is an illustrative arithmetic example, not an empirical result. Reporting the selected-audience rate without multiplying by reach changes the denominator.

## Why not rank by probability of purchase?

`m1(x)` rewards users likely to convert under treatment; `m1(x)-m0(x)` rewards changes attributable to assignment. High `m1` can come with high `m0`. The empirical score-decile figure shows both arms' conversion rates, making that distinction visible. In this run, conversion ranking nevertheless won the primary comparison: treatment-effect estimation introduces error from two models, especially a control model with relatively few positives. A conceptual reason for uplift targeting is not evidence that a particular uplift estimator works better.

The logistic model had stronger treated-arm predictive metrics than the tree, yet predictive performance alone did not determine the ordering of their targeting policies. Be clear that prediction metrics evaluate m1 while uplift policies rank a difference.

## What makes this causal, and what remains an assumption?

The original intervention was assignment to advertising eligibility versus suppression. The corrected paper describes resampling advertiser experiments to a common treatment fraction. A causal interpretation still needs consistency, no interference, overlap and exchangeability in the retained release. Unknown label-dependent sampling prevents reconstructing original-population incrementality. A source-based design argument and diagnostics are both needed; nonsignificant balance tests do not prove randomization survived every processing step.

If that retained-sample assumption fails, the estimated quantity is a standardized contrast of released conditional outcome distributions, not a source-population causal effect. This qualification is part of the finding, not a footnote to hide.

## Why not use exposure as treatment or a feature?

Exposure happens after assignment and depends on auction/delivery and user behavior. Conditioning on it changes the population and may create selection bias. We estimate intention-to-treat assignment effects. A treatment-on-the-treated or instrumental-variable analysis would need additional assumptions and a different estimand; it was not part of this project.

## Why 0.85 instead of 0.5?

The corrected dataset has strongly unequal treatment allocation. Under constant assignment probability e, `T*Y/e - (1-T)*Y/(1-e)` has expectation `m1-m0`. Using 0.5 when e is 0.85 gives the wrong weighting. The code has an exact synthetic test showing this failure. We used the paper's nominal ratio, then checked the development estimate and a learned assignment model. The feature-based check changed absolute values materially while preserving the direction of the primary comparison; this warrants caution about effect magnitudes.

## What does “doubly robust” mean here?

The score adds a modeled effect to propensity-weighted outcome residuals. Under identification and regularity conditions it targets the same value when either the propensity or outcome regression is correct. It does not protect against unknown selection, wrong treatment definitions or interference. The nuisance outcome models were trained on separate development rows. “DR evaluator” does not mean we trained a DR learner.

## Why a T-learner and logistic interactions?

They are easy to explain and offer a focused simple/flexible comparison. The logistic model shares a parametric structure and shrinks interaction coefficients; trees allow nonlinearities and interactions but fit the low-prevalence control outcome separately. Separate-model error can dominate a small effect difference. Anonymous categorical tokens were encoded categorically; imposing an arbitrary numeric order would be unjustified. Rare-category pooling trades detail for variance and runtime.

## How did you prevent leakage?

- Only f0–f11 are allowed into baseline preprocessing.
- Treatment enters the logistic model only as the assignment term and treatment interactions; it is not a baseline ranking input.
- Exposure, outcomes and generated row keys never enter covariates.
- Scaling, category dictionaries and models use training rows only.
- Hyperparameters use validation, and all settings are hashed and frozen before opening test outcomes.
- Split/sample hashes are independent of outcomes and treatment. No stratification or outcome oversampling alters prevalence.

The file has no real identifiers or timestamps. A generated ordinal is useful for stable bookkeeping, not for grouping people. We cannot rule out the same person appearing across advertiser trials, and a random split does not establish future-time generalization.

## How are policies evaluated fairly?

Every exact-capacity policy selects the same number of eligible users. Ties are broken without labels. Evaluation uses both assigned arms; comparing selected treated conversion rates alone is invalid. Random targeting is averaged over uniform fixed-size audiences. It has the same expected reach without additional random-draw noise. The no-targeting incremental value is identically zero.

An up-to-budget policy additionally rejects nonpositive estimated effects and can spend less. A zero score cutoff does not prove the rejected users would be harmed.

## Why are the paired intervals narrow compared with individual policy intervals?

The policies select many of the same users. In a paired contrast, identical decisions cancel, and uncertainty is driven by disagreements. The code computes the standard deviation of `(pi_A-pi_B)*psi` directly; it does not treat policy estimates as independent or subtract confidence-interval endpoints.

The intervals condition on fitted policies and the realized feature-based cutoff. They do not capture retraining, threshold estimation or unavailable experiment-level clustering. Pointwise curve bands are not simultaneous statements about the whole curve. Secondary comparisons are descriptive, and selecting the best test point would invalidate the prespecified interpretation.

## What did SQL contribute?

Readable SQL performs strict CSV ingestion, quality constraints, missingness and distinct-value counts, label-blind split assignment, Parquet construction, arm aggregation, continuous/categorical balance, and score-decile diagnostics using window functions. Python handles fitting, estimator logic, inference, figures and reporting. It is substantive analytical SQL, not a single SELECT used to add a keyword.

## What tests matter?

Test known effects under unequal allocation, heterogeneous propensity, DR reduction to IPW, DR with correct outcome models, identical-policy cancellation, exact audience counts, tie stability under permutation, split stability after changing labels, schema and structural constraints, and train-only category handling. Real-data verification independently reconciles key saved estimates with stored predictions and checks report claims against machine-readable summaries.

## What would you do next?

First validate a business objective and real randomization unit, then run a prospective equal-budget policy experiment. Include cost per assignment/impression, incremental profit, user-experience guardrails and privacy constraints. If offline modeling continues, use new development data to investigate control-model regularization and category pooling. Do not tune on this already-opened test set. A causal forest or DR learner would need to address a specific weakness rather than decorate the technology stack.

## What should you never claim?

No actual advertiser revenue/ROI, observed individual effects, demographic fairness, temporal generalization, cluster-robust precision, or production deployment. No claim that causal targeting always wins. No claim that the corrected release's sampling mechanism is fully known. The credible story is a completed, auditable experiment analysis with an informative negative model-comparison result.
