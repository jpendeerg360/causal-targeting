# Research plan — frozen before final-test evaluation

Written 2026-11-02, before fitting models or inspecting final-test outcomes.

**Question.** At a fixed audience capacity, which ranking maximizes incremental conversions relative to advertising assignment for nobody?

**Population and unit.** A row in the corrected Criteo Uplift v2.1 released benchmark. The source describes a user/browser in an advertiser experiment; the release has no usable user, advertiser, experiment, or time identifiers. A generated source-row ordinal is an ingestion key only. No claims about unique people across experiments or temporal transportability.

**Treatment/outcomes.** `treatment` is randomized advertising eligibility/assignment, not receipt. Primary outcome: binary conversion. Secondary: binary visit. `exposure` is used only for quality control. Only `f0`–`f11` enter prediction models. The paper identifies continuous f0/f2/f7/f10; other features are anonymous categorical tokens.

**Estimands.** Benchmark assignment ATE and V(pi) = E[pi(X){Y(1)-Y(0)}]. The primary contrast is V(boosted T-learner, 20%) minus V(conversion ranking, 20%). Comparisons against random and the logistic interaction policy are secondary. Targeting is evaluated per eligible user, not per selected user. Scale values to 100,000 eligible users.

**Identification.** Consistency, no interference, positive assignment probabilities, and exchangeability of assignment for the released benchmark. Use nominal e=0.85 from the corrected-release paper's common-ratio resampling; check sensitivity to the development allocation estimate and a feature-dependent assignment model. The undisclosed negative-label sampling precludes recovery of original-population effects. Covariate balance cannot prove exchangeability. Treat the contrast of released conditional outcome regressions as a benchmark functional if sampling breaks causal identification.

**Splits.** Deterministic MD5 hash of source-row ordinal and salt, independent of features and labels: 60% train, 10% validation, 30% final test. Separate salted hash selects nested train samples and breaks score ties. Entire final-test outcomes remain unused for tuning. No invented grouping or chronology. Distribution diagnostics and experiment exploration use development rows; full-file validation is mechanical only.

**Resource budget.** Pilot 200,000 training rows; tune on 500,000 training rows with up to 500,000 validation rows; final training on 2,000,000 rows with the fixed selected settings. All final-test rows are evaluated. Training subsets are hash-selected from the full training pool without outcome balancing or oversampling.

**Models.** (1) Boosted-tree conversion predictor fitted in the treatment arm, scored by m1(x). (2) L2 logistic regression on pretreatment covariates, treatment, and treatment interactions; score m1(x)-m0(x). (3) Boosted-tree T-learner with separate arm regressions; score m1(x)-m0(x). Two modest tree settings, two logistic penalties. Select conversion model by treatment-arm validation log loss; causal settings by validation IPW policy value at 20%. Train-only category pooling/encoding and scaling. Common outcome nuisance models are the selected T-learner, fitted on train rows only.

**Policies.** Prespecified budgets 5%, 10%, 20%, 50%; plot a 0–100% curve. Exactly floor(q*n) users selected within an evaluation batch, descending score with deterministic label-blind tie breaking. Negative estimated effects remain eligible when enforcing exact capacity. Separately report up-to-budget causal policies that additionally require score > 0. Random benchmark is the expectation over uniform fixed-size audiences, pi_i=floor(q*n)/n. No-targeting value is zero.

**Evaluation.** Primary IPW score T*Y/e-(1-T)*Y/(1-e); robustness DR score m1-m0+T*(Y-m1)/e-(1-T)*(Y-m0)/(1-e). Evaluate mean pi*score. Paired differences use the same rows and score with pi_A-pi_B. Report normal 95% intervals from observation-level score variance, with no multiplicity-adjusted discovery claims outside the one primary contrast. Intervals condition on fitted models, nuisance estimates and the realized label-blind audience cutoffs; they omit retraining and cutoff-estimation uncertainty. Independence is a benchmark approximation because cluster identifiers are absent. They are not advertiser-cluster-robust intervals.

**Secondary metrics.** AUUC = integral_0^1 V(q)dq; Qini area = integral_0^1 [V(q)-q V(1)]dq. Integrate every empirical ranking prefix (ties resolved deterministically), with probability units and per-100,000 versions; no oracle or perfect-model normalization. Also treatment-arm ROC AUC, average precision, log loss, Brier score, and score-decile assignment-effect diagnostics.

**Robustness.** Three prespecified training-subset seeds at 500,000 rows, plus sample-size sensitivity at 200,000/500,000/2,000,000. These checks use validation only and are not used to select a favorable test result. Secondary visit analysis evaluates the conversion-trained policies; it does not retune the primary analysis. Hypothetical cost sensitivity uses net value = conversion_value*V - cost_per_assigned_user*q; no real revenue or ROI.

**Decision rule.** Report the held-out estimate and interval even if inconclusive or unfavorable. Deployment requires a new prospective policy experiment, real costs, privacy review, and monitoring. No subgroup significance search, individual-effect RMSE, invented feature meanings, or demographic fairness claims.
