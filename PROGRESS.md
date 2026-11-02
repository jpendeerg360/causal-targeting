# Progress

- 2026-11-02: Inspected empty workspace and ancestor paths; no AGENTS.md found.
- Read official dataset page, erratum, 2018 paper, 2021 corrected-release paper and current license.
- Old go.criteo.net link returned HTTP 404; Azure hostname failed DNS. Located Criteo-owned Hugging Face mirror, pinned commit 2424920019e49d52d72c13ac1143ec5d53af276b.
- Prespecified primary budget/comparison, split procedure, modeling budget and inference in docs/research_plan.md before empirical modeling.
- Downloaded 311,422,618 bytes; SHA-256 matched publisher Git LFS metadata. Validated all 13,979,592 rows, 16 source fields, binary indicators, finite features and structural exposure/visit constraints.
- Built eight substantive SQL queries, deterministic 60/10/30 partitions and Parquet intermediates. Exact-quantile balance computation initially hit the 4 GB DuckDB cap; switched only those descriptive quantiles to bounded-memory approximations.
- Development sizes: 200,000-row pilot; 500,000-row tuning; 2,000,000-row final training; 500,000-row validation; 4,193,056-row test. No test-dependent tuning decisions.
- Selected 15-leaf/100-iteration trees and logistic C=0.1. Ran three training-subset seeds, size sensitivity and assignment-predictability diagnostics. Model/config/plan hashes frozen at 22:55:46 UTC, before final-test evaluation at 22:56:19 UTC.
- Executed final evaluation. At 20%, tree uplift minus conversion = -15.53 [-20.50, -10.57] incremental conversions per 100,000; DR = -13.30 [-18.37, -8.23]. Tree versus random = 72.24 [63.37, 81.11]. Kept the negative model-comparison result.
- Feature-dependent propensity sensitivity changed absolute values materially but retained the negative primary contrast. Reported that limitation prominently; no ROI or original-population effect claim.
- Generated CSV tables, five PNG/SVG/PDF figure sets, README, research report, source/license record, interview guide and two empirical résumé bullets. Visually inspected all figures and adjusted crowded labels.
- Independent verification caught a prediction-export column collision (conversion score versus observed conversion). Fixed the export names, restored labels by source-row join, and added a regression test. Reported estimates were computed before export and unchanged. Independent SQL reproduced the primary point estimate and paired SE from the repaired row-level artifact.
- 22 targeted tests and 25 artifact checks pass. Original software compiles; editable installation and CLI reporting/verification run successfully. Git ignore rules exclude raw, processed and row-level artifacts.
- Fresh-copy full reproduction completed: re-ingestion, model fitting, evaluation, reports and 25 artifact checks passed. All four model files were byte-identical, and 20 aggregate tables matched within 1e-10 tolerance. This repeated the same fixed analysis/test split, not a new experiment. Recorded results/reproduction_check.json.

No external access blocker remains. No remote repository was published. Scientific limitations are retained-sample identification, unknown negative-label sampling, absent cluster/time keys, rare outcomes and uncertainty conditional on the fitted policies.

- Packaged the Git-ready source and aggregate deliverables. Large downloads, processed records, model files and execution caches were retained in workspace scratch storage outside the deliverable; the reproduction command recreates them.
