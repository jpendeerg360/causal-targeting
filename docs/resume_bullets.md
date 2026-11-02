# Résumé bullets

- Built and executed a reproducible Python/DuckDB causal-targeting pipeline on **13,979,592 corrected Criteo records**, with validated Parquet ingestion, leakage-safe model fitting on **2,000,000 rows**, and evaluation on **4,193,056 held-out rows**.
- Compared conversion ranking, logistic treatment interactions and boosted-tree uplift policies using unequal-propensity IPW, doubly robust checks and paired confidence intervals; at 20% targeting, estimated tree uplift versus conversion ranking at **-15.5 [-20.5, -10.6] incremental conversions per 100,000 eligible users**, and versus random at **72.2 [63.4, 81.1]**.

Intervals are 95% CIs. These describe a research benchmark, not realized business revenue or a deployed campaign. If space is limited, retain the comparison and its uncertainty rather than replacing it with an unsupported uplift percentage.
