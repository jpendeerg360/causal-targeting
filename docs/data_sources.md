# Data provenance and interpretation

Verified on 2026-11-02. This is a noncommercial research benchmark project.

| Source | What it establishes |
|---|---|
| [Criteo's official resource and erratum](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | Corrected release: 13,979,592 rows; original advertiser leakage; current CC BY-NC-SA 4.0 terms |
| [Diemert et al. (2018), original paper](https://lig-aptikal.imag.fr/~amini/Publis/large-scale-benchmark.pdf) | Original incrementality experiments and why features were logged before intervention |
| [Diemert et al. (2021), corrected-release paper](https://arxiv.org/abs/2111.10106), §3 and Appendix A | Advertiser trials were resampled to a common 85% treatment ratio; four continuous/eight categorical variables; negative-label sampling |
| [Criteo-owned Hugging Face repository](https://huggingface.co/datasets/criteo/criteo-uplift/tree/2424920019e49d52d72c13ac1143ec5d53af276b) | Publisher-hosted corrected v2.1 file and Git LFS SHA-256 |
| [Current license](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Attribution, noncommercial use, share-alike and no warranty |

The original endpoint linked under the erratum, `http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`, returned HTTP 404 in this environment (HTTP and HTTPS). The web reader resolved it to an Azure blob endpoint whose hostname was unavailable locally. We therefore used the file in **Criteo's own organization**, pinned to commit `2424920019e49d52d72c13ac1143ec5d53af276b`. This is not a third-party repackaging.

The 311,422,618-byte compressed file has SHA-256:

```
2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc
```

The pipeline checks that checksum, the complete header and the corrected row count. A matching filename alone is insufficient. No publicly signed checksum was found; the checksum matches publisher repository metadata and records content identity, not an independent audit of Criteo's correction. The version is called v2 in the paper and v2.1 in the filename.

The actual header is `f0,...,f11,treatment,conversion,visit,exposure`. The official landing page's introductory feature count and original-release statistics are stale; the file, corrected erratum and updated paper agree on 12 features and the corrected size. No user, advertiser, experiment or timestamp field exists. `row_id` is generated from source position for reproducibility, not identity.

The experiments assigned browser users to advertising eligibility or suppression. Eligibility did not guarantee an ad impression. Features preceded assignment; visit/conversion outcomes covered two weeks. Actual exposure is post-assignment and excluded from all model covariates and policy scores. Adjusting for it would change the estimand and can induce selection bias. These design facts are described in the [2018 paper](https://lig-aptikal.imag.fr/~amini/Publis/large-scale-benchmark.pdf).

The [updated paper](https://arxiv.org/abs/2111.10106) describes balancing advertiser experiments to a common treatment ratio, supporting a constant 0.85 benchmark propensity. It also describes undisclosed label-dependent sampling. Our interpretation is that these are distinct issues: equalizing allocation addresses advertiser confounding, but does not supply the selection probabilities needed to reconstruct the source customer population. We therefore state the extra retained-sample exchangeability assumption, show empirical-propensity and learned-propensity sensitivity, and avoid original-population or advertiser ROI claims. A weak assignment classifier cannot prove the assumption.

All feature values are anonymous. The categorical indices follow Table 2 of the updated paper; numerical-looking category tokens are pooled and encoded as categories. No business or demographic meaning is assigned to them. Grouping, clustered standard errors and chronological evaluation cannot be recovered from the released columns.

The 2021 appendix's short license description omits “NonCommercial”; the current official terms and publisher mirror specify **CC BY-NC-SA 4.0**, which this project follows. Raw and row-level derived data, model files and prediction files are excluded from Git and the distribution archive. See [DATA_LICENSE.md](../DATA_LICENSE.md). Aggregated analysis products are modified/derived research outputs; they are provided under the same CC terms, with source attribution and no endorsement.

```bibtex
@inproceedings{diemert2018,
  author = {Diemert, Eustache and Betlei, Artem and Renaudin, Christophe and Amini, Massih-Reza},
  title = {A Large Scale Benchmark for Uplift Modeling},
  booktitle = {AdKDD and TargetAd Workshop},
  year = {2018}
}
@article{diemert2021,
  author = {Diemert, Eustache and Betlei, Artem and Renaudin, Christophe and Amini, Massih-Reza and Gregoir, Theophane and Rahier, Thibaud},
  title = {A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling},
  journal = {arXiv:2111.10106},
  year = {2021}
}
```
