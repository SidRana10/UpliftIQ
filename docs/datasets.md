# Datasets

## 1. Hillstrom E-Mail Analytics (original, in use)

- **Source:** Kevin Hillstrom, MineThatData E-Mail Analytics And Data Mining Challenge (2008-03-20)
- **URL:** http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- **Size:** 64,000 rows, 12 columns, ~3.9 MB
- **SHA-256:** `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`
- **Design:** randomized experiment, three arms (`Mens E-Mail`, `Womens E-Mail`, `No E-Mail`) with ~1/3 each.
- **Outcomes:** `visit`, `conversion`, `spend` (two weeks after the email).
- **Covariates:** `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`.
- **Note:** `zip_code` contains the value `Surburban` (sic) in the source file; we keep it as-is.
- **Fetch:** `python -m upliftiq.data.download` (writes `data/raw/hillstrom.csv` and `hillstrom.source.json`).
- **Validate:** `python -m upliftiq.data.validate`

## 2. Criteo Uplift Prediction Dataset, corrected v2.1 (downloaded and validated)

- **Source page (verified 2026-11-03):** https://ailab.criteo.com/criteo-uplift-prediction-dataset/
  (Criteo AI Lab). Paper: Diemert, Betlei, Renaudin, Amini, "A Large Scale Benchmark for Uplift Modeling",
  AdKDD 2018. Cite it when using the data.
- **Version:** the page's ERRATUM says the first release (25,309,483 rows, 459 MB) leaked information because
  the incrementality level differed across advertisers and features were advertiser-dependent. The
  **un-biased version (v2.1)** is the one we use.
- **Official link on the page:** http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz redirects to
  `criteostorage.blob.core.windows.net`, which no longer resolves (NXDOMAIN, checked 2026-11-03).
- **Download source we use:** the publisher-hosted Hugging Face repository `criteo/criteo-uplift`
  (https://huggingface.co/datasets/criteo/criteo-uplift), file
  `https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz`.
  The data remain the original Criteo AI Lab dataset (Diemert et al., AdKDD 2018); the Hugging Face repo is
  only the hosting location.
- **SHA-256 (enforced on download, pinned in `registry.py`):**
  `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`; file size 311,422,618 bytes
  (the page's "297M" is MiB). Provenance: `data/raw/criteo-uplift-v2.1.source.json` (URL, hash, version,
  license, timestamp; git-ignored with the data).
- **Key figures stated on the page for v2.1:** CSV, 297 MB compressed, 13,979,592 rows, average visit rate
  0.046992, average conversion rate 0.00292, treatment ratio 0.85.
- **License:** CC BY-NC-SA 4.0 (Criteo Data Terms of Use): non-commercial use only; adaptations must be shared
  alike under the same terms; attribution required.
- **Schema (comma-separated):** `f0`-`f11` (dense floats, anonymized and randomly projected), `treatment`
  (1 = treated, 0 = control), `conversion`, `visit` (binary labels), `exposure` (binary: whether the user was
  actually exposed). The page's prose still says "11 features" from the first release; the field list has twelve.
- **Fetch:** `python -m upliftiq.data.download --dataset criteo` (disk-space check, streamed, checksum enforced). **Validate / summarize (chunked, gzip streamed):**
  `python -m upliftiq.data.criteo`, which writes `reports/criteo/dataset_summary.{json,md}`.

### Validation results (2026-11-03; full report `reports/criteo/dataset_summary.md`)

13,979,592 rows, exactly the 12 features + `treatment`, `exposure`, `conversion`, `visit`; no missing or
non-finite values; all four binary columns strictly 0/1. Treated 85.00% (11,882,655) / control 15.00%
(2,096,937). Visit rate 4.699% (treated 4.854%, control 3.820%); conversion 0.292% (treated 0.309%,
control 0.194%), matching the figures on the Criteo page. No control user is exposed; 3.60% of treated users
are exposed, and their visit rate (41.5%) dwarfs that of unexposed treated users (3.5%), showing why
`exposure` must never be a feature. No conversion occurs without a visit. Several features are heavily
concentrated on a single value (e.g. f1, f2, f4, f10 have their median at the minimum).

### Limitations that matter for causal work

- **Treatment vs exposure.** `treatment` is the randomized assignment; `exposure` is a post-treatment
  outcome (non-compliance: not every treated user is exposed). Never use `exposure` as a feature. Analyses on
  `treatment` estimate intention-to-treat effects; effects of actual exposure need extra assumptions.
- **Non-uniform subsampling.** Criteo sub-sampled the data non-uniformly "so that the original incrementality
  level cannot be deduced". The data come from several incrementality tests (advertisers); the file has no
  test/advertiser id. Treatment ratio is about 0.85, nothing like Hillstrom's 1/3, and may vary across the
  unobserved tests. **We must not assume Hillstrom's propensity (1/3).** A constant propensity is only
  defensible if it is validated (for example, features balanced across arms); otherwise IPW/AIPW policy
  values need an estimated propensity, with the risk that the subsampling makes assignment depend on covariates.
- **Absolute levels are not interpretable.** Because of the subsampling, overall treatment/control rates and
  the base incrementality are not those of the original campaigns; use the data for relative model comparison.
- **Anonymized, randomly projected features** (no interpretation, no feature engineering from meaning);
  outcome windows and units are undocumented; conversions are rare (about 0.3%).
- **Possible residual bias:** the corrected release removes the advertiser leak, but covariate balance
  between treatment arms still has to be checked before any causal evaluation.
- Raw data and large processed files stay out of Git (`data/raw/` is ignored).
