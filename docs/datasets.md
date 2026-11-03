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

## 2. Criteo Uplift Prediction Dataset, corrected (planned, NOT downloaded)

- **What:** Large-scale uplift dataset from Criteo AI Lab (~13.98M rows, 12 anonymized float features
  `f0`..`f11`, `treatment`, `conversion`, `visit`, and `exposure`). Roughly 85% treated.
- **Why "corrected":** the original release (v2.1) reportedly had issues with the way treatment and
  exposure were handled; we use the corrected version. **Before downloading, confirm the exact
  corrected-release URL, file name, license and checksum from the official Criteo AI Lab page and
  record them here.** They have not been verified yet.
- **License:** CC BY-NC-SA 4.0 for the original release (non-commercial); re-check for the corrected one.
- **Status:** intentionally not downloaded. It is large (hundreds of MB compressed), so download will be
  a separate, explicit step, likely with sampling support.
