# Hillstrom Model Selection (Stage 4, development data only)

**Selected policy: Men's E-Mail for all.** No personalized model earned selection under the pre-specified rule.

The best-valued personalized model (LightGBM T-learner) differs from Men's-for-all by +0.12 pp [-0.24, +0.49] pp. The interval includes zero, so the development data do **not** show that personalization beats sending the Men's e-mail to everyone.

Scope: 51,200 development customers (80% of Hillstrom), 5-fold stratified cross-validation, primary outcome
= visit. **The held-out 12,800-customer test set was not accessed.** Numbers below are out-of-fold
estimates on development data; the one-time test evaluation is a later step.

## 1. Policy values (out-of-fold, AIPW; IPW as a check)

| Policy                  | Config                                        |   AIPW visit rate % |   CI low |   CI high |   IPW check % |
|:------------------------|:----------------------------------------------|--------------------:|---------:|----------:|--------------:|
| Logistic (interactions) | logistic:C=0.01                               |               18.24 |    17.66 |     18.8  |         18.19 |
| LightGBM S-learner      | s_learner:num_leaves=4,min_child_samples=300  |               18.25 |    17.67 |     18.8  |         18.22 |
| LightGBM T-learner      | t_learner:num_leaves=4,min_child_samples=300  |               18.38 |    17.8  |     18.96 |         18.28 |
| DR-learner              | dr_learner:num_leaves=4,min_child_samples=300 |               18.21 |    17.64 |     18.74 |         18.13 |
| No E-Mail for all       | fixed baseline                                |               10.64 |    10.18 |     11.09 |         10.61 |
| Men's E-Mail for all    | fixed baseline                                |               18.26 |    17.67 |     18.81 |         18.25 |
| Women's E-Mail for all  | fixed baseline                                |               15.13 |    14.57 |     15.68 |         15.18 |
| Uniform random          | fixed baseline                                |               14.68 |    14.38 |     14.98 |         14.68 |

AIPW uses a fixed T-learner, cross-fitted so each customer's outcome prediction never comes from a model
trained on that customer. IPW needs no outcome model; agreement between the two columns is a sanity check.
Intervals are 95% percentile bootstrap intervals over customers (2,000
resamples, seed 42). Plot: `value_comparison.png`.

## 2. Paired differences (candidate minus baseline, 95% bootstrap CI)

Differences are computed on the same customers, so they are tighter than the overlap of the intervals above suggests.

| Model                   | vs No E-Mail for all       | vs Men's E-Mail for all    | vs Women's E-Mail for all   | vs Uniform random          |
|:------------------------|:---------------------------|:---------------------------|:----------------------------|:---------------------------|
| Logistic (interactions) | +7.60 pp [+6.89, +8.33] pp | -0.02 pp [-0.36, +0.34] pp | +3.11 pp [+2.40, +3.80] pp  | +3.57 pp [+3.14, +4.01] pp |
| LightGBM S-learner      | +7.61 pp [+6.87, +8.34] pp | -0.01 pp [-0.08, +0.06] pp | +3.12 pp [+2.32, +3.89] pp  | +3.57 pp [+3.10, +4.01] pp |
| LightGBM T-learner      | +7.74 pp [+7.03, +8.48] pp | +0.12 pp [-0.24, +0.49] pp | +3.25 pp [+2.55, +3.91] pp  | +3.70 pp [+3.26, +4.13] pp |
| DR-learner              | +7.57 pp [+6.85, +8.28] pp | -0.05 pp [-0.35, +0.25] pp | +3.08 pp [+2.35, +3.78] pp  | +3.53 pp [+3.10, +3.96] pp |

## 3. Selection

- Within each model family, the simplest hyperparameter setting within one paired standard error of the best
  was kept (`family_best` in `cv_results.csv`).
- Across options, the best by CV value was **LightGBM T-learner**; the rule
  then picks the simplest option within one paired SE of it (order: fixed baselines, logistic, S, T, DR).
- Personalized models below the best fixed baseline (Men's E-Mail for all) are ineligible:
  Logistic (interactions), LightGBM S-learner, DR-learner.
- Result: **Men's E-Mail for all**.

Full grid (policy = family:hyperparameters; shares are the fraction of customers recommended each arm):

| policy                                         |   aipw_value |   aipw_se |   ipw_value |   share_no_email |   share_mens |   share_womens | family_best   |
|:-----------------------------------------------|-------------:|----------:|------------:|-----------------:|-------------:|---------------:|:--------------|
| logistic:C=0.01                                |       0.1824 |    0.0029 |      0.1819 |           0      |       0.776  |         0.224  | True          |
| logistic:C=0.1                                 |       0.1831 |    0.0029 |      0.1828 |           0      |       0.7171 |         0.2829 | False         |
| logistic:C=1.0                                 |       0.1831 |    0.0029 |      0.183  |           0      |       0.7184 |         0.2816 | False         |
| logistic:C=10.0                                |       0.1833 |    0.0029 |      0.1832 |           0      |       0.7185 |         0.2815 | False         |
| s_learner:num_leaves=4,min_child_samples=100   |       0.1828 |    0.0029 |      0.1828 |           0      |       0.9931 |         0.0069 | False         |
| s_learner:num_leaves=4,min_child_samples=300   |       0.1825 |    0.0029 |      0.1822 |           0      |       0.9902 |         0.0098 | True          |
| s_learner:num_leaves=8,min_child_samples=100   |       0.1803 |    0.0029 |      0.1801 |           0      |       0.8924 |         0.1076 | False         |
| s_learner:num_leaves=8,min_child_samples=300   |       0.1821 |    0.0029 |      0.1822 |           0      |       0.9001 |         0.0999 | False         |
| s_learner:num_leaves=16,min_child_samples=100  |       0.1792 |    0.0029 |      0.1789 |           0.0011 |       0.8542 |         0.1446 | False         |
| s_learner:num_leaves=16,min_child_samples=300  |       0.1812 |    0.0029 |      0.1794 |           0.0002 |       0.8673 |         0.1325 | False         |
| t_learner:num_leaves=4,min_child_samples=100   |       0.1834 |    0.0029 |      0.1828 |           0.0051 |       0.7573 |         0.2376 | False         |
| t_learner:num_leaves=4,min_child_samples=300   |       0.1838 |    0.0029 |      0.1828 |           0.0031 |       0.7546 |         0.2422 | True          |
| t_learner:num_leaves=8,min_child_samples=100   |       0.1809 |    0.0029 |      0.1799 |           0.0113 |       0.7454 |         0.2432 | False         |
| t_learner:num_leaves=8,min_child_samples=300   |       0.1803 |    0.0029 |      0.1799 |           0.0065 |       0.7454 |         0.2481 | False         |
| t_learner:num_leaves=16,min_child_samples=100  |       0.1797 |    0.0029 |      0.1781 |           0.0139 |       0.7427 |         0.2433 | False         |
| t_learner:num_leaves=16,min_child_samples=300  |       0.1811 |    0.0029 |      0.1798 |           0.0107 |       0.7407 |         0.2486 | False         |
| dr_learner:num_leaves=4,min_child_samples=100  |       0.1813 |    0.0029 |      0.1802 |           0.0001 |       0.8555 |         0.1445 | False         |
| dr_learner:num_leaves=4,min_child_samples=300  |       0.1821 |    0.0029 |      0.1813 |           0      |       0.8512 |         0.1488 | True          |
| dr_learner:num_leaves=8,min_child_samples=100  |       0.18   |    0.0029 |      0.1794 |           0      |       0.8914 |         0.1086 | False         |
| dr_learner:num_leaves=8,min_child_samples=300  |       0.1812 |    0.0029 |      0.1809 |           0      |       0.8852 |         0.1148 | False         |
| dr_learner:num_leaves=16,min_child_samples=100 |       0.1808 |    0.0029 |      0.1808 |           0      |       0.9429 |         0.0571 | False         |
| dr_learner:num_leaves=16,min_child_samples=300 |       0.1815 |    0.0029 |      0.181  |           0      |       0.8985 |         0.1015 | False         |

## 4. Caveats

- Hyperparameters were chosen on the same out-of-fold data used to compare models, so the reported CV values
  of tuned models are slightly optimistic (22 settings were tried). The untouched test set is the unbiased check.
- Bootstrap intervals reflect sampling variability of customers, not the variability from refitting
  models; per-fold values are in `cv_results.csv`.
- Visit is a proxy: this stage does not evaluate revenue or conversion (planned later), and policies ignore
  e-mail cost.
- Qini curves and decile calibration from the protocol are not part of this stage's selection (the one-SE rule
  with a simplicity order makes them tie-breakers only); they are not reported here.
- A single two-week experiment from 2008 limits generalization.

## 5. Reproducibility

Frozen configuration: `frozen_config.json` (chosen policy, seed, split fingerprint
`c3ae8f2b921ec92d...`, evaluation settings, data checksum, git commit). Regenerate with
`python -m upliftiq.modeling.select`.
