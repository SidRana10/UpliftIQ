# Final Held-Out Evaluation (Stage 5)

**Frozen policy: Men's E-Mail for all.** Evaluated once on **12,800 held-out customers** that were not used for any fitting, tuning or selection.

**Primary result (visit rate, AIPW):** Men's E-Mail for all yields an expected visit rate of
18.19% [17.10%, 19.33%]. Against the strongest alternative
fixed baseline chosen on development data (Women's E-Mail for all), the difference is
+2.94 pp [+1.44 pp, +4.44 pp]
(Holm p = 0.0002), which excludes zero.

Outcome roles: **visit is the primary, pre-specified outcome. Conversion and spend are exploratory**;
conversions are rare, spend is 99% zeros, and no multiplicity control spans outcomes.

## 1. Primary outcome: visit rate (modeled policy estimates)

Expected visit rate if every held-out customer followed each policy. IPW needs no outcome model;
AIPW adds out-of-sample outcome predictions from a T-learner fitted on development data only. Intervals:
95% percentile bootstrap over customers (2000 resamples).

| Policy                 | AIPW                    | IPW                     |
|:-----------------------|:------------------------|:------------------------|
| Frozen policy          | 18.19% [17.10%, 19.33%] | 18.26% [17.06%, 19.48%] |
| No E-Mail for all      | 10.55% [9.65%, 11.49%]  | 10.59% [9.66%, 11.53%]  |
| Men's E-Mail for all   | 18.19% [17.10%, 19.33%] | 18.26% [17.06%, 19.48%] |
| Women's E-Mail for all | 15.26% [14.26%, 16.31%] | 15.16% [14.06%, 16.31%] |
| Uniform random         | 14.67% [14.09%, 15.30%] | 14.67% [14.09%, 15.30%] |

Paired comparisons (same customers; Holm adjustment across the baselines compared):

| Frozen policy minus              | AIPW                                                     | IPW                                                      |
|:---------------------------------|:---------------------------------------------------------|:---------------------------------------------------------|
| No E-Mail for all                | +7.64 pp [+6.15 pp, +9.07 pp], p=<0.0001, Holm p=<0.0001 | +7.66 pp [+6.07 pp, +9.21 pp], p=<0.0001, Holm p=<0.0001 |
| Women's E-Mail for all (primary) | +2.94 pp [+1.44 pp, +4.44 pp], p=0.0002, Holm p=0.0002   | +3.09 pp [+1.45 pp, +4.76 pp], p=0.0005, Holm p=0.0005   |
| Uniform random                   | +3.53 pp [+2.64 pp, +4.40 pp], p=<0.0001, Holm p=<0.0001 | +3.59 pp [+2.61 pp, +4.55 pp], p=<0.0001, Holm p=<0.0001 |

Plot: `figures/primary_visit_policy_values.png`.

## 2. Secondary outcomes (exploratory)

### Conversion rate

| Policy                 | AIPW                 | IPW                  |
|:-----------------------|:---------------------|:---------------------|
| Frozen policy          | 0.91% [0.64%, 1.20%] | 0.91% [0.63%, 1.20%] |
| No E-Mail for all      | 0.54% [0.34%, 0.78%] | 0.54% [0.33%, 0.77%] |
| Men's E-Mail for all   | 0.91% [0.64%, 1.20%] | 0.91% [0.63%, 1.20%] |
| Women's E-Mail for all | 0.87% [0.61%, 1.17%] | 0.87% [0.61%, 1.17%] |
| Uniform random         | 0.78% [0.63%, 0.93%] | 0.77% [0.62%, 0.92%] |

| Frozen policy minus              | AIPW                                                   | IPW                                                    |
|:---------------------------------|:-------------------------------------------------------|:-------------------------------------------------------|
| No E-Mail for all                | +0.38 pp [-0.00 pp, +0.72 pp], p=0.0414, Holm p=0.1241 | +0.38 pp [+0.00 pp, +0.73 pp], p=0.0421, Holm p=0.1264 |
| Women's E-Mail for all (primary) | +0.04 pp [-0.37 pp, +0.43 pp], p=0.8334, Holm p=0.8334 | +0.05 pp [-0.38 pp, +0.45 pp], p=0.8186, Holm p=0.8186 |
| Uniform random                   | +0.14 pp [-0.08 pp, +0.36 pp], p=0.2233, Holm p=0.4465 | +0.14 pp [-0.08 pp, +0.36 pp], p=0.2207, Holm p=0.4413 |

### Mean spend per assigned customer (zeros included)

Outcome model for AIPW: per-arm LightGBM Tweedie regressors (fixed, untuned, fitted on development data).

| Policy                 | AIPW                    | IPW                     |
|:-----------------------|:------------------------|:------------------------|
| Frozen policy          | $1.156 [$0.676, $1.681] | $1.169 [$0.687, $1.696] |
| No E-Mail for all      | $0.527 [$0.284, $0.823] | $0.526 [$0.285, $0.819] |
| Men's E-Mail for all   | $1.156 [$0.676, $1.681] | $1.169 [$0.687, $1.696] |
| Women's E-Mail for all | $1.265 [$0.777, $1.834] | $1.260 [$0.774, $1.832] |
| Uniform random         | $0.982 [$0.734, $1.257] | $0.985 [$0.738, $1.258] |

| Frozen policy minus              | AIPW                                                | IPW                                                 |
|:---------------------------------|:----------------------------------------------------|:----------------------------------------------------|
| No E-Mail for all                | +$0.629 [+$0.059, +$1.206], p=0.0289, Holm p=0.0866 | +$0.643 [+$0.072, +$1.218], p=0.0257, Holm p=0.0771 |
| Women's E-Mail for all (primary) | -$0.109 [-$0.834, +$0.607], p=0.7633, Holm p=0.7633 | -$0.091 [-$0.820, +$0.630], p=0.8030, Holm p=0.8030 |
| Uniform random                   | +$0.173 [-$0.202, +$0.571], p=0.3736, Holm p=0.7472 | +$0.184 [-$0.197, +$0.586], p=0.3449, Holm p=0.6898 |

Plot: `figures/secondary_policy_values.png`. Spend is heavily right-skewed, so its intervals are wide;
treat these results as descriptive.

## 3. Observed randomized outcomes (no model)

Raw held-out outcomes by assigned arm, and the randomized contrasts versus No E-Mail with the
Milestone 1 methods (Newcombe/permutation/bootstrap; Holm across six tests). These are observed
experimental results, not model-based policy estimates; the Men's-for-all policy value above is
essentially the Men's arm's observed mean.

| arm           |    n |   visit_rate |   conversion_rate |   mean_spend_per_customer |   share_zero_spend |
|:--------------|-----:|-------------:|------------------:|--------------------------:|-------------------:|
| No E-Mail     | 4261 |        10.61 |              0.54 |                     0.527 |              99.46 |
| Mens E-Mail   | 4262 |        18.28 |              0.92 |                     1.171 |              99.08 |
| Womens E-Mail | 4277 |        15.13 |              0.87 |                     1.257 |              99.13 |

| comparison                 | outcome    |   abs_effect |   abs_ci_low |   abs_ci_high |   p_value_holm |
|:---------------------------|:-----------|-------------:|-------------:|--------------:|---------------:|
| Mens E-Mail vs No E-Mail   | visit      |       0.0767 |       0.0619 |        0.0915 |         0      |
| Mens E-Mail vs No E-Mail   | conversion |       0.0038 |       0.0001 |        0.0075 |         0.083  |
| Mens E-Mail vs No E-Mail   | spend      |       0.6438 |       0.1051 |        1.2066 |         0.0697 |
| Womens E-Mail vs No E-Mail | visit      |       0.0452 |       0.031  |        0.0594 |         0      |
| Womens E-Mail vs No E-Mail | conversion |       0.0033 |      -0.0003 |        0.007  |         0.083  |
| Womens E-Mail vs No E-Mail | spend      |       0.73   |       0.1808 |        1.3233 |         0.0412 |

Plot: `figures/observed_outcomes.png`.

## 4. What was frozen and what was fitted

- The policy, seed, split and evaluation settings come from `reports/model_selection/frozen_config.json`,
  verified (checksum, split fingerprint, committed file) before scoring. Nothing was retuned.
- The test rows were used only to compute predictions and to score outcomes. Every fitted component
  (preprocessing, AIPW outcome models) was fitted on the 51,200 development rows.
- The held-out evaluation was run once; results were not revised afterwards.

## 5. What this does and does not show

- It estimates how well a fixed "Men's e-mail for everyone" rule performs on new customers from the same
  experiment, relative to sending nothing, Women's e-mail or random assignment.
- It does **not** show that personalized machine learning improves targeting. In development cross-validation the best personalized model (t_learner) differed from Men's-for-all by +0.12 pp [-0.24, +0.49] pp, an interval that includes zero. No
  personalized model was evaluated on the test set, and none was selected.

## 6. Limitations

- One 2008 experiment, two-week outcomes; no costs or margins. Visit is a proxy for revenue.
- Test-set intervals reflect customer sampling only. About 12,800 customers resolve policy
  differences of roughly 1 percentage point on visit; conversion and spend are much noisier.
- Policy comparisons here are for fixed rules and are not a Men's-vs-Women's causal heterogeneity analysis.
