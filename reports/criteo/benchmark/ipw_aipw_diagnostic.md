# Criteo C5 diagnostic: why IPW and AIPW disagree on the treat-all effect

Scope: 500,000 validation rows (the C5 500K validation subset), cached independent nuisance
predictions (LightGBM T-learner trained on 500,000 separate train rows), propensity
0.85/0.15. Nothing was retrained, no propensity model was fitted, the test set was not
touched, and the original C5 artifacts are unchanged. Effects are percentage points of visit rate.

## Verdict
- **No implementation error found.** Independent NumPy formulas reproduce the library IPW and AIPW
  per-customer scores to 0.0e+00
  (tolerance 1e-12: met); the weights sum to 1 in both arms, so normalization is not the issue.
- **The gap is statistical/assumption-related, not arithmetic.** IPW (+1.100 pp) equals the plain
  treated-minus-control difference; AIPW (+0.808 pp) corrects it using independently predicted baselines.
  IPW minus AIPW = 0.292 (SE 0.046; 95% [0.201, 0.378]) pp, clearly nonzero on the same observations.
- **Mechanism:** the treated and control arms differ in the predicted baseline visit rate
  (4.120% vs 3.848%, gap 0.272 pp,
  permutation z = 6.3, p = 0.0005). Treated customers were more likely to visit anyway, which inflates the
  unadjusted treated-minus-control contrast. That is evidence against an unconditional constant 0.85 propensity in this file.
  It cannot say why (the publisher's subsampling, hidden test structure, or something else).

## 1. Formula check
| quantity | value |
|:--|--:|
| mean(T/e), mean((1-T)/(1-e)) | 1.00000, 1.00000 |
| IPW, textbook formula (pp) | 1.0997 |
| IPW, normalized (Hajek) = treated mean - control mean (pp) | 1.0997 |
| AIPW, textbook formula (pp) | 0.8078 |
| max abs difference vs `binary_policy` scores, IPW / AIPW | 0.0e+00 / 0.0e+00 |
| AIPW = g-computation + treated augmentation - control augmentation (residual) | 0.0e+00 |

## 2. Decomposition of IPW - AIPW by arm (pp, paired SE)
Exact identity: IPW - AIPW = mean(mu1 (T/e - 1)) - mean(mu0 ((1-T)/(1-e) - 1)), residual -3.9e-18.

| term | estimate (SE; 95% bootstrap interval) |
|:--|:--|
| treated-arm term, mean(mu1 (T/e - 1)) | 0.061 (SE 0.008; 95% [0.046, 0.075]) |
| control-arm term, mean(mu0 ((1-T)/(1-e) - 1)) | -0.231 (SE 0.038; 95% [-0.302, -0.156]) |
| IPW - AIPW = treated term - control term | 0.292 (SE 0.046; 95% [0.201, 0.378]) |
| g-computation, mean(mu1 - mu0) | 0.739 (SE 0.004; 95% [0.731, 0.747]) |
| AIPW | 0.808 (SE 0.064; 95% [0.685, 0.936]) |
| IPW | 1.100 (SE 0.079; 95% [0.938, 1.259]) |

The control-arm term carries most of the gap: the control arm's predicted baseline is below the whole
population's (mean(mu0 | control) - mean(mu0) is the control term), so the control mean that IPW subtracts is too low and the
contrast too high. The treated arm's predicted outcome is slightly above the population's, which adds a smaller push in the
same direction. AIPW removes both with the outcome model. The interval is paired (same 500,000 customers, 1000 Poisson replicates).

## 3. Observed vs independently predicted rates by arm (%)
| | treated | control |
|:--|--:|--:|
| observed visit rate | 4.894 | 3.795 |
| predicted for own arm (calibration) | 4.879 | 3.848 |
| predicted control baseline mu0 | 4.120 | 3.848 |
| predicted baseline, whole population | 4.079 | |

The outcome model is calibrated within each arm (gaps 0.015 and -0.054 pp), so the arm difference in mu0
reflects different feature distributions, not model error. Permutation test (labels shuffled, predictions fixed, 2000 shuffles):
null SD 0.0429 pp, observed gap 0.2716 pp.
Largest feature imbalances on the validation subset:
- f3: SMD -0.051
- f6: SMD -0.047
- f1: SMD +0.029
- f5: SMD -0.028

## 4. Comparison with simpler adjusted estimates (treat-all effect, pp)
| estimator | estimate |
|:--|--:|
| unadjusted treated - control (= IPW) | 1.100 |
| g-computation (outcome model only) | 0.739 |
| post-stratified on deciles of predicted baseline, no propensity (SE 0.070) | 0.847 |
| AIPW | 0.808 |

Treated share and effects by decile of the independent predicted baseline:

|   stratum |     n |   mean_predicted_baseline |   treated_share |   rate_treated |   rate_control |   effect |
|----------:|------:|--------------------------:|----------------:|---------------:|---------------:|---------:|
|         1 | 46455 |                    0.0009 |          0.8482 |         0.0007 |         0.0001 |   0.0006 |
|         2 | 39454 |                    0.0011 |          0.8505 |         0.0009 |         0.0005 |   0.0004 |
|         3 | 63740 |                    0.0011 |          0.8487 |         0.0007 |         0.0005 |   0.0001 |
|         4 | 50108 |                    0.0014 |          0.8494 |         0.0012 |         0.0005 |   0.0007 |
|         5 | 49462 |                    0.0021 |          0.8487 |         0.0027 |         0.0023 |   0.0004 |
|         6 | 50501 |                    0.0037 |          0.8449 |         0.0047 |         0.0043 |   0.0004 |
|         7 | 50280 |                    0.0067 |          0.8462 |         0.0098 |         0.0075 |   0.0023 |
|         8 | 50000 |                    0.0152 |          0.8516 |         0.0203 |         0.017  |   0.0033 |
|         9 | 50000 |                    0.0513 |          0.8528 |         0.0727 |         0.0507 |   0.022  |
|        10 | 50000 |                    0.3243 |          0.8595 |         0.3712 |         0.3168 |   0.0544 |

The treated share rises with the predicted baseline, from 0.8482 in the lowest decile to 0.8595 in the highest (about 4.9 SE of a
difference of two shares), and the treatment effect is concentrated in the top decile. The treated arm therefore
over-represents the customers who both visit most and respond most, which is what inflates the unadjusted contrast.

The three adjusted estimates (0.74 to 0.85 pp) sit well below the unadjusted 1.10 pp and agree with each other within
their uncertainty (post-stratification vs AIPW differ by 0.04 pp, SE of post-stratified 0.07 pp). The AIPW implementation
therefore agrees with an independent, propensity-free adjustment. This supports AIPW *if* the treatment is as good as random
given f0-f11; imbalance on anything not in f0-f11 would remain, and nothing here can detect it.

## 5. What this means for C5 and C6
- Not a bug: do not change the evaluator or the stored results.
- Present AIPW as the primary estimate and IPW as an unadjusted check that is **expected to be upward-biased in this file**.
  Report the IPW-AIPW gap with every benchmark table, and describe gains as validation-sample estimates conditional on the
  outcome-model adjustment, not as population ROI.
- Model comparisons (paired, same nuisance, same estimator) are less exposed to the imbalance than absolute gains, but they
  inherit it: the imbalance is in the baseline, and uplift rankings are scored by AIPW differences between policies. They are
  safest as relative comparisons.
- Not tested here: hidden-group structure (row-order runs) as the source of the imbalance; creating group IDs from run
  boundaries remains out of scope. A propensity model or a sensitivity analysis would be the next tools, but were not used.
