# Causal Modeling and Evaluation Protocol (Milestone 2)

Status: design only, nothing implemented. Data: Hillstrom, 64,000 customers, randomized three-arm
experiment (`No E-Mail`, `Mens E-Mail`, `Womens E-Mail`, about 1/3 each). Milestone 1 findings that
shape this plan: both e-mails raise visits (+7.7 pp Men's, +4.5 pp Women's), conversions are rare
(0.9%), spend is 99% zeros and skewed, and covariates are balanced.

## 1. Decisions at a glance

| Topic | Decision |
|---|---|
| Goal | Learn a policy `pi(x) in {none, mens, womens}` that maximizes expected **visit rate**; judge it by held-out, randomization-based policy value. |
| Splits | 80/20 dev/test (stratified by arm x visit, seed 42). Test is touched once. Selection uses 5-fold stratified CV **inside dev** (out-of-fold predictions). |
| Features | `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`. |
| Models | (a) Logistic regression with treatment interactions, (b) LightGBM S-learner, (c) LightGBM T-learner, (d) challenger: **DR-learner** (doubly robust, cross-fitted). |
| Primary metric | Cross-validated **AIPW policy value** on visit using the known propensities (1/3 per arm). |
| Baselines | No e-mail, Men's for all, Women's for all, random assignment (1/3 each). |
| Winner | Pre-specified rule applied to dev CV results only (section 7). Config is frozen, then evaluated once on test. |

## 2. Data splitting, preprocessing, leakage prevention

- **Test split:** 20% (12,800 rows), stratified on `segment x visit`, fixed seed. It is never used for fitting,
  tuning, early stopping, thresholding or model choice. Everything before the final step uses dev (51,200 rows).
- **Train/validation:** repeated use of dev via 5-fold stratified CV (same stratification). Out-of-fold (OOF)
  predictions on dev serve as validation. This gives about 10,000 validation rows per fold instead of one
  small fixed validation set, which matters because uplift signals are weak and noisy. Folds are generated once
  and saved so every model sees identical splits.
- **Allowed inputs (pre-treatment only):** the seven features above. `history_segment` is a binned copy of
  `history` and is dropped as redundant. `mens`/`womens` are purchase history flags from the prior year, so they are
  legitimately pre-treatment.
- **Forbidden inputs:** `visit`, `conversion`, `spend` (post-treatment outcomes) and anything derived from them.
  `segment` is the treatment, used only as the arm indicator. A unit test asserts the feature list is exactly
  the allowed set.
- **Preprocessing is fit inside each training fold only** (never on validation or test rows): one-hot encoding
  for zip_code/channel, `log1p` and standardization of `history` and `recency` for logistic regression;
  native categorical handling for LightGBM (no scaling needed). Encoders are part of the model pipeline so
  CV cannot leak them.
- **Propensities are not modeled.** Randomization gives `e(a) = 1/3` for each arm by design. We use 1/3, not the
  empirical arm shares, which differ by only about 0.1 pp. The Milestone 1 balance table already supports this.

## 3. Three-arm treatment modeling

For every outcome `Y`, each model produces `mu_a(x) = E[Y | X=x, arm=a]` for `a in {0, m, w}`
(or the uplift directly for the DR-learner). Per-customer uplifts are
`tau_m(x) = mu_m - mu_0` and `tau_w(x) = mu_w - mu_0`.

Policy: `pi(x) = argmax over {0, m, w} of mu_a(x)`, i.e. send the e-mail with the largest predicted uplift
if it is positive, otherwise send nothing. Costs are unknown, so the base policy is cost-free. A budget view
(treat only the top `q` fraction of customers ranked by `max(tau_m, tau_w)`) is used for uplift curves
(section 5) and is the only place a "treat/don't treat" threshold appears.

## 4. Models

All models are trained per outcome, with the same folds. Hyperparameter grids are small and fixed in advance.

1. **Logistic regression baseline (S-learner with interactions).** One L2-penalized logistic model on
   `[x, arm one-hot, arm x x]`. Tune `C` in {0.01, 0.1, 1, 10}. Rationale: transparent, hard to beat for tiny
   effects, and a reality check for the tree models.
2. **LightGBM S-learner.** One classifier with the arm as a categorical feature; counterfactual predictions by
   setting the arm to each value. Risk: with weak effects the model may ignore the arm; we report this as a
   diagnostic (variance of `tau_hat`).
3. **LightGBM T-learner.** Three separate classifiers, one per arm. Flexible, but each uses only a third of the data.
4. **Challenger: DR-learner (doubly robust), with LightGBM.** Why this one and not a causal forest or
   X-learner: (i) the propensities are *known* and constant, so the doubly robust pseudo-outcome has no
   propensity-model error; (ii) it targets uplift directly instead of subtracting two noisy predictions
   (the main weakness of S and T); (iii) it extends cleanly to three arms and to any outcome type;
   (iv) it needs only LightGBM, with no heavy new dependency. Procedure per training fold:
   - cross-fit outcome models `mu_hat_a` (2 folds inside the training fold, so a row's own outcome never
     feeds its own pseudo-outcome);
   - for each `k in {m, w}`:
     `Gamma_k = mu_hat_k - mu_hat_0 + 1{A=k}/e * (Y - mu_hat_k) - 1{A=0}/e * (Y - mu_hat_0)`;
   - fit a strongly regularized LightGBM *regressor* of `Gamma_k` on `x` to get `tau_hat_k(x)`.
   Risk: pseudo-outcomes are high variance, so regularization (shallow trees, large `min_child_samples`) is
   essential. It may not beat the simpler models, and that is an acceptable outcome.

LightGBM grids (visit): `num_leaves` {4, 8, 16}, `min_child_samples` {100, 300}, `learning_rate` 0.05,
`n_estimators` chosen by early stopping on a held-out piece of the *training* fold, `seed=42`. Total search
is a few dozen fits, with no random search or Optuna.

## 5. Outcomes, metrics, diagnostics

**Primary: visit** (binary, 14.7%). Selection is based on visit only.

**Secondary: conversion and spend.** They are exploratory and not used for selection, because
conversions are rare (about 460 events in dev train folds) and spend is zero-inflated.
- *Policy-effect check (main secondary result):* the visit-selected policy is evaluated on conversion and
  spend with the same estimators. This answers "does optimizing visits also help revenue?".
- *Outcome-specific refit (only for the winning model family):* conversion with the same classifier setup;
  spend with LightGBM `tweedie` objective (S/T) or the DR regression on pseudo-outcomes. Same folds, fixed
  mid-sized hyperparameters (no tuning), reported with intervals and explicitly labeled low-power.
- Spend always includes zeros (per assigned customer), as in Milestone 1.

**Model-selection metrics (dev OOF, visit):**
1. **AIPW policy value of the model's policy** (primary). With nuisance `m_hat_a(x)`:
   `psi_i(pi) = m_hat_{pi(x_i)}(x_i) + 1{A_i = pi(x_i)}/e * (Y_i - m_hat_{pi(x_i)}(x_i))`, and `V(pi) = mean(psi_i)`.
   `m_hat` is a fixed, simple LightGBM T-learner used only as a variance-reducing control variate and fitted
   out-of-fold, never on the row being scored. It is the same for all candidates, so it cannot favor one. The
   unbiasedness of `V(pi)` comes from the known propensities, not from `m_hat`. The pure IPW value
   (`m_hat = 0`) is reported as a sensitivity check.
2. **Qini / uplift curve (budget view):** rank customers by `max(tau_m, tau_w)`, treat the top `q` with their
   best arm, none otherwise; the gain over no e-mail is estimated with IPW/AIPW. Report the area between the curve and the
   random line (Qini coefficient) and gain at q = 20/40/60%.
3. **Calibration by decile:** per decile of predicted uplift, mean predicted vs observed (treated minus control)
   uplift with CIs. Diagnostic, not a selection criterion.
4. Outcome-model log-loss and AUC per arm are reported as sanity diagnostics only: good outcome prediction does
   not imply good uplift ranking.

**Uncertainty (all metrics are means of per-customer scores, so this is cheap and valid):**
- Policy value and differences between two policies (paired, using `psi_i(pi_1) - psi_i(pi_2)` over the same
  customers): normal-approximation 95% CI plus a nonparametric bootstrap over customers (2,000 resamples, seed
  42) as a check. Bootstrap is over customers, not folds, so model-fold variability is acknowledged separately by
  reporting per-fold values.
- Qini and decile curves: bootstrap bands.
- Rough scale: with 12,800 test customers a single policy's value has a 95% CI of about +/-1 pp for visit rate
  (IPW; AIPW and paired differences are tighter). Expect to resolve policy differences of about 1 pp, not
  fractions of a point.

## 6. Baseline policies

Evaluated with the identical estimators, folds and test set:

| Baseline | Policy | Estimate |
|---|---|---|
| No e-mail | `pi = 0` | arm-0 visit rate (AIPW) |
| Men's for all | `pi = m` | arm-m value |
| Women's for all | `pi = w` | arm-w value |
| Random targeting | uniform over {0, m, w} (and, for curves, random ordering) | mean of the three arm values; random line in Qini plots |

"Men's for all" is the baseline that matters: Milestone 1 shows Men's beats Women's on average, so a model only
adds value if it beats that, not merely no e-mail. A model that does not beat Men's-for-all is a legitimate result.

## 7. Winner selection without touching test

1. Hyperparameters of each family are chosen by the *highest mean CV AIPW policy value* on dev OOF
   predictions, using the fixed grids. Ties and near ties go to the simpler setting.
2. Across families, pick the model with the highest CV policy value, but apply a **one-standard-error rule**: if a
   simpler model (order: logistic < S < T < DR) is within one paired SE of the best, choose the simpler. Qini
   coefficient and calibration are the tie-break and sanity checks. A model with a policy value below the best
   baseline's CV value is not eligible to be "the winner" (the recommendation is then the baseline).
3. Refit the chosen model on all of dev with the chosen hyperparameters.
4. **Freeze** model, hyperparameters, seed, policy rule and the list of analyses in `reports/model_selection/
   frozen_config.json` (with the git commit hash and a hash of the dev data file) *before* scoring test.
5. Score test **once**. The primary test hypothesis is single and pre-specified: winner's policy value minus the
   best baseline's value on test (paired, 95% CI). Everything else on test is descriptive; if more than one
   hypothesis is reported, Holm adjustment is applied as in Milestone 1.
6. If a bug is later found, fix it and state that the test set has been used and the result is no longer a
   clean holdout (no silent re-runs and no re-tuning on test).

## 8. Implementation stages and acceptance criteria

| Stage | Work | Acceptance criteria |
|---|---|---|
| 0. Environment | Add `scikit-learn`, `lightgbm`; verify LightGBM imports on this Mac (libomp). | `import lightgbm` works; pytest and Ruff still pass; no heavy extras (no EconML, Optuna). |
| 1. Split & features | `modeling/split.py`, `modeling/features.py`: test split, saved CV folds, feature whitelist, fold-safe preprocessing. | Same seed gives identical split; stratification holds (arm and visit rates match within 0.1 pp); no row in two splits; feature list equals the whitelist; test rows never reach any fit function (unit test). |
| 2. Policy evaluator | `evaluation/policy.py`: IPW/AIPW values, paired differences, bootstrap CIs, Qini curve, baselines. | On synthetic data with known effects, estimates match the truth within the CI (coverage about 95% over repeated simulations); in the real data, no-email/Men's/Women's values equal the Milestone 1 arm rates; A/A check: shuffling arm labels gives mean policy gain over baseline near 0 with CI containing 0. |
| 3. Learners | `modeling/learners.py`: logistic, S, T, DR with a common `fit/predict_uplift` interface for 3 arms. | Each recovers a known planted uplift on synthetic data (rank correlation clearly above 0); predictions are deterministic given the seed; shapes and edge cases tested; DR cross-fitting verified not to use a row's own outcome. |
| 4. CV selection | `modeling/select.py`: run CV, tune fixed grids, produce selection table, curves, calibration, `frozen_config.json`. Visit only. | One command reproduces the table; each model reports CV policy value +/- CI, Qini and per-fold values; all five policies (4 models + baselines) compared on identical OOF rows; winner chosen by the written rule; test data never loaded in this stage (enforced by the code path and a test). |
| 5. Test evaluation | Single run on the held-out test set with the frozen model. | Config hash checked before scoring; output holds the primary paired comparison with 95% CI plus descriptive baselines; the run can only execute once per frozen config (guard). |
| 6. Secondary outcomes | Evaluate the chosen policy on conversion and spend; optional outcome-specific refit for the winning family only. | Reported with CIs and labeled exploratory; no model re-selection based on these. |
| 7. Report | `reports/model_selection/report.md` (plain-language, same style as Milestone 1), figures, tables. | States the winner, whether it beats Men's-for-all (with the CI), limitations, and what was and was not tuned; all figures reproducible from one command. |

## 9. Key tradeoffs and limitations

- **CV on dev vs a single fixed validation set:** CV is lower-variance and uses data better; costs more compute (tiny here).
- **80/20 test:** 12,800 customers can only resolve about 1 pp policy differences. A bigger test would shrink
  development data for already-weak signals. If heterogeneity is small, the honest result may be "no model beats
  Men's for everyone".
- **Visit as the primary outcome:** it has about 10,000 events (adequate power) but is not revenue. The secondary
  check guards against optimizing a proxy that does not pay.
- **AIPW nuisance model:** reduces variance but could be questioned; IPW is always reported alongside.
- **DR-learner variance:** the advanced model might lose to simple ones; the one-SE rule is intended to favor the simpler choice.
- **No cost model:** policies ignore e-mail cost and margin; the budget curve is the only cost-aware view.
- **One historical, two-week experiment:** results do not extrapolate to other campaigns or horizons.
- **Out of scope here:** API, frontend, Criteo, feature engineering beyond the raw covariates, deep or
  neural uplift models, and hyperparameter search frameworks.

## 10. Clarifications (added at Stage 0/1 approval)

- **AIPW nuisance models must be out of sample.** The outcome model `m_hat` used inside `psi_i` is always fitted
  on rows that exclude row `i` (CV: fitted on the fold's training rows; test: fitted on dev only). Evaluating a
  policy with an `m_hat` fitted on the scored rows is not allowed, and Stage 2 must include a test for it.
- **Contact-budget curves are not monetary cost analysis.** The Qini/uplift curve (treat the top `q` fraction)
  limits the *number of contacts*; it assumes nothing about e-mail cost or margin. Any dollar-based statement
  (net revenue, ROI, break-even uplift) requires explicit cost and margin assumptions and is a separate,
  clearly labeled sensitivity analysis, never mixed into the budget-curve results.
