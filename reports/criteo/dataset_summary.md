# Criteo Uplift v2.1: dataset summary

Validation: **PASSED**

- Rows: 13,979,592 (expected 13,979,592); missing values: 0
- Treatment allocation: treated 0.8500 (11,882,655), control 0.1500 (2,096,937)
- Overall visit rate 0.04699, conversion rate 0.00292
- Visit / conversion rate, treated: 0.04854 / 0.00309; control: 0.03820 / 0.00194
- Exposed share: treated 0.0360, control 0.0000
- Control users flagged exposed: 0; conversions without a visit: 0

These are observed descriptive rates, not causal estimates. The treated share is about 0.85, nothing like
Hillstrom's 1/3, and the file is a non-uniform subsample, so propensities must not be assumed.
`exposure` is post-treatment and must not be a feature.

## Features (quantiles from a 139,815-row sample)

|     |    mean |    std |      min |      q01 |     q50 |     q99 |     max |
|:----|--------:|-------:|---------:|---------:|--------:|--------:|--------:|
| f0  | 19.6203 | 5.3775 |  12.6164 |  12.6164 | 21.9224 | 26.674  | 26.7453 |
| f1  | 10.07   | 0.1048 |  10.0597 |  10.0597 | 10.0597 | 10.6795 | 16.3442 |
| f2  |  8.4466 | 0.2993 |   8.2144 |   8.2144 |  8.2144 |  9.0428 |  9.052  |
| f3  |  4.1789 | 1.3366 |  -8.3984 |  -1.7001 |  4.6799 |  4.6799 |  4.6799 |
| f4  | 10.3388 | 0.3433 |  10.2805 |  10.2805 | 10.2805 | 11.9733 | 21.1235 |
| f5  |  4.0285 | 0.4311 |  -9.0119 |   2.2309 |  4.1155 |  4.1155 |  4.1155 |
| f6  | -4.1554 | 4.5779 | -31.4298 | -17.7773 | -2.4111 |  0.2944 |  0.2944 |
| f7  |  5.1018 | 1.2052 |   4.8338 |   4.8338 |  4.8338 | 11.4728 | 11.9984 |
| f8  |  3.9336 | 0.0567 |   3.6351 |   3.7521 |  3.9719 |  3.9719 |  3.9719 |
| f9  | 16.0276 | 7.019  |  13.1901 |  13.1901 | 13.1901 | 44.8392 | 75.295  |
| f10 |  5.3334 | 0.1682 |   5.3004 |   5.3004 |  5.3004 |  6.333  |  6.4739 |
| f11 | -0.171  | 0.0228 |  -1.3839 |  -0.2674 | -0.1687 | -0.1687 | -0.1687 |

## Treatment x exposure

|   treatment |   exposure |                n |   visit_rate |   conversion_rate |
|------------:|-----------:|-----------------:|-------------:|------------------:|
|           0 |          0 |      2.09694e+06 |      0.0382  |           0.00194 |
|           1 |          0 |      1.14544e+07 |      0.03486 |           0.00119 |
|           1 |          1 | 428212           |      0.41454 |           0.05378 |
