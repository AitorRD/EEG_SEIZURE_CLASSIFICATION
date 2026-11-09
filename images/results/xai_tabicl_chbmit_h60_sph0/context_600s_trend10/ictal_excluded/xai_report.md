# XAI — TabICL — LOSO seizure prediction — CHB-MIT

EEG seen per prediction 600 s · history trend10 · ictal windows excluded · 23 folds · SHAP on up to 40 preictal + 40 interictal test windows per fold. Share = mean |SHAP| of a column as % of the fold total, averaged over folds (a column not selected in a fold counts 0 there).

## Top 20 columns

| column                                                        |   share % |
|:--------------------------------------------------------------|----------:|
| P7-T7__permutation_entropy__dimension_7__tau_1  (20/23 folds) |      1.28 |
| P7-T7__permutation_entropy__dimension_6__tau_1  (19/23 folds) |      1.11 |
| FT10-T8__ar_coefficient__coeff_3__k_10  (19/23 folds)         |      1.08 |
| P7-T7__permutation_entropy__dimension_5__tau_1  (20/23 folds) |      1.07 |
| CZ-PZ__permutation_entropy__dimension_7__tau_1  (10/23 folds) |      1.04 |
| P7-T7__permutation_entropy__dimension_3__tau_1  (16/23 folds) |      1    |
| T8-P8-1__partial_autocorrelation__lag_4  (23/23 folds)        |      0.96 |
| CZ-PZ__permutation_entropy__dimension_6__tau_1  (11/23 folds) |      0.94 |
| P7-T7__permutation_entropy__dimension_4__tau_1  (19/23 folds) |      0.93 |
| FP2-F4__ar_coefficient__coeff_2__k_10  (17/23 folds)          |      0.84 |
| F4-C4__partial_autocorrelation__lag_3  (14/23 folds)          |      0.83 |
| FP2-F4__ar_coefficient__coeff_4__k_10  (19/23 folds)          |      0.81 |
| F7-T7__ar_coefficient__coeff_3__k_10  (19/23 folds)           |      0.81 |
| FP1-F7__ar_coefficient__coeff_3__k_10  (19/23 folds)          |      0.8  |
| FZ-CZ__partial_autocorrelation__lag_4  (21/23 folds)          |      0.78 |
| CZ-PZ__permutation_entropy__dimension_5__tau_1  (11/23 folds) |      0.75 |
| T8-P8-1__ar_coefficient__coeff_4__k_10  (20/23 folds)         |      0.75 |
| P3-O1__partial_autocorrelation__lag_4  (20/23 folds)          |      0.74 |
| P4-O2__value_count__value_0  (22/23 folds)                    |      0.72 |
| T7-FT9__partial_autocorrelation__lag_4  (19/23 folds)         |      0.69 |

## By bipolar derivation

| derivation   |   share % |
|:-------------|----------:|
| P7-T7        |     22.28 |
| P4-O2        |     14.24 |
| T8-P8        |     10.17 |
| CZ-PZ        |      6.18 |
| P8-O2        |      4.82 |
| FP2-F4       |      3.89 |
| T7-FT9       |      3.79 |
| F4-C4        |      3.6  |
| C3-P3        |      3.59 |
| FP1-F3       |      3.42 |
| FZ-CZ        |      3.37 |
| FT9-FT10     |      3.34 |
| FT10-T8      |      2.68 |
| FP1-F7       |      2.53 |
| P7-O1        |      2.46 |
| F7-T7        |      2.25 |
| FP2-F8       |      2.22 |
| P3-O1        |      2.11 |
| F8-T8        |      1.67 |
| C4-P4        |      0.81 |
| F3-C3        |      0.59 |

## By tsfresh feature family (top 15)

| family                  |   share % |
|:------------------------|----------:|
| ar_coefficient          |     25.53 |
| partial_autocorrelation |     17.37 |
| permutation_entropy     |     15.31 |
| autocorrelation         |      9.22 |
| symmetry_looking        |      6.5  |
| number_peaks            |      2.95 |
| number_cwt_peaks        |      2.84 |
| ratio_beyond_r_sigma    |      2.47 |
| augmented_dickey_fuller |      2.01 |
| lempel_ziv_complexity   |      1.9  |
| has_duplicate_max       |      1.31 |
| has_duplicate_min       |      1.3  |
| index_mass_quantile     |      1.16 |
| value_count             |      1.15 |
| agg_autocorrelation     |      1.11 |

## Current window vs. history summary

| block   |   share % |
|:--------|----------:|
| window  |     95.88 |
| trend   |      4.12 |

Figures: shap_top_features.png, shap_by_family_and_block.png, topomap_bipolar.png, fold_plots/*_shap_beeswarm.png, lime/*.png