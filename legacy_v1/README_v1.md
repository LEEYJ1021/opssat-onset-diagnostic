# Rate-of-change variance, not level shifts, marks anomaly onset in satellite telemetry

Reproducibility package for the placebo-controlled representation analysis (level vs. rate of change vs. acceleration) and the sixteen-model attribution study on the public **OPS-SAT-AD** benchmark.

## Repository layout

```
.
├── data/OPSSAT-AD/              # dataset.csv, segments.csv (download; not committed)
├── scripts/
│   ├── 00_eda_verify.py         # schema / integrity checks
│   ├── 01_channel_scope.py      # Kalman filter + BOCPD calibration, MCC channel scope
│   ├── 02_onset_estimation.py   # Monte Carlo onset draws + reliability tier
│   ├── 03_modelclass_analysis.py# dataset build, 16 models, attribution, agreement
│   └── 04_placebo_contrast.py   # Table 2 and Fig. 2
├── outputs/                     # all results are written here
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # choose the torch build matching your CUDA
pip freeze > outputs/requirements.lock.txt   # record the versions you actually used

# 1. put dataset.csv and segments.csv into data/OPSSAT-AD/
python scripts/00_eda_verify.py
python scripts/01_channel_scope.py        # minutes (O(n^2) BOCPD per segment)
python scripts/02_onset_estimation.py
python scripts/03_modelclass_analysis.py 2>&1 | tee outputs/run_log.txt
python scripts/04_placebo_contrast.py
```

Paths default to `data/OPSSAT-AD` and `outputs`; override with `OPSSAT_DATA_DIR` and `OPSSAT_OUT_DIR`.

| Paper element | Script | Output file |
|---|---|---|
| Table 1 (samples) | 03 | `modelclass_dataset_tabular_raw.csv` |
| Table 2, Fig. 2 | 04 | `placebo_contrast_table2.csv`, `fig2_placebo_contrast.png` |
| Table 3, Fig. 3, Fig. 4 (data) | 03 | `modelclass_feature_importance.csv`, `modelclass_performance_leaderboard.csv` |
| Kendall's W, binomial | 03 | `modelclass_agreement_summary.csv` |
| Supplementary Table S2 | 03 | `modelclass_family_summary.csv` |
| Supplementary bootstrap note | 03 | `modelclass_bootstrap_stability.csv` |
| Channel scope (section 5) | 01 | `channel_scope_final_locked.csv`, `channel_mcc_bootstrap.csv` |

Figures 3 and 4 are not plotted by the scripts; their data are the two CSVs above.

> ### Read this before claiming a reproduction
> **`02_onset_estimation.py` is a reconstruction.** The original onset-estimation script was not available when this package was assembled. Script 02 re-implements the described procedure (200 Monte Carlo draws, median onset, high/medium/low reliability) on top of the BOCPD onset posterior from script 01. The **tier thresholds are assumptions**, so the retained segment count may differ from the paper's 177 (872: 41, 873: 29, 874: 50, 888: 36, 894: 21). Script 02 prints its counts next to these. If you have the original `stage2_mc_draws_raw.csv` and `stage2_mc_segment_summary_corrected.csv` (schema below), drop them into `outputs/` and skip script 02. All downstream numbers depend on these two files.
>
> **The pipeline was smoke-tested on synthetic data only** (scripts 01–04, without torch/xgboost/lightgbm/shap). It has not been run on the real benchmark by the package author, so compare your outputs with the reference values below.

## Step 0 - Data (script 00)

OPS-SAT-AD: Ruszczak, Kotowski, Evans & Nalepa, *Sci. Data* 12, 710 (2025). Get the files from the Zenodo record given in that paper's Data Availability section and place them in `data/OPSSAT-AD/`.

| File | Shape | Content |
|---|---|---|
| `dataset.csv` | 2,123 x 23 | per-segment summary features, `anomaly`, `train`, `channel` |
| `segments.csv` | 303,493 x 8 | `channel, timestamp, value, label, sampling, anomaly, segment, train` |

The analysis uses only `segments.csv` (`channel`, `segment`, `anomaly`, `timestamp`, `value`). `dataset.csv` is used for sanity checks. A *segment* is one labelled interval of one channel; a *tick* is a sample position after sorting the segment by `timestamp` (no resampling; sampling is 1 s or 5 s). All onset/pivot/window indices are positions in that sorted array. Labels delimit a segment but **do not give the onset instant**.

Script 00 asserts: shapes, 434 anomalous / 1,689 normal segments, one channel and one label per segment, no NaNs, per-channel (normal, anomalous) counts of 872 (415, 131), 873 (488, 105), 874 (125, 69), 888 (192, 60), 894 (123, 21). It exits non-zero on failure.

## Step 1 - Channel scope (script 01)

Decides which channels have a detectable anomaly signature. Paper scope: **CADC0872, 0873, 0874, 0888, 0894**; excluded: 0884 (no anomalies), 0886 and 0890 (too few segments), 0892 (MCC < 0).

1. **Profiling** on normal segments: `frac_zero_diff >= 0.10` -> quantized (step = 25th percentile of non-zero |dx|); else min non-zero |dx| < 1e-6 -> float-noise; else continuous. Noise `r = var(dx)/2`, `q = var(d2x)/6` (floor 1e-14), shrunk across channels with empirical Bayes on log-variances. Quantization floor `step^2/12` added to observation noise.
2. **Kalman filter**, local linear trend: state [level, slope], `A=[[1,1],[0,1]]`, `H=[1,0]`, `Q=diag(0,q)`, `P0=10 I`; z = innovation / sqrt(S).
3. **BOCPD** on z: Normal-Inverse-Gamma, hazard 1/250, prior (0,1,1,1), forgetting with `kappa_max=40`. Onset declared when MAP run length drops from >5 to <=2; candidate onset = t - run length.
4. Locked switches: `USE_MIXTURE_NOISE_ESTIMATOR=False`, `USE_BOCPD_FORGETTING=True`.
5. **Scope rule**: exclude if `n_anom=0`, `n_anom<5`, `n_norm<5`, or MCC <= 0; 2,000-resample bootstrap MCC CI.

Expected (paper-level) values with locked switches: 872 recall 0.321 / FA 0.000 / MCC 0.514; 873 0.276 / 0.000 / 0.489; 874 0.725 / 0.032 / 0.740; 888 0.617 / 0.391 / 0.194; 894 1.000 / 0.797 / 0.189; 892 0.971 / 0.994 / -0.090 (excluded). The script prints whether the resulting scope matches the paper.

BOCPD is used here only for scope and as the onset posterior for script 02. The train-only ablation re-selection of the two switches from the original work is not included.

## Step 2 - Onset estimation (script 02)

Per anomalous in-scope segment on which BOCPD fired: draw 200 onsets from the normalised onset posterior, onset = rounded median of draws, tier from posterior mass near the reset (`onset_confidence`) and draw SD. Only `high`/`medium` are kept. **Interface contract** (what script 03 reads):

`outputs/stage2_mc_draws_raw.csv`: `channel, segment, draw, onset_idx_sampled` (200 rows per segment; 0-based tick index).
`outputs/stage2_mc_segment_summary_corrected.csv`: at least `channel, segment, reliability_tier` (`high`/`medium`/`low`).

Tunables: `--high-conf 0.8 --medium-conf 0.5 --high-sd 2.0`.

## Step 3 - Dataset construction (script 03, part 1)

| Parameter | Value |
|---|---|
| `BURN_IN` | 5 |
| `MIN_WINDOW_POINTS` | 5 |
| placebo pivots per normal segment | 5 |
| max normal segments per channel | 300 (`random_state=42`) |
| sequence half-window | 15 ticks (30-tick window) |
| seed | 42 |

**Anomalous samples (label 1).** For each retained segment (values sorted by time, length n, onset tau): skip if `tau < 10` or `tau > n-5`; record `tau/n` in the channel's ratio pool.

**Placebo samples (label 0).** For each in-scope channel: take its normal segments (sample 300 if more); per segment draw 5 ratios `rho` from the channel's ratio pool (`rng.choice`); `tau = clip(round(rho*n), 10, max(n-5, 10))`; skip if outside `[10, n-5]`. This matches the relative position of pivots to onsets of the same channel.

**Statistics at pivot tau** (`dx = diff(x)`, `d2x = diff(dx)`):

| Feature | Pre window | Post window | Statistic |
|---|---|---|---|
| `level` | `x[5:tau]` | `x[tau:]` | `abs(mean_post - mean_pre) / sqrt((var_pre + var_post)/2)` (ddof=1) |
| `diff` | `dx[4:tau-1]` | `dx[tau-1:]` | `ln(var_post / var_pre)` (ddof=1) |
| `diff2` | `d2x[3:tau-2]` | `d2x[tau-2:]` | `ln(var_post / var_pre)` |

A statistic is NaN if a window has < 5 points (< 2 for level) or zero variance.

**Sequence windows.** For pivot tau and h=15: `lo=tau-h`, `hi=tau+h`; dropped if `lo-2<0` or `hi>n`. Channels: `x[lo:hi]`, `dx[lo-1:hi-1]`, `d2x[lo-2:hi-2]`, each z-scored within its own window.

**Checkpoints printed by the script (paper values).** Retained anomalous segments 177; normal segments 1,040; placebo pivots 4,980; anomalous with all three statistics defined 168; placebo 4,828; tabular set 4,996 rows (168/4,828); sequence set 2,733 windows. If these differ, stop and fix the onset inputs before interpreting model results.

`modelclass_dataset_tabular_raw.csv` holds all pivots including NaN statistics (used by script 04, because Table 2 uses per-statistic sample sizes); `modelclass_dataset_tabular.csv` holds rows with all three defined.

## Step 4 - Placebo contrast (script 04)

For each channel x representation (15 tests): one-sided Mann-Whitney U (onset > placebo), `A = U/(n1 n2)`, Bonferroni `p_adj = min(1, 15 p)`, NaNs dropped per statistic. Paper: all ten difference-based contrasts `p_adj <= 2.0e-4` and `A` in 0.75-0.98; all five level contrasts `p_adj = 1.000` and `A` in 0.23-0.56. P values are descriptive (five pivots share a normal segment); effect sizes are primary. Script 04 prints these ranges next to the paper's.

## Step 5 - Sixteen models (script 03, part 2)

Task: onset (1) vs placebo pivot (0). Tabular input = the three statistics; sequence input = the 3 x 30 window.

**Validation.** Tabular: `GroupKFold(5)` by `channel_segment`, `StandardScaler` fitted on the training fold. Sequence: segments assigned to 5 folds via `rng.permutation(unique_groups)` and `i % 5`. Class imbalance: balanced class weights (tabular) / `pos_weight = n_neg/n_pos` (sequence). Recall and false-alarm rate at the Youden optimum of the out-of-fold predictions.

**Tabular models (10).**

| Model | Family | Configuration |
|---|---|---|
| Logistic L2 / L1 | Linear / sparse | balanced, max_iter 2000 (L1: liblinear) |
| Gaussian NB | Probabilistic | defaults |
| kNN | Instance-based | k=15 |
| SVM (RBF) | Kernel | balanced, probability=True |
| Random forest | Bagging | 300 trees, depth 6, balanced |
| Extra trees | Bagging, extra-random | 300 trees, depth 6, balanced |
| Gradient boosting | Boosting | 200 est., depth 3 |
| XGBoost | Boosting | 300 est., depth 4, lr 0.05 |
| LightGBM | Boosting | 300 est., depth 4, lr 0.05 |

**Sequence models (6).** 1D CNN (2 x Conv1d k=5, 32 ch); TCN (3 residual blocks, dilations 1/2/4); BiLSTM and BiGRU (1 bidirectional layer, 32 units); small Transformer (d=32, 2 layers, 4 heads, FFN 64, sinusoidal PE); simplified selective state-space block (d=32, state 16; **not** the published Mamba). Adam (lr 1e-3, wd 1e-4), batch 32, <= 60 epochs, early stopping (patience 6) that **monitors the held-out fold**, so AUCs are upper-range.

**Attribution.** Tabular: mean |SHAP| over up to 400 held-out examples (Tree / Linear / Kernel explainer with 20 k-means background points, 100 samples); fallback to permutation importance (20 repeats, AUC) if SHAP fails (watch the log for `[SHAP ... -> permutation importance]`). Sequence: Integrated Gradients (Captum, zero baseline, 32 Gauss-Legendre steps), |attribution| summed over time. Shares sum to 1 per model and fold, then averaged over folds.

**Agreement.** Rank the three representations within each model; Kendall's W with m=16, n=3, `chi2 = m(n-1)W`; exact binomial `(2/3)^m`; family means over 13 families.

**Bootstrap criterion.** RF and L2 logistic, 200 resamples (pivots resampled by class), permutation importance; records whether the two difference representations jointly exceed level. This passes with probability 0.75 under uninformative attribution, so it is weak evidence alone.

**Excluded model.** `--include-mlp` re-enables the shallow network (16, 8) that was excluded post hoc (out-of-fold AUC 0.403). Counting it, level ranks first in 1 of 17 models and last in 14 of 17.

**Flags.** `--skip-deep`, `--skip-bootstrap`, `--include-mlp`; `DETERMINISTIC=1` enables deterministic torch kernels (will shift sequence-model numbers slightly from the published ones).

## Reference values (paper)

| Model | AUC | Level | First | Second |
|---|---|---|---|---|
| Small Transformer | 0.994 | 0.178 | 0.438 | 0.384 |
| TCN | 0.993 | 0.200 | 0.380 | 0.419 |
| BiGRU | 0.989 | 0.285 | 0.388 | 0.327 |
| 1D CNN | 0.983 | 0.303 | 0.301 | 0.396 |
| BiLSTM | 0.980 | 0.211 | 0.497 | 0.292 |
| State-space (simplified) | 0.969 | 0.428 | 0.441 | 0.130 |
| SVM (RBF) | 0.948 | 0.061 | 0.466 | 0.472 |
| LightGBM | 0.940 | 0.232 | 0.459 | 0.309 |
| XGBoost | 0.935 | 0.148 | 0.419 | 0.433 |
| Random forest | 0.931 | 0.084 | 0.456 | 0.460 |
| Extra trees | 0.927 | 0.145 | 0.389 | 0.466 |
| Gradient boosting | 0.924 | 0.068 | 0.371 | 0.561 |
| Logistic (L2) | 0.920 | 0.060 | 0.407 | 0.534 |
| Logistic (L1) | 0.920 | 0.057 | 0.412 | 0.531 |
| kNN | 0.918 | 0.129 | 0.391 | 0.480 |
| Gaussian NB | 0.900 | 0.051 | 0.446 | 0.503 |

Kendall's W = 0.609, chi2 = 19.50 (d.f. 2), p = 5.8e-5; level ranked first in 0/16 and last in 14/16; binomial p = 0.0015; bootstrap 200/200 for both models.

**Suggested acceptance criteria.** Sample counts must match. Tabular shares and AUCs should be close (seeded; small differences from SHAP/library versions). Sequence models: AUC within about 0.01 and no model ranking level first.

## Randomness and determinism

Global seed 42 (`numpy`, `random`, `torch`). One module-level `numpy.random.Generator` is consumed sequentially (pivot sampling, sequence fold assignment, bootstrap), so **run script 03 top to bottom, unmodified**. Explained examples are drawn with the global `np.random` state. The original run did not enable deterministic GPU kernels, so sequence-model AUCs and shares may differ in the 2nd-3rd decimal across hardware or library versions. Library versions of the original run were not recorded; commit your `pip freeze`.

Version pitfalls handled in code: `xgboost >= 2` removed `use_label_encoder`; recent `shap` can return a 3-D array for binary `predict_proba` (normalised to the positive class); cuDNN RNN backward fails in `eval()` (attribution is wrapped in `torch.backends.cudnn.flags(enabled=False)`); without Captum a manual IG is used, which differs slightly from Gauss-Legendre quadrature.

## Citation and license

Cite the paper (add final reference) and: Ruszczak, B., Kotowski, K., Evans, D. & Nalepa, J. The OPS-SAT benchmark for detecting anomalies in satellite telemetry. *Sci. Data* 12, 710 (2025). Add your code license; OPS-SAT-AD is distributed under the license on its Zenodo record.
