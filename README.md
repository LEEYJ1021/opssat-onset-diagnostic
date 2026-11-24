# OPS-SAT-AD: difference-domain variability and estimator dependence of onset-centred analyses

This repository contains the code, intermediate files and reference values for a set of analyses of the OPS-SAT-AD satellite telemetry benchmark. The analyses examine (i) segment-level separation of anomalous and nominal telemetry using first- and second-difference features, and (ii) onset-centred statistics together with their dependence on the change-point estimator.

## Status and scope

- **All analyses are exploratory.** The analysis plan (hypotheses H1-H5 and parameters, `docs/analysis_plan_freeze.json`) was recorded on 14 October 2026, after extensive exploratory work. It is provided for transparency and documents the parameters used; it is not a pre-registration.
- The earlier implementation (placebo contrast, sixteen-model attribution, Kendall's W) is preserved in `legacy_v1/` for reference. The current pipeline extends it with length-matched segment-level analysis and estimator-aware controls, and supersedes it.
- Two interpretations from the earlier implementation were refined once estimator-aware controls were applied: (i) that rate-of-change variance marks anomaly onset, and (ii) that sixteen models never rely on level. The current analysis instead reports a length-controlled segment-level result and an estimator-aware analysis of onset-centred statistics.
- Early detection and operator utility are outside the scope of the present analyses and are natural directions for follow-up work.

## Summary of results

- **Segment level.** After matching segment length, first- and second-difference features separate anomalous from nominal segments in four of six analysable channels (CADC0872, 0873, 0874, 0888; length-matched AUC 0.95-0.99). Length and sampling interval alone perform near chance. CADC0892 shows high separation under logistic regression and random forest (AUC 0.93); CADC0894, the channel with the fewest anomalous segments (21), shows a model-dependent result (logistic regression 0.543, random forest 0.816) and is reported descriptively.
- **Increment over distributed features.** The additional gain over the derivative features distributed with the benchmark is modest and channel-dependent.
- **Onset-centred contrasts.** Around estimated onsets, contrasts against random pivots are large (A up to 0.976 for the first-difference variance ratio) and are conditional on the change-point estimator.
- **Controls.** Pseudo-onset controls (the same estimator applied to normal segments), five estimator configurations, window scans and an injection analysis together indicate that the signal is carried by extended intervals around the change rather than by a sharply localised onset instant. Characterising an onset-localised signature is a target for datasets with labelled onset instants.

## Repository layout

```
.
├── src/common/rev_common.py          # shared utilities, constants, paths (reads environment variables)
├── pipeline/
│   ├── 01_onset_estimation/          # original pipeline incl. KF-BOCPD onset stage (paper2_pipeline.py)
│   ├── 02_segment_level/             # segment-level analysis (primary result): rev8, rev8b, rev8c
│   ├── 03_onset_contrasts/           # symmetric statistics, cluster CIs: rev2
│   ├── 04_estimator_controls/        # pseudo-onsets, estimator matrix, synthetic validation, agreement: rev1, rev3, rev4, rev9
│   ├── 05_window_power/              # window scan (rev10), injection analysis (rev11)
│   ├── 06_sequence_models/           # sequence models (rev6), attribution baselines (rev12); sensitivity analyses
│   ├── 07_supplementary/             # tabular models (rev5), seventeenth model (rev7)
│   ├── 08_figures/                   # number/figure helpers
│   ├── run_all.sh, run_rev6.sh       # launchers used for the runs (see "Launchers")
├── data/
│   ├── OPSSAT-AD/                    # place dataset.csv and segments.csv here (not committed)
│   └── derived/                      # intermediate files required by the scripts (see below)
├── outputs/
│   ├── tables/                       # result tables (CSV)
│   └── logs/                         # run logs
├── ledger/numbers_ledger.csv         # sample-size ledger: every n with its source script
├── docs/analysis_plan_freeze.json    # recorded analysis plan
├── env/requirements.lock.txt         # pinned environment
└── legacy_v1/                        # earlier implementation (superseded, kept for reference)
```

## Data

- **OPS-SAT-AD** (Ruszczak, Kotowski, Evans & Nalepa, *Sci. Data* 12, 710, 2025), Zenodo version v2: <https://doi.org/10.5281/zenodo.15108715> (accessed October 2026).
- Download `dataset.csv` and `segments.csv` into `data/OPSSAT-AD/`. They are not redistributed here; the dataset licence on its Zenodo record applies.
- A *segment* is one labelled interval of one channel (anomalous or normal). A *tick* is a sample position after sorting the segment by timestamp (no resampling; the sampling interval is 1 s or 5 s). Labels delimit a segment but do not give the onset instant; onsets are data-derived estimates.
- Dataset facts used as sanity checks: 434 anomalous and 1,689 normal segments over nine channels. Analysable channels (at least ten segments of each class): CADC0872 (131 anomalous / 415 normal), CADC0873 (105 / 488), CADC0874 (69 / 125), CADC0888 (60 / 192), CADC0892 (34 / 177), CADC0894 (21 / 123). CADC0884 has no anomalous segments; CADC0886 (3 / 8) and CADC0890 (11 / 3) have too few segments of one class for classification.

## Intermediate files (`data/derived/`) and their provenance

The analysis scripts read intermediate files in addition to the raw data. Their provenance is as follows.

| File | Content | Produced by | Read by |
|---|---|---|---|
| `data/derived/p2_segment_features.csv` | per-segment features (level, dispersion, derivative statistics, covariates) | `pipeline/01_onset_estimation/paper2_pipeline.py` | `rev8*`, `rev8c` |
| `data/derived/rev4_onsets.csv` | onsets (true and pseudo) for PELT(3), PELT(5), CUSUM, level-input BOCPD | `pipeline/04_estimator_controls/rev4_estimator_matrix.py` | `rev5`, `rev6`, `rev9`, `rev10`, `rev12` |
| `causal_pipeline_outputs/stage2_mc_draws_raw.csv`, `stage2_mc_segment_summary_corrected.csv` (+ `_NORMAL` variants) | KF-BOCPD Monte Carlo onset draws and reliability tiers (primary estimator) | Precomputed input from the original pipeline. `legacy_v1/scripts/02_onset_estimation.py` is a reconstruction with assumed tier thresholds and is not guaranteed to reproduce these files | `rev_common.py` (`STAGE`), `rev2`, `rev9`, `rev5`, `rev7` |

> **Reproduction scope.** Onset-centred results with the primary estimator depend on the `stage2_*` files, which are provided as fixed inputs; reproduction therefore starts downstream of them. Regenerating the primary-estimator onsets from raw data additionally requires re-running that stage and comparing it with the reference counts below.

## Environment

Python 3.12.3 with NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.14.1 (GPU), XGBoost 3.4.1, LightGBM 4.7.0, ruptures 1.1.10. The full pinned list is in `env/requirements.lock.txt`. Runs used an NVIDIA RTX PRO 6000 Blackwell (driver 580.126.09).

SHAP and Captum are not part of this environment. The SHAP/Captum analyses of the earlier implementation were run in a separate environment and are not used in the main results. Integrated gradients in `rev12` are implemented directly in PyTorch.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.lock.txt   # choose the torch build matching your CUDA
```

## Configuration

Paths are controlled by environment variables (no author-specific paths are hard-coded):

```bash
export OPSSAT_BASE_DIR=/path/to/working/dir     # must contain paper2_outputs/ and causal_pipeline_outputs/
                                                # (copy files from data/derived/ there as needed)
export OPSSAT_OUT_DIR=$OPSSAT_BASE_DIR/revision_outputs   # default
```

`rev_common.py` defines `BASE`, `STAGE = BASE/causal_pipeline_outputs`, `OUT`, the channel list `[CADC0872, 0873, 0874, 0888, 0894]` (onset-centred analysis), `BURN_IN = 5` and `MIN_W = 5`. Each script inserts `src/common` into `sys.path` automatically.

## Running

Smoke tests (minutes) before full runs:

```bash
python pipeline/04_estimator_controls/rev9_estimator_agreement.py
python pipeline/02_segment_level/rev8_segment_level.py --quick
python pipeline/05_window_power/rev10_window_scan.py --quick
python pipeline/05_window_power/rev11_injection.py --quick
```

Full runs, in dependency order:

| Step | Command | Approx. runtime* | Output |
|---|---|---|---|
| 1 | `python pipeline/04_estimator_controls/rev4_estimator_matrix.py` | minutes | `rev4_onsets.csv` |
| 2 | `python pipeline/03_onset_contrasts/rev2_symmetric_ci.py` | minutes | `rev2_symmetric_A_ci.csv` |
| 3 | `python pipeline/04_estimator_controls/rev9_estimator_agreement.py` | minutes | `rev9_estimator_agreement.csv` |
| 4 | `python pipeline/02_segment_level/rev8b_segment_level.py` | ~20 min | `rev8b_segment_level.csv` |
| 5 | `python pipeline/02_segment_level/rev8c_rough_ablation.py` | minutes | roughness ablation |
| 6 | `python pipeline/05_window_power/rev10_window_scan.py` | ~15-25 min | `rev10_window_scan.csv` |
| 7 | `python pipeline/05_window_power/rev11_injection.py` | ~10-20 min | `rev11_injection_auc.csv`, `rev11_estimator_recovery.csv` |
| 8 | `python pipeline/06_sequence_models/rev6_seq_models.py --neg random --estimators PELT3 --models CNN1D,TCN,BiLSTM,BiGRU,TinyTransformer --seed 1 --gpu 0 --n-boot 100` (also `--neg pseudo`; seeds 1, 2, 3; seed 42 is the default) | tens of minutes per seed | `rev6_auc*.csv` |
| 9 | `python pipeline/06_sequence_models/rev12_seq_attr.py ...` (same arguments) | ~20-40 min | `rev12_attr_baselines.csv` |
| 10 | `python pipeline/07_supplementary/rev5_models.py`, `rev7_mlp_attrib.py` | minutes | tabular models, seventeenth model |

\*Rough values from runs on one machine with several jobs sharing CPU/GPU; not benchmarks.

### Launchers

`pipeline/run_all.sh` and `pipeline/run_rev6.sh` are the scripts used to run jobs in parallel. They are provided as templates and contain author-specific paths (`cd`, virtual-environment `source`) that should be adapted to your setup before use. `run_all.sh` launches GPU jobs on GPU 0 only.

## Method summary

**Representations and statistics.** Level, first difference (`dx`) and second difference (`d2x`). At a pivot, the series is split into pre and post windows after a burn-in of 5 ticks, with at least 5 points per window. Three statistics are computed for each representation: standardised mean shift (pooled SD), standardised mean shift (pre-window SD) and log variance ratio `ln(var_post/var_pre)`. A statistic is undefined (NaN) if a window has fewer than 5 points or zero variance. Effect size is the probability of superiority `A = U/(n1 n2)` (Mann-Whitney U with mid-ranks); `A = 0.5` means no difference. Intervals are 95% percentile intervals from a segment-clustered bootstrap (anomalous and normal segments resampled independently; 1000 resamples for onset contrasts, 300 for segment-level analyses). P values are two-sided and descriptive; they do not account for clustering, and no multiplicity adjustment is applied.

**Primary onset estimator (KF-BOCPD).** Bayesian online change-point detection with a normal-inverse-gamma observation model (hazard 1/250, `kappa_max` 40) applied to Kalman innovations (local linear trend, channel-specific noise). Onset = median of 200 Monte Carlo draws. Reliability grades (high/medium/low) come from the spread of the draws around their median; high and medium are retained. The estimator is variance-sensitive (synthetic validation in `rev3`). Its sample is defined by detection: of 386 anomalous segments in the five onset channels, 203 showed no detected change-point pattern, 4 were too short and 1 had no admissible window; of the 178 detected, 176 were high, 1 medium and 1 low (not retained), giving 177 onsets. The onset sample therefore consists of segments with clearly detectable change-points.

**Comparison estimators.** PELT (penalties 3 and 5, `ruptures`), CUSUM, and a level-input BOCPD. "PELT" without a penalty means PELT(3).

**Random pivots.** For each channel a relative position is drawn from that channel's relative onset positions; pivot `tau = clip(round(rho*n), 10, n-5)`; five pivots per normal segment, from up to 300 normal segments per channel, seed 42. Normal segments are those evaluated by the primary estimator pipeline (`ev_n`); this explains, for example, 152 x 5 = 760 pivots in CADC0888 and 299 x 5 = 1,495 in CADC0872.

**Pseudo-onsets.** The same estimator applied to normal segments. It produced no detections in CADC0872/0873 and 4 in CADC0874, so the control is quantitatively informative in CADC0888 and CADC0894, where detections were sufficient.

**Segment-level analysis.** Logistic regression (balanced) and random forest (300 trees, depth 6, balanced), stratified five-fold cross-validation (one sample per segment), three repeats; features are median-imputed and quantile-transformed within training folds. Feature sets: covariates only (length, duration, sampling interval), level, dispersion, the four benchmark-distributed derivative features, our derivative set (adds first- and second-difference roughness), and all features. Length matching pairs anomalous and normal segments within quartile bins of length.

**Window scan and injection.** Window half-lengths 8, 10, 15, 20, 30, 50 ticks and the full segment, for PELT(3) and CUSUM onsets, five seeds, grouped cross-validation. Injection: mean shifts (2, 5, 10 sigma), variance increases (x2, x3, x5) and slopes (0.05, 0.15, 0.5 sigma/tick) added at a known position (40-60% of the segment) to real normal segments (oracle pivot); estimator recovery is scored within +/-5 ticks.

**Sequence models (sensitivity analysis).** 1D CNN, TCN, BiLSTM, BiGRU and a small Transformer on three-channel windows (level, `dx`, `d2x`) of +/-15 ticks, under window-wise z-scoring or global robust scaling. Folds are formed by segment. Within each training fold, 15% of segments are held out as an inner validation set for early stopping (patience 6, up to 50 epochs, best-epoch restoration); the outer fold is used only for prediction. Seeds 1, 2, 3 and 42 change the fold assignment, the inner split and the weight initialisation. A simplified state-space block (not the published Mamba) is included as an exploratory variant.

**Operating points.** The sequence scripts can also print recall, false-alarm rate and precision at a Youden threshold. These are selected on the same out-of-fold predictions and are best read as upper-range values; they are not part of the reported results.

## Reference values for checking a reproduction

Sample sizes (see `ledger/numbers_ledger.csv`; each analysis uses its own sample):

| Analysis | n | Positive / negative |
|---|---|---|
| Segment level (six channels) | 1,940 | 420 / 1,520 |
| Length-matched pairs per channel (0872, 0873, 0874, 0888, 0892, 0894) | - | 131, 105, 40, 60, 34, 21 |
| Onset sample (primary estimator) | 177 onsets | 386 anomalous segments in five channels |
| Random pivots (primary estimator) | 4,980 | five per evaluated normal segment |
| Tabular models (KF-BOCPD, nine statistics defined) | 4,961 | 168 / 4,793 (1,163 segment groups) |
| Seventeenth-model sensitivity (three statistics defined) | 4,974 | 168 / 4,806 |
| Sequence, random negatives (PELT(3)) | 2,693 | 297 / 2,396 (prevalence 11.0%) |
| Sequence, pseudo-onset negatives (PELT(3)) | 784 | 297 / 487 (prevalence 37.9%) |

Segment-level length-matched AUC of the derivative set (`outputs/tables/rev8_segment_level.csv`):

| Channel | Logistic regression | Random forest |
|---|---|---|
| CADC0872 | 0.962 | 0.976 |
| CADC0873 | 0.954 | 0.975 |
| CADC0874 | 0.980 | 0.993 |
| CADC0888 | 0.963 | 0.973 |
| CADC0892 | 0.934 | 0.928 |
| CADC0894 | 0.543 | 0.816 |

Onset versus random pivots, first-difference variance ratio, primary estimator (`outputs/tables/rev2_symmetric_A_ci.csv`, `config = rest_k5`): A = 0.968 (CADC0872), 0.976 (CADC0873), 0.865 (CADC0874), 0.761 (CADC0888), 0.865 (CADC0894).

Tolerances: `rev8b` reproduces `rev8` to within 1.5e-5 (same seeds). Sequence-model AUCs vary with the seed (SD roughly 0.01-0.03 per model) and may differ in the second or third decimal across hardware and library versions (GPU kernels are not deterministic).

## Result-to-script map

| Result | Script | Output file |
|---|---|---|
| Sample flow and counts | `rev_common.py`, `paper2_pipeline.py`, `ledger/numbers_ledger.csv` | `ledger/numbers_ledger.csv` |
| Segment-level separation and increment over distributed features | `rev8_segment_level.py`, `rev8b_segment_level.py` | `rev8_segment_level.csv`, `rev8b_segment_level.csv` |
| Roughness ablation | `rev8c_rough_ablation.py` | `rev8c_rough_ablation.csv` |
| Onset versus random-pivot contrasts with clustered CIs | `rev2_symmetric_ci.py` | `rev2_symmetric_A_ci.csv` |
| Estimator matrix and pseudo-onset controls | `rev4_estimator_matrix.py` | `rev4_onsets.csv`, `rev4_firing_rates.csv`, `rev4_estimator_matrix.csv` |
| Estimator agreement | `rev9_estimator_agreement.py` | `rev9_estimator_agreement.csv` |
| Synthetic validation of the estimator | `rev3_synthetic.py` | log / table |
| Window scan | `rev10_window_scan.py` | `rev10_window_scan.csv` |
| Injection analysis | `rev11_injection.py` | `rev11_injection_auc.csv`, `rev11_estimator_recovery.csv` |
| Sequence-model sensitivity analysis | `rev6_seq_models.py` | `rev6_auc*.csv`, `rev6_seed_model_long.csv` |
| Attribution baselines | `rev12_seq_attr.py` | `rev12_attr_baselines.csv` |
| Seventeenth-model sensitivity | `rev5_models.py`, `rev7_mlp_attrib.py` | `rev7_attrib_sensitivity.csv` |
| Detector firing rate by segment length | `rev1_estimator_control.py` | `rev1_firing_by_length.csv` |

## Randomness and determinism

Seed 42 is the default (`numpy`, `torch`); sequence-model seeds 1, 2, 3 and 42 were run. Pivot sampling uses `numpy.random.default_rng` with fixed seeds in the code. Random forests may use multiple threads. GPU runs were not made deterministic.

## Scope and extensions

- The primary-estimator onset files (`stage2_*`) are provided as fixed inputs; see the reproduction scope note above.
- Sequence-model and attribution analyses use PELT(3) onsets, one attribution seed and a single estimator, and serve as sensitivity analyses that complement the main results.
- The pseudo-onset control is quantitatively informative in two channels (CADC0888, CADC0894); increasing the number of normal-segment detections would extend it, and results depend on pivot position and sample size.
- Onset-level channels were selected using detector performance (CADC0892 was not carried into the onset-centred analysis because the detector performed below chance on it), so the onset sample emphasises clearly detectable cases. CADC0892 is covered by the segment-level analysis.
- Different analyses use different samples (see the ledger); sequence and tabular results are compared only within the same sample.
- Launcher scripts contain author-specific paths and serve as templates.
- External-dataset replication (hypothesis H5) is a planned extension; the ESA Anomaly Detection Benchmark (ESA-ADB) is the natural next dataset.

## Legacy code (earlier implementation)

`legacy_v1/` contains the scripts of the earlier implementation (`00_eda_verify.py` ... `04_placebo_contrast.py`) and its README. They implement the placebo contrast, the sixteen-model attribution study, Kendall's W and a binomial calculation, as well as an early-stopping protocol that monitored the held-out fold (replaced in the current pipeline by an inner validation split). They are kept for transparency and reference.

## Citation and licence

Please cite the benchmark: Ruszczak, B., Kotowski, K., Evans, D. & Nalepa, J. The OPS-SAT benchmark for detecting anomalies in satellite telemetry. *Sci. Data* 12, 710 (2025). A citation for this repository will be added.

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0.
