#!/usr/bin/env python3
"""
01_channel_scope.py -- channel profiling, Kalman filter + BOCPD (with forgetting)
calibration, and MCC-based channel scope selection (README section 5).

Inputs : data/OPSSAT-AD/segments.csv
Outputs (in outputs/):
  channel_profiles_v2.csv
  detector_calibration_v2_final_locked.csv
  onset_bayesian_results.csv            (one row per segment; used by 02)
  onset_posterior_distributions.csv     (BOCPD onset posterior; used by 02)
  channel_scope.csv
  channel_mcc_bootstrap.csv
  channel_scope_final_locked.csv

Locked switches (selected by a 4-combination ablation in the original work):
  USE_MIXTURE_NOISE_ESTIMATOR = False, USE_BOCPD_FORGETTING = True.
The BOCPD detector is used ONLY to decide channel scope and to provide an
onset posterior for 02. Runtime: several minutes (O(n^2) BOCPD per segment).
"""
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OPSSAT_DATA_DIR", ROOT / "data" / "OPSSAT-AD"))
OUT_DIR = Path(os.environ.get("OPSSAT_OUT_DIR", ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0884", "CADC0886",
                "CADC0888", "CADC0890", "CADC0892", "CADC0894"]
QUANT_FRAC_ZERO_THRESHOLD = 0.10      # >= this fraction of zero diffs -> quantized
FLOAT_NOISE_ABS_DIFF_THRESHOLD = 1e-6  # smallest non-zero |diff| below this -> float-noise
USE_MIXTURE_NOISE_ESTIMATOR = False
USE_BOCPD_FORGETTING = True
HAZARD_LAMBDA = 250.0
KAPPA_MAX = 40.0
N_BOOT = 2000
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Channel profiling and noise estimation
# --------------------------------------------------------------------------
@dataclass
class ChannelProfile:
    channel: str
    channel_type: str
    frac_zero_diff: float
    min_nonzero_abs_diff: float
    quantization_step: float
    r_robust: float
    q_robust: float
    n_nominal_points: int


def mad_variance(x):
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    return float((1.4826 * np.median(np.abs(x - med))) ** 2)


def mixture_variance(d):
    """Zero-inflated mixture: Var ~= p * Var(jump | jump != 0). Only used if the switch is on."""
    if len(d) < 2:
        return np.nan
    p = float(np.mean(d != 0))
    if p == 0:
        return 0.0
    nz = d[d != 0]
    jv = mad_variance(nz) if len(nz) >= 2 else float(nz[0] ** 2)
    return p * jv


def profile_channel(channel, nominal_values_by_segment):
    diffs = [np.diff(v) for v in nominal_values_by_segment if len(v) > 1]
    if not diffs:
        return ChannelProfile(channel, "unknown", np.nan, np.nan, np.nan, 1.0, 1e-6, 0)
    d = np.concatenate(diffs)
    n_points = len(d) + len(nominal_values_by_segment)
    frac_zero = float(np.mean(d == 0))
    nz = d[d != 0]
    min_nz = float(np.min(np.abs(nz))) if len(nz) else np.nan
    if frac_zero >= QUANT_FRAC_ZERO_THRESHOLD:
        ctype, qstep = "quantized", float(np.percentile(np.abs(nz), 25))
    elif np.isfinite(min_nz) and min_nz < FLOAT_NOISE_ABS_DIFF_THRESHOLD:
        ctype, qstep = "float_noise_suspect", np.nan
    else:
        ctype, qstep = "continuous", np.nan
    d2 = np.diff(d)
    if USE_MIXTURE_NOISE_ESTIMATOR:
        r = max(mixture_variance(d) / 2.0, 1e-14)
        q = max(mixture_variance(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14
    else:
        r = max(np.var(d) / 2.0, 1e-14)
        q = max(np.var(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14
    return ChannelProfile(channel, ctype, frac_zero, min_nz, qstep, r, q, n_points)


def hierarchical_shrink_variance(est):
    """Empirical-Bayes (DerSimonian-Laird style) shrinkage of log-variances across channels."""
    chs = list(est.keys())
    log_v = np.array([np.log(max(est[c][0], 1e-12)) for c in chs])
    n_eff = np.array([max(est[c][1], 2) for c in chs])
    se = np.sqrt(2.0 / (n_eff - 1))
    w = 1.0 / se ** 2
    gm = np.sum(w * log_v) / np.sum(w)
    wss = np.sum(w * (log_v - gm) ** 2)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (wss - (len(chs) - 1)) / denom) if denom > 0 else 0.0
    out = {}
    for c, lv, s in zip(chs, log_v, se):
        wc = tau2 / (tau2 + s ** 2) if (tau2 + s ** 2) > 0 else 0.0
        out[c] = float(np.exp(wc * lv + (1 - wc) * gm))
    return out


# --------------------------------------------------------------------------
# Local-linear-trend Kalman filter and BOCPD
# --------------------------------------------------------------------------
@dataclass
class LocalLinearTrendKF:
    q: float
    r_nominal: float
    theta: float = 0.0
    quantization_floor: float = 0.0
    x0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P0_scale: float = 10.0

    def __post_init__(self):
        self.r_eff = self.r_nominal + self.theta + self.quantization_floor
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[0.0, 0.0], [0.0, self.q]])

    def run(self, y):
        n = len(y)
        x, P = self.x0.copy(), np.eye(2) * self.P0_scale
        nu_arr, S_arr = np.empty(n), np.empty(n)
        for t in range(n):
            xp = self.A @ x
            Pp = self.A @ P @ self.A.T + self.Q
            S = (self.H @ Pp @ self.H.T)[0, 0] + self.r_eff
            nu = y[t] - (self.H @ xp)[0]
            nu_arr[t], S_arr[t] = nu, S
            K = (Pp @ self.H.T) / S
            x = xp + K.flatten() * nu
            P = Pp - K @ self.H @ Pp
        return nu_arr, S_arr


class BOCPD:
    """Adams & MacKay BOCPD, Normal-Inverse-Gamma model, with optional forgetting
    (cap on kappa; alpha and beta are rescaled proportionally)."""

    def __init__(self, hazard_lambda=250.0, mu0=0.0, kappa0=1.0, alpha0=1.0,
                 beta0=1.0, kappa_max=40.0, use_forgetting=True):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.kappa_max, self.use_forgetting = kappa_max, use_forgetting

    def _cap(self, kappa, alpha, beta):
        if not self.use_forgetting:
            return kappa, alpha, beta
        over = kappa > self.kappa_max
        if not np.any(over):
            return kappa, alpha, beta
        scale = np.where(over, self.kappa_max / kappa, 1.0)
        return (np.where(over, self.kappa_max, kappa),
                np.maximum(np.where(over, alpha * scale, alpha), 1e-3),
                np.where(over, beta * scale, beta))

    def run(self, x):
        n = len(x)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0
        mu, kappa = np.array([self.mu0]), np.array([self.kappa0])
        alpha, beta = np.array([self.alpha0]), np.array([self.beta0])
        mrl = np.zeros(n, dtype=int)
        for t in range(n):
            xt = x[t]
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            pp = stats.t.pdf(xt, df=2 * alpha, loc=mu, scale=scale)
            pp = np.nan_to_num(pp, nan=1e-12, posinf=1e-12)
            rp = R[t, : t + 1]
            new_R = np.zeros(t + 2)
            new_R[0] = np.sum(rp * pp * self.hazard)
            new_R[1:] = rp * pp * (1 - self.hazard)
            tot = new_R.sum()
            if tot <= 0 or not np.isfinite(tot):
                new_R[:] = 0
                new_R[0] = 1.0
            else:
                new_R /= tot
            R[t + 1, : t + 2] = new_R
            mrl[t] = int(np.argmax(new_R))
            nm, nk = np.empty(t + 2), np.empty(t + 2)
            na, nb = np.empty(t + 2), np.empty(t + 2)
            nm[0], nk[0], na[0], nb[0] = self.mu0, self.kappa0, self.alpha0, self.beta0
            nm[1:] = (kappa * mu + xt) / (kappa + 1)
            nk[1:] = kappa + 1
            na[1:] = alpha + 0.5
            nb[1:] = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))
            nk, na, nb = self._cap(nk, na, nb)
            mu, kappa, alpha, beta = nm, nk, na, nb

        onset_idx = conf = confirm = post = None
        for t in range(1, n):
            if mrl[t] <= 2 and mrl[t - 1] > 5:
                onset_idx, confirm = t - mrl[t], t
                row = R[t + 1, : t + 2]
                k = min(len(row), 30)
                post = {int(r): float(row[r]) for r in range(k) if row[r] > 1e-6}
                conf = float(np.sum(row[:3]))
                break
        return dict(onset_idx=onset_idx, onset_confidence=conf, confirm_idx=confirm,
                    posterior_over_onset=post, map_run_length=mrl)


# --------------------------------------------------------------------------
# MCC helpers
# --------------------------------------------------------------------------
def mcc_from_counts(tp, fn, fp, tn):
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def scope_decision(row):
    n_anom, n_norm = row["n_anomaly_segments"], row["n_normal_segments"]
    recall, fa = row["recall"], row["false_alarm_rate"]
    if n_anom == 0 or pd.isna(recall):
        return np.nan, np.nan, "structural exclusion (no anomalies)"
    tp = recall * n_anom
    fp = fa * n_norm
    mcc = mcc_from_counts(tp, n_anom - tp, fp, n_norm - fp)
    j = recall - fa
    if n_anom < 5 or n_norm < 5:
        reason = f"insufficient sample (n_anom={int(n_anom)}, n_norm={int(n_norm)})"
    elif mcc <= 0:
        reason = "at or below chance (MCC<=0)"
    else:
        reason = "included"
    return mcc, j, reason


def main():
    seg = pd.read_csv(DATA_DIR / "segments.csv", parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(ALL_CHANNELS)].sort_values(["channel", "segment", "timestamp"]).reset_index(drop=True)

    # ---- (A) profiling on normal segments ----
    profiles = {}
    for ch in ALL_CHANNELS:
        nominal = [g["value"].values for _, g in
                   seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment")]
        profiles[ch] = profile_channel(ch, nominal)
        p = profiles[ch]
        print(f"[{ch}] type={p.channel_type:<19s} frac_zero={p.frac_zero_diff:.3f} "
              f"r={p.r_robust:.4g} q={p.q_robust:.4g}")
    pd.DataFrame([vars(p) for p in profiles.values()]).to_csv(OUT_DIR / "channel_profiles_v2.csv", index=False)

    r_shr = hierarchical_shrink_variance({c: (profiles[c].r_robust, max(profiles[c].n_nominal_points, 2)) for c in ALL_CHANNELS})
    q_shr = hierarchical_shrink_variance({c: (profiles[c].q_robust, max(profiles[c].n_nominal_points, 2)) for c in ALL_CHANNELS})
    qfloor = {c: (profiles[c].quantization_step ** 2 / 12.0) if np.isfinite(profiles[c].quantization_step) else 0.0
              for c in ALL_CHANNELS}

    # ---- (B) online filtering + BOCPD ----
    calib_rows, onset_rows, post_rows = [], [], []
    for ch in ALL_CHANNELS:
        kf = LocalLinearTrendKF(q=q_shr[ch], r_nominal=r_shr[ch], quantization_floor=qfloor[ch])
        bocpd = BOCPD(HAZARD_LAMBDA, kappa_max=KAPPA_MAX, use_forgetting=USE_BOCPD_FORGETTING)
        n_anom = n_anom_det = n_norm = n_norm_fa = 0
        for sid, s in seg[seg["channel"] == ch].groupby("segment"):
            values = s["value"].values
            is_anom = bool(s["anomaly"].iloc[0])
            if len(values) < 5:
                continue
            nu, S = kf.run(values)
            z = nu / np.sqrt(np.clip(S, 1e-8, None))
            res = bocpd.run(z)
            det = res["onset_idx"] is not None
            onset_rows.append(dict(channel=ch, segment=int(sid), is_anomaly=int(is_anom),
                                   n_points=len(values), onset_idx=res["onset_idx"],
                                   onset_confidence=res["onset_confidence"],
                                   confirm_idx=res["confirm_idx"], detected=int(det)))
            if res["posterior_over_onset"] is not None:
                for rl, pr in res["posterior_over_onset"].items():
                    post_rows.append(dict(channel=ch, segment=int(sid), confirm_idx=res["confirm_idx"],
                                          run_length_at_confirm=rl,
                                          candidate_onset_idx=res["confirm_idx"] - rl,
                                          posterior_prob=pr))
            if is_anom:
                n_anom += 1
                n_anom_det += int(det)
            else:
                n_norm += 1
                n_norm_fa += int(det)
        calib_rows.append(dict(channel=ch, channel_type=profiles[ch].channel_type,
                               n_anomaly_segments=n_anom, n_anomaly_detected=n_anom_det,
                               recall=n_anom_det / n_anom if n_anom else np.nan,
                               n_normal_segments=n_norm, n_normal_false_alarm=n_norm_fa,
                               false_alarm_rate=n_norm_fa / n_norm if n_norm else np.nan))
        r = calib_rows[-1]
        print(f"[{ch}] recall={r['recall']:.3f} (n_anom={n_anom})  FA={r['false_alarm_rate']:.3f} (n_norm={n_norm})")

    calib = pd.DataFrame(calib_rows)
    calib.to_csv(OUT_DIR / "detector_calibration_v2_final_locked.csv", index=False)
    onset_df = pd.DataFrame(onset_rows)
    onset_df.to_csv(OUT_DIR / "onset_bayesian_results.csv", index=False)
    pd.DataFrame(post_rows).to_csv(OUT_DIR / "onset_posterior_distributions.csv", index=False)

    # ---- (E) scope ----
    scope_rows = []
    for _, row in calib.iterrows():
        mcc, j, reason = scope_decision(row)
        scope_rows.append(dict(channel=row["channel"], channel_type=row["channel_type"],
                               recall=row["recall"], false_alarm_rate=row["false_alarm_rate"],
                               mcc=mcc, youden_j=j, scope_reason=reason,
                               include_in_stage2=(reason == "included")))
    scope = pd.DataFrame(scope_rows)
    scope.to_csv(OUT_DIR / "channel_scope.csv", index=False)
    print("\n" + scope.to_string(index=False))

    # ---- (H) bootstrap MCC CI ----
    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for ch in ALL_CHANNELS:
        sub = onset_df[onset_df["channel"] == ch]
        a = sub[sub["is_anomaly"] == 1]["detected"].values
        nrm = sub[sub["is_anomaly"] == 0]["detected"].values
        if len(a) == 0 or len(nrm) == 0:
            boot.append(dict(channel=ch, mcc_median=np.nan, mcc_ci_low=np.nan, mcc_ci_high=np.nan,
                             n_anomaly=len(a), n_normal=len(nrm), ci_crosses_zero=np.nan))
            continue
        m = np.empty(N_BOOT)
        for i in range(N_BOOT):
            a_s = rng.choice(a, len(a), replace=True)
            n_s = rng.choice(nrm, len(nrm), replace=True)
            m[i] = mcc_from_counts(a_s.sum(), len(a_s) - a_s.sum(), n_s.sum(), len(n_s) - n_s.sum())
        lo, hi = np.percentile(m, [2.5, 97.5])
        boot.append(dict(channel=ch, mcc_median=float(np.median(m)), mcc_ci_low=float(lo),
                         mcc_ci_high=float(hi), n_anomaly=len(a), n_normal=len(nrm),
                         ci_crosses_zero=bool(lo < 0 < hi)))
    pd.DataFrame(boot).to_csv(OUT_DIR / "channel_mcc_bootstrap.csv", index=False)

    final = scope[["channel", "include_in_stage2", "scope_reason"]].rename(columns={"scope_reason": "scope_reason_final"})
    final.to_csv(OUT_DIR / "channel_scope_final_locked.csv", index=False)
    included = final.loc[final["include_in_stage2"], "channel"].tolist()
    print(f"\nChannels in scope ({len(included)}): {included}")
    expected = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]
    print("Matches paper scope." if included == expected else
          f"WARNING: paper scope is {expected}")


if __name__ == "__main__":
    main()
