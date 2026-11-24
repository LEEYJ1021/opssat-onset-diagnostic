#!/usr/bin/env python3
"""
02_onset_estimation.py -- Monte Carlo onset draws and reliability tier (README section 6).

RECONSTRUCTION NOTICE
---------------------
The original onset-estimation script was not part of the code supplied for this
package. This file is a documented re-implementation of the procedure described
in the paper (median of 200 Monte Carlo draws from a Bayesian change-point
model, noise estimated per channel, reliability tiers high/medium/low), built on
the BOCPD onset posterior written by 01_channel_scope.py:

  * an anomalous segment gets an onset only if the BOCPD detector fired on it;
  * 200 onset indices are drawn from the normalised BOCPD onset posterior
    (candidate_onset_idx, posterior_prob);
  * the reliability tier is derived from the posterior mass near the MAP run
    length reset (onset_confidence) and from the spread of the draws.

The tier thresholds below are ASSUMPTIONS, not the original values. The paper
reports n = 177 retained segments (872: 41, 873: 29, 874: 50, 888: 36, 894: 21).
This script prints the counts it obtains; if they differ, adjust
--high-conf / --medium-conf or substitute the original files (same schema,
see below) -- everything downstream only needs those two files.

Inputs : outputs/onset_bayesian_results.csv, outputs/onset_posterior_distributions.csv
Outputs: outputs/stage2_mc_draws_raw.csv                      (channel, segment, draw, onset_idx_sampled)
         outputs/stage2_mc_segment_summary_corrected.csv      (channel, segment, reliability_tier, ...)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OPSSAT_OUT_DIR", ROOT / "outputs"))
FINAL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]
PAPER_COUNTS = {"CADC0872": 41, "CADC0873": 29, "CADC0874": 50, "CADC0888": 36, "CADC0894": 21}
N_DRAWS = 200
RANDOM_STATE = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--high-conf", type=float, default=0.8,
                    help="onset_confidence >= this (and small spread) -> 'high'")
    ap.add_argument("--medium-conf", type=float, default=0.5,
                    help="onset_confidence >= this -> 'medium', otherwise 'low'")
    ap.add_argument("--high-sd", type=float, default=2.0,
                    help="max SD (ticks) of the draws for 'high'")
    args = ap.parse_args()

    res = pd.read_csv(OUT_DIR / "onset_bayesian_results.csv")
    post = pd.read_csv(OUT_DIR / "onset_posterior_distributions.csv")

    anom = res[(res["is_anomaly"] == 1) & (res["detected"] == 1) & (res["channel"].isin(FINAL_CHANNELS))]
    rng = np.random.default_rng(RANDOM_STATE)

    draws_rows, summ_rows = [], []
    for _, r in anom.sort_values(["channel", "segment"]).iterrows():
        p = post[(post["channel"] == r["channel"]) & (post["segment"] == r["segment"])]
        if p.empty:
            continue
        cand = p["candidate_onset_idx"].values.astype(int)
        prob = p["posterior_prob"].values.astype(float)
        prob = prob / prob.sum()
        n = int(r["n_points"])
        d = np.clip(rng.choice(cand, size=N_DRAWS, replace=True, p=prob), 0, n - 1)
        sd = float(np.std(d))
        conf = float(r["onset_confidence"])
        if conf >= args.high_conf and sd <= args.high_sd:
            tier = "high"
        elif conf >= args.medium_conf:
            tier = "medium"
        else:
            tier = "low"
        for k, v in enumerate(d):
            draws_rows.append((r["channel"], int(r["segment"]), k, int(v)))
        summ_rows.append(dict(channel=r["channel"], segment=int(r["segment"]), n_points=n,
                              onset_median=float(np.median(d)), onset_sd=sd,
                              onset_confidence=conf, reliability_tier=tier))

    draws = pd.DataFrame(draws_rows, columns=["channel", "segment", "draw", "onset_idx_sampled"])
    summ = pd.DataFrame(summ_rows)
    draws.to_csv(OUT_DIR / "stage2_mc_draws_raw.csv", index=False)
    summ.to_csv(OUT_DIR / "stage2_mc_segment_summary_corrected.csv", index=False)

    kept = summ[summ["reliability_tier"].isin(["high", "medium"])]
    print("Segments retained (high+medium) per channel vs. paper:")
    for ch in FINAL_CHANNELS:
        print(f"  {ch}: {int((kept['channel'] == ch).sum()):3d}   (paper: {PAPER_COUNTS[ch]})")
    print(f"  total: {len(kept)}   (paper: 177)")
    print("\nTier counts:\n" + pd.crosstab(summ["channel"], summ["reliability_tier"]).to_string())
    if len(kept) != 177:
        print("\nNOTE: total differs from 177. Tier thresholds are assumptions (see docstring).")


if __name__ == "__main__":
    main()
