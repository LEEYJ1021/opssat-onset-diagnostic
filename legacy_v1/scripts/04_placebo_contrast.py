#!/usr/bin/env python3
"""
04_placebo_contrast.py -- placebo contrast, Table 2 and Fig. 2 (README section 8).

Input : outputs/modelclass_dataset_tabular_raw.csv   (written by 03)
Output: outputs/placebo_contrast_table2.csv, outputs/fig2_placebo_contrast.png

For each channel x representation (5 x 3 = 15 tests): one-sided Mann-Whitney U
(alternative: onset > placebo), effect size A = U/(n1*n2) (probability of
superiority), Bonferroni p_adj = min(1, 15 p). NaNs are dropped per statistic.
P values are descriptive (five pivots share a normal segment); A is primary.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OPSSAT_OUT_DIR", ROOT / "outputs"))
N_TESTS = 15
REPS = [("level", "Level"), ("diff", "First difference"), ("diff2", "Second difference")]

raw = pd.read_csv(OUT_DIR / "modelclass_dataset_tabular_raw.csv")
rows = []
for ch, g in raw.groupby("channel"):
    for feat, name in REPS:
        a = g.loc[g.label == 1, feat].dropna()
        b = g.loc[g.label == 0, feat].dropna()
        u = mannwhitneyu(a, b, alternative="greater")   # U statistic of the first sample (onsets)
        rows.append(dict(channel=ch, representation=name, n_onset=len(a), n_placebo=len(b),
                         median_onset=a.median(), median_placebo=b.median(),
                         A=u.statistic / (len(a) * len(b)), p=u.pvalue,
                         p_adj=min(1.0, N_TESTS * u.pvalue)))
tbl = pd.DataFrame(rows)
tbl.to_csv(OUT_DIR / "placebo_contrast_table2.csv", index=False)

pd.set_option("display.width", 200)
print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
diff = tbl[tbl.representation != "Level"]
lvl = tbl[tbl.representation == "Level"]
print(f"\ndifference-based: max p_adj = {diff.p_adj.max():.2e} (paper <= 2.0e-4); A range {diff.A.min():.2f}-{diff.A.max():.2f} (paper 0.75-0.98)")
print(f"level: p_adj values = {sorted(set(lvl.p_adj.round(3)))} (paper 1.000); A range {lvl.A.min():.2f}-{lvl.A.max():.2f} (paper 0.23-0.56)")

# ---- Fig. 2 ----
chs = sorted(tbl.channel.unique())
x = range(len(chs))
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
marks = {"Level": ("o", "white"), "First difference": ("s", "black"), "Second difference": ("^", "gray")}
for rep, (mk, fc) in marks.items():
    s = tbl[tbl.representation == rep].set_index("channel").loc[chs]
    ax[0, 0].scatter(x, s.A, marker=mk, facecolor=fc, edgecolor="black", label=rep, s=50)
ax[0, 0].axhline(0.5, ls="--", c="black")
ax[0, 0].set(ylabel="Effect size U/(n1 n2)", ylim=(0, 1.05), title="a")
ax[0, 0].legend(fontsize=8)
for axis, rep, lab, t in [(ax[0, 1], "Level", "Median standardised level change |d|", "b"),
                          (ax[1, 0], "First difference", "Median log variance ratio, first difference", "c"),
                          (ax[1, 1], "Second difference", "Median log variance ratio, second difference", "d")]:
    s = tbl[tbl.representation == rep].set_index("channel").loc[chs]
    axis.vlines(list(x), s.median_placebo, s.median_onset, color="gray")
    axis.scatter(x, s.median_placebo, facecolor="white", edgecolor="black", label="Placebo pivots", s=50, zorder=3)
    axis.scatter(x, s.median_onset, color="black", label="Onsets", s=50, zorder=3)
    axis.set(ylabel=lab, title=t)
    if t == "b":
        axis.legend(fontsize=8)
for a in ax.flat:
    a.set_xticks(list(x))
    a.set_xticklabels(chs, rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_placebo_contrast.png", dpi=200)
print(f"saved {OUT_DIR / 'fig2_placebo_contrast.png'}")
