#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper2_figures.py
======================
Paper 2 (derivative-variance signature, OPS-SAT-AD) 그림/표 생성기.
paper2_pipeline.py 출력(CSV)과 stage2~stage3 산출물을 읽어 F1~F9, S1, 표 CSV를 만든다.
각 그림은 독립 try/except 이므로 일부 파일이 없어도 나머지는 생성된다.

사용 예 (프로젝트 루트 = ~/Research/Satellite):
  python make_paper2_figures.py --base-dir "OPSSAT-AD_LEO data" --tiff
"""
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# 전역 스타일 (Elsevier: 단단 90 mm, 이단 190 mm / 폰트 7-9 pt / 300 dpi)
# ----------------------------------------------------------------------------
SINGLE, DOUBLE = 3.54, 7.48  # inch
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.unicode_minus": False,
})
BLUE, ORANGE, GREEN, PINK, YELLOW, SKY, BLACK = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")  # Okabe-Ito

CH_ORDER = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0892", "CADC0894"]
ONSET_CH = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]
CH_LABEL = {"CADC0872": "MAG-0 (872)", "CADC0873": "MAG-1 (873)", "CADC0874": "MAG-2 (874)",
            "CADC0888": "PD-3 (888)", "CADC0892": "PD-5 (892)", "CADC0894": "PD-6 (894)",
            "POOLED": "Pooled"}
CH_COLOR = {c: (BLUE if c in CH_ORDER[:3] else ORANGE) for c in CH_ORDER}
CH_COLOR["POOLED"] = "#444444"

OUT = None
TIFF = False


def read(path, quiet=False):
    path = Path(path)
    if not path.exists():
        if not quiet:
            print(f"  [skip] 파일 없음: {path}")
        return None
    return pd.read_csv(path)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    if TIFF:
        fig.savefig(OUT / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  [저장] {OUT / name}.(pdf|png{'|tiff' if TIFF else ''})")


def plabel(ax, s, dx=-0.12, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def ci_row(ax, y, pt, lo, hi, color, marker="o", filled=True, ms=4.5, lw=1.2):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt")
    ax.plot(pt, y, marker=marker, ms=ms, mfc=color if filled else "white", mec=color, ls="none", zorder=3)


# ----------------------------------------------------------------------------
# F1 파이프라인/표본 흐름
# ----------------------------------------------------------------------------
def fig_pipeline(P):
    t1 = read(P / "t1_sample_flow.csv", quiet=True)
    n_seg, n_an, n_nm, n_on_an, n_on = 2123, 434, 1689, 386, 177
    if t1 is not None:
        try:
            n_seg = int(t1["n_segments"].sum()); n_an = int(t1["n_anomaly"].sum()); n_nm = int(t1["n_normal"].sum())
            col = "n_anomaly_with_onset(reliable)"
            m = t1[col].notna()
            n_on_an = int(t1.loc[m, "n_anomaly"].sum()); n_on = int(t1.loc[m, col].sum())
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", ls="-", fs=7.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=fc, ec=ec, lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333", ls=ls, shrinkA=0, shrinkB=0))

    box(0.01, 0.38, 0.19, 0.24, f"OPS-SAT-AD\n{n_seg:,} segments\n9 channels\n{n_an} anomalous / {n_nm:,} normal",
        "#F2F2F2", bold=True)
    box(0.27, 0.66, 0.22, 0.26, "Onset-free set\n6 channels (3 MAG + 3 PD)\n872/873/874/888/892/894\nall labelled segments used",
        "#D6EAF8")
    box(0.27, 0.10, 0.22, 0.26,
        f"Onset-based set\n5 channels (no 892)\n{n_on_an} anomalies -> {n_on} with\nBOCPD onset (46%)", "#FDEBD0")
    arrow(0.20, 0.55, 0.27, 0.78); arrow(0.20, 0.45, 0.27, 0.25)
    box(0.56, 0.78, 0.20, 0.14, "E1 segment-level AUC\n(group-removal, cluster bootstrap)", "#AED6F1", bold=True)
    box(0.56, 0.60, 0.20, 0.14, "E2 scan statistic\n(surrogate-calibrated null)", "#AED6F1", bold=True)
    box(0.56, 0.38, 0.20, 0.14, "E3 four onset definitions\n(A KF-BOCPD, B PELT, C CUSUM, D BOCPD)", "#F8C471")
    box(0.56, 0.20, 0.20, 0.14, "E4 within-segment pivots\n+ event study; E8 strata", "#F8C471")
    box(0.56, 0.02, 0.20, 0.14, "E5 model classes\n(representation-controlled)", "#F8C471")
    for y in (0.85, 0.67):
        arrow(0.49, 0.79, 0.56, y)
    for y in (0.45, 0.27, 0.09):
        arrow(0.49, 0.23, 0.56, y)
    box(0.82, 0.60, 0.17, 0.30, "CONFIRMATORY\n(frozen hypotheses\nH1, H3-H5)", "#EAF2F8", ls="--", bold=True)
    box(0.82, 0.18, 0.17, 0.30, "ONSET-CONDITIONAL\n(detector-dependent;\nreported with caveats)", "#FEF5E7", ls="--", bold=True)
    arrow(0.76, 0.75, 0.82, 0.75); arrow(0.76, 0.33, 0.82, 0.33)
    box(0.30, 0.455, 0.16, 0.09, "Exploratory (appendix):\nprecedence, meta-analysis", "#EEEEEE", ls=":", fs=6.5)
    box(0.82, 0.02, 0.17, 0.12, "External validation (E6)\nESA-ADB  [pending]", "#FFFFFF", ls=":", fs=7)
    ax.legend(handles=[Patch(fc="#AED6F1", ec="k", label="onset-free"),
                       Patch(fc="#F8C471", ec="k", label="onset-based")],
              loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=2)
    save(fig, "F1_pipeline")


# ----------------------------------------------------------------------------
# F2 대표 세그먼트
# ----------------------------------------------------------------------------
def fig_cases(P, S, base):
    seg_path = base / "segments.csv"
    drw = read(S / "stage2_mc_draws_raw.csv")
    e4 = read(P / "e4_within_segment_null.csv")
    if not seg_path.exists() or drw is None or e4 is None:
        print("  [skip] F2: segments.csv / stage2_mc_draws_raw.csv / e4_within_segment_null.csv 필요")
        return
    seg = pd.read_csv(seg_path, usecols=["channel", "segment", "timestamp", "value", "anomaly"], parse_dates=["timestamp"])
    onset = drw.groupby(["channel", "segment"])["onset_idx_sampled"].median().round().astype(int)
    d = e4[e4["feature"] == "diff_logvar"].copy()
    fig, axes = plt.subplots(3, len(ONSET_CH), figsize=(DOUBLE, 5.0))
    for j, ch in enumerate(ONSET_CH):
        sub = d[d["channel"] == ch]
        chosen = None
        if len(sub):
            med = sub["observed"].median()
            sub = sub.assign(dist=(sub["observed"] - med).abs()).sort_values("dist")
            for sid in sub["segment"]:
                n = int(((seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 1)).sum())
                if n >= 60 and (ch, int(sid)) in onset.index:
                    chosen = int(sid); break
        if chosen is None:
            for i in range(3):
                axes[i, j].axis("off")
            continue
        v = seg[(seg["channel"] == ch) & (seg["segment"] == chosen) & (seg["anomaly"] == 1)].sort_values("timestamp")["value"].to_numpy()
        on = int(onset[(ch, chosen)])
        series = [v, np.diff(v), np.diff(v, 2)]
        names = ["level", "1st diff", "2nd diff"]
        for i, (y, nm) in enumerate(zip(series, names)):
            ax = axes[i, j]
            z = (y - np.mean(y)) / (np.std(y) + 1e-12)
            ax.plot(np.arange(len(z)), z, color=CH_COLOR[ch], lw=0.7)
            ax.axvline(on - i, color="k", ls="--", lw=0.8)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{CH_LABEL[ch]}\nseg {chosen}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(nm + " (z)")
            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("sample index")
    plabel(axes[0, 0], "a", dx=-0.35)
    fig.suptitle("Representative anomalous segments (dashed = canonical onset; z-scored for display)", fontsize=8, y=1.0)
    save(fig, "F2_case_segments")


# ----------------------------------------------------------------------------
# F3 / S1  E1 forest
# ----------------------------------------------------------------------------
def fig_e1(P, tag="full", name="F3_E1_forest"):
    d = read(P / "e1_group_auc.csv")
    if d is None:
        return
    x = d[(d["kind"] == "contrast") & (d["item"] == "deriv_minus_level") & (d["tag"] == tag)]
    rows = CH_ORDER + ["POOLED"]
    ypos = {c: i for i, c in enumerate(rows[::-1])}
    panels = [("oof_cv", "(a) Out-of-fold CV"), ("official_test", "(b) Official test split")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.5), sharey=True)
    for ax, (ev, title) in zip(axes, panels):
        for mi, (m, mk, filled) in enumerate([("hgb", "o", True), ("logreg", "s", False)]):
            s = x[(x["evaluation"] == ev) & (x["model"] == m)]
            for _, r in s.iterrows():
                if r["block"] not in ypos:
                    continue
                ci_row(ax, ypos[r["block"]] + (mi - 0.5) * 0.28, r["point"], r["ci_lo"], r["ci_hi"],
                       CH_COLOR.get(r["block"], BLACK), marker=mk, filled=filled)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel(r"$\Delta$AUC  (derivative-variance group $-$ level-shift group)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        ref = x[(x["evaluation"] == ev) & (x["model"] == "hgb")].set_index("block")["n_pos"].to_dict()
        ax.set_yticks([ypos[c] for c in rows])
        ax.set_yticklabels([f"{CH_LABEL[c]}  ($n_+$={int(ref.get(c, 0))})" for c in rows])
    axes[0].axhspan(ypos["CADC0874"] - 0.5, ypos["CADC0872"] + 0.5, color=BLUE, alpha=0.06, lw=0)
    axes[1].axhspan(ypos["CADC0874"] - 0.5, ypos["CADC0872"] + 0.5, color=BLUE, alpha=0.06, lw=0)
    axes[0].legend(handles=[Line2D([], [], marker="o", color="k", mfc="k", ls="none", label="HistGB"),
                            Line2D([], [], marker="s", color="k", mfc="white", ls="none", label="Logistic")],
                   loc="lower right", frameon=False)
    fig.suptitle(f"E1 onset-free segment classification ({tag}; 95% cluster-bootstrap CI; blue = magnetometer, orange = photodiode)",
                 fontsize=8, y=1.02)
    save(fig, name)


# ----------------------------------------------------------------------------
# F4 E2 scan forest
# ----------------------------------------------------------------------------
def fig_e2(P):
    d = read(P / "e2_scan_auc.csv")
    if d is None:
        return
    d = d[d["kind"] == "contrast"]
    items = [("LV_minus_LS", "var-scan vs level-scan"), ("LV_minus_LSdt", "vs detrended level-scan"),
             ("LV2_minus_LS", "diff2 var-scan vs level-scan"), ("LV2_minus_LSdt", "diff2 vs detrended"),
             ("LV_raw_minus_LS_raw", "raw (uncalibrated) var vs level"),
             ("LV2_raw_minus_LSdt_raw", "raw diff2 vs detrended")]
    rows = CH_ORDER + ["POOLED"]
    ypos = {c: i for i, c in enumerate(rows[::-1])}
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 5.0), sharey=True)
    for k, (ax, (it, ttl)) in enumerate(zip(axes.ravel(), items)):
        s = d[d["item"] == it].set_index("block")
        for c in rows:
            if c in s.index:
                r = s.loc[c]
                ci_row(ax, ypos[c], r["point"], r["ci_lo"], r["ci_hi"], CH_COLOR[c])
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.set_title(f"({'abcdef'[k]}) {ttl}", loc="left", fontsize=7.5)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        ax.set_yticks([ypos[c] for c in rows]); ax.set_yticklabels([CH_LABEL[c] for c in rows])
        if k >= 3:
            ax.set_xlabel(r"$\Delta$AUC (variance scan $-$ level scan)")
    fig.suptitle("E2 detector-free scan statistics: advantage of variance over level (95% bootstrap CI)", fontsize=8, y=1.0)
    plt.tight_layout()
    save(fig, "F4_E2_scan_forest")


# ----------------------------------------------------------------------------
# F5 E4 이벤트 스터디 + 내부 위약
# ----------------------------------------------------------------------------
def fig_e4_event(P):
    d = read(P / "e4_event_study_profiles.csv")
    if d is None:
        return
    chans = [c for c in ONSET_CH if c in d["channel"].unique()]
    curves = list(d["curve"].unique())
    pal = [BLUE, ORANGE, GREEN, PINK, YELLOW, SKY]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 4.4), sharex=True)
    for ax, ch in zip(axes.ravel(), chans):
        for k, cv in enumerate(curves):
            s = d[(d["channel"] == ch) & (d["curve"] == cv)].sort_values("offset")
            if len(s):
                ax.plot(s["offset"], s["median"], color=pal[k % 6], lw=1.1, label=str(cv))
                ax.fill_between(s["offset"], s["lo"], s["hi"], color=pal[k % 6], alpha=0.18, lw=0)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.set_title(CH_LABEL[ch], loc="left", fontsize=8)
        ax.set_xlabel("offset from pivot (samples)"); ax.set_ylabel("rolling log variance ratio")
    for ax in axes.ravel()[len(chans):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center", frameon=False)
    plabel(axes[0, 0], "a")
    fig.suptitle("E4 event study aligned to onset/pivot (median, 95% band)", fontsize=8, y=1.0)
    plt.tight_layout()
    save(fig, "F5a_E4_event_study")


def fig_e4_summary(P):
    d = read(P / "e4_summary.csv")
    if d is None:
        return
    feats = [("level_d_abs", "level |d|"), ("diff_logvar", "diff log-var ratio"), ("diff2_logvar", "diff2 log-var ratio")]
    rows = [c for c in ONSET_CH if c in d["channel"].unique()]
    ypos = {c: i for i, c in enumerate(rows[::-1])}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.0), sharey=True)
    for ax, (f, ttl) in zip(axes, feats):
        s = d[d["feature"] == f].set_index("channel")
        for c in rows:
            if c not in s.index:
                continue
            r = s.loc[c]; y = ypos[c]
            ax.plot(r["median_observed"], y, "o", color=CH_COLOR[c], ms=5.5, label="observed onset")
            ax.plot(r["median_internal_null"], y, "s", mfc="white", mec=CH_COLOR[c], ms=5.5, label="within-segment random pivots")
            ax.plot(r["median_normal_placebo"], y, "x", color="#555555", ms=5.5, label="normal-segment placebo")
            ax.text(1.02, y, f"{r['frac_p_lt_05']:.0%}\n(n={int(r['n'])})", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5)
        ax.set_title(ttl, fontsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        ax.set_yticks([ypos[c] for c in rows]); ax.set_yticklabels([CH_LABEL[c] for c in rows])
    axes[0].legend(handles=[Line2D([], [], marker="o", color="k", ls="none", label="observed onset"),
                            Line2D([], [], marker="s", mfc="white", color="k", ls="none", label="within-segment random pivots"),
                            Line2D([], [], marker="x", color="#555555", ls="none", label="normal-segment placebo")],
                   loc="lower left", frameon=False, fontsize=6.5)
    fig.suptitle("E4 onset-locking: medians; right-hand numbers = share of segments with empirical p<0.05 (null ~5%)", fontsize=8, y=1.02)
    save(fig, "F5b_E4_within_segment")


# ----------------------------------------------------------------------------
# F6 E3 onset 정의 민감도
# ----------------------------------------------------------------------------
def fig_e3(P, base, pelt_dirs):
    main = read(P / "e3_h3_summary.csv")
    if main is None:
        return
    rows = []
    name_map = {"A_KF_BOCPD": "A  KF+BOCPD\n(variance-aware)", "C_level_CUSUM": "C  level CUSUM",
                "D_level_BOCPD": "D  level BOCPD"}
    for k in ["A_KF_BOCPD", "C_level_CUSUM", "D_level_BOCPD"]:
        r = main[main["definition"] == k]
        if len(r):
            rows.append((name_map[k], r.iloc[0]))
    r = main[main["definition"] == "B_PELT"]
    if len(r):
        rows.append(("B  PELT\n(default pen)", r.iloc[0]))
    for lab, dd in pelt_dirs:
        t = read(Path(dd) / "e3_h3_summary.csv", quiet=True)
        if t is not None:
            r = t[t["definition"] == "B_PELT"]
            if len(r):
                rows.append((f"B  PELT\n({lab})", r.iloc[0]))
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.4))
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w / 2, [r["frac_deriv_significant"] for _, r in rows], w, color=BLUE, label="derivative-variance significant")
    ax.bar(x + w / 2, [r["frac_level_significant"] for _, r in rows], w, color=ORANGE, label="level shift significant")
    ax.plot(x, [r["frac_deriv_positive"] for _, r in rows], "kD", ms=4, label="derivative effect positive (sign)")
    for i, (_, r) in enumerate(rows):
        ax.text(i, 1.04, f"n={int(r['n_deriv_tests'])}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows], fontsize=7)
    ax.set_ylim(0, 1.12); ax.set_ylabel("fraction of channel x feature tests")
    ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=6.8, bbox_to_anchor=(1.0, 0.92))
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_title("E3 onset-definition sensitivity: placebo contrast under four onset detectors (Mann-Whitney, Bonferroni)", fontsize=8, loc="left")
    save(fig, "F6_E3_onset_definitions")


# ----------------------------------------------------------------------------
# F7 전체/train/test 3중 검증 히트맵
# ----------------------------------------------------------------------------
def fig_triple(S):
    d = read(S / "stage3_triple_verification_summary.csv")
    if d is None:
        return
    feats = [("level_d_abs", "level |d|"), ("diff_logvar", "diff log-var"), ("diff2_logvar", "diff2 log-var")]
    chans = [c for c in ONSET_CH if c in d["channel"].unique()]
    rows = [(c, f, fl) for c in chans for f, fl in feats]
    cols = ["p_value_bonferroni_full", "p_value_bonferroni_train", "p_value_bonferroni_test"]
    M = np.full((len(rows), 3), np.nan)
    for i, (c, f, _) in enumerate(rows):
        r = d[(d["channel"] == c) & (d["feature"] == f)]
        if len(r):
            for j, col in enumerate(cols):
                M[i, j] = r.iloc[0][col]
    L = np.clip(-np.log10(np.clip(M, 1e-300, 1)), 0, 15)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#DDDDDD")
    fig, ax = plt.subplots(figsize=(SINGLE * 1.6, 0.26 * len(rows) + 1.0))
    im = ax.imshow(np.ma.masked_invalid(L), aspect="auto", cmap=cmap, vmin=0, vmax=15)
    for i in range(len(rows)):
        for j in range(3):
            p = M[i, j]
            if np.isnan(p):
                t, col = "n.d.", "k"
            else:
                t = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
                t += "*" if p < 0.05 else ""
                col = "white" if L[i, j] > 8 else "k"
            ax.text(j, i, t, ha="center", va="center", fontsize=6, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels(["all data", "train only", "test only"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{CH_LABEL[c].split(' ')[0]} {fl}" for c, _, fl in rows], fontsize=6.5)
    for k in range(1, len(chans)):
        ax.axhline(3 * k - 0.5, color="k", lw=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(r"$-\log_{10}\,p_{\mathrm{Bonf}}$ (capped at 15)", fontsize=7)
    ax.set_title("Placebo contrast, three independent slices (* p<0.05; n.d. = n<5, not determinable)", fontsize=7.5, pad=18)
    save(fig, "F7_triple_verification")


# ----------------------------------------------------------------------------
# F8 E5 모델군
# ----------------------------------------------------------------------------
def fig_e5(P):
    c = read(P / "e5_contrasts_cluster_boot.csv"); t = read(P / "e5_model_level_table.csv")
    if c is None or t is None:
        return
    reps = [("tabular", "Tabular", BLUE), ("seq_global_robust", "Sequence, global robust scaling", GREEN),
            ("seq_window_z", "Sequence, window z-score", ORANGE)]
    order = []
    for rep, _, _ in reps:
        ms = list(t[t["rep"] == rep]["model"])
        order += [(rep, m) for m in ms] + [None]
    order = order[:-1] if order and order[-1] is None else order
    n = len(order)
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 0.2 * n + 1.2), sharey=True,
                               gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.05})
    col_of = {r: col for r, _, col in reps}
    cs = c[c["contrast"] == "deriv_minus_level"].set_index(["rep", "model"])
    for i, o in enumerate(order):
        if o is None:
            continue
        if o in cs.index:
            r = cs.loc[o]
            ci_row(a, i, r["delta_auc"], r["ci_lo"], r["ci_hi"], col_of[o[0]])
    a.axvline(0, color="k", ls="--", lw=0.7); a.invert_yaxis()
    a.set_xlabel(r"$\Delta$AUC (derivative $-$ level), 95% cluster-bootstrap CI")
    a.set_yticks(range(n)); a.set_yticklabels(["" if o is None else o[1] for o in order])
    a.grid(axis="x", alpha=0.25, lw=0.4); plabel(a, "a", dx=-0.4)
    cols = ["level", "diff", "diff2", "deriv", "all"]
    ti = t.set_index(["rep", "model"])
    M = np.full((n, len(cols)), np.nan)
    for i, o in enumerate(order):
        if o is not None and o in ti.index:
            M[i] = [ti.loc[o, k] for k in cols]
    im = b.imshow(np.ma.masked_invalid(M), aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    for i in range(n):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                b.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < 0.8 else "k")
    b.set_xticks(range(len(cols))); b.set_xticklabels(cols); b.xaxis.tick_top()
    b.tick_params(left=False); b.spines["left"].set_visible(False); plabel(b, "b", dx=-0.02)
    fig.colorbar(im, ax=b, fraction=0.035, pad=0.02).set_label("AUC (grouped CV)", fontsize=7)
    a.legend(handles=[Patch(fc=col, label=lab) for _, lab, col in reps], loc="lower right", frameon=False, fontsize=6.5)
    save(fig, "F8_E5_model_classes")


# ----------------------------------------------------------------------------
# F9 채널별 증거 매트릭스 (논문의 요약 그림)
# ----------------------------------------------------------------------------
def fig_matrix(P, S):
    e1 = read(P / "e1_group_auc.csv"); e2 = read(P / "e2_scan_auc.csv"); e4 = read(P / "e4_summary.csv")
    tri = read(S / "stage3_triple_verification_summary.csv", quiet=True)
    if e1 is None or e2 is None:
        return
    C = {"sup": "#8FD3B6", "part": "#F6D68A", "no": "#F1A27A", "rev": "#D45C5C", "ind": "#C8C8C8", "na": "#FFFFFF"}
    x = e1[(e1["kind"] == "contrast") & (e1["item"] == "deriv_minus_level") & (e1["block"].isin(CH_ORDER))]
    cols = ["E1 segment AUC\n(# of 8 variants, CI>0)", "E2 scan\nvar vs level", "E2 scan\nraw variant",
            "E4 onset-locked\n(diff, % p<.05)", "E4 onset-locked\n(diff2, % p<.05)", "3-slice check\n(diff/diff2 agree)"]
    cells = {}
    for ch in CH_ORDER:
        row = []
        s = x[x["block"] == ch]
        k = int((s["ci_lo"] > 0).sum()); n = len(s)
        npos = s[s["evaluation"] == "oof_cv"]["n_pos"].min() if len(s) else 0
        cat = "ind" if npos < 30 else ("sup" if k >= 6 else ("part" if k >= 2 else "no"))
        row.append((f"{k}/{n}", cat))
        for it in ["LV_minus_LS", "LV_raw_minus_LS_raw"]:
            r = e2[(e2["block"] == ch) & (e2["item"] == it)]
            if len(r):
                r = r.iloc[0]
                cat = "sup" if r["ci_lo"] > 0 else ("rev" if r["ci_hi"] < 0 else "ind")
                row.append((f"{r['point']:+.2f}", cat))
            else:
                row.append(("-", "na"))
        for f in ["diff_logvar", "diff2_logvar"]:
            if e4 is not None and ch in ONSET_CH:
                r = e4[(e4["channel"] == ch) & (e4["feature"] == f)]
                if len(r):
                    r = r.iloc[0]; fr = r["frac_p_lt_05"]
                    cat = "ind" if r["n"] < 20 else ("sup" if fr >= 0.30 else ("part" if fr >= 0.10 else "no"))
                    row.append((f"{fr:.0%}", cat)); continue
            row.append(("n/a", "na"))
        if tri is not None and ch in ONSET_CH:
            r = tri[(tri["channel"] == ch) & (tri["feature"].isin(["diff_logvar", "diff2_logvar"]))]
            if len(r):
                ag = r["all_three_agree"]
                if ag.isna().any():
                    row.append(("n.d.", "ind"))
                else:
                    ok = int(ag.astype(bool).sum())
                    row.append((f"{ok}/{len(r)}", "sup" if ok == len(r) else "part"))
            else:
                row.append(("n/a", "na"))
        else:
            row.append(("n/a", "na"))
        cells[ch] = row
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.2))
    for i, ch in enumerate(CH_ORDER):
        for j, (txt, cat) in enumerate(cells[ch]):
            ax.add_patch(Rectangle((j, len(CH_ORDER) - 1 - i), 1, 1, fc=C[cat], ec="white", lw=1.5))
            ax.text(j + 0.5, len(CH_ORDER) - 1 - i + 0.5, txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(CH_ORDER)); ax.set_aspect("auto")
    ax.set_xticks([j + 0.5 for j in range(len(cols))]); ax.set_xticklabels(cols, fontsize=6.8); ax.xaxis.tick_top()
    ax.set_yticks([len(CH_ORDER) - 1 - i + 0.5 for i in range(len(CH_ORDER))])
    ax.set_yticklabels([CH_LABEL[c] for c in CH_ORDER]); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(len(CH_ORDER) - 3, color="k", lw=1.0)
    ax.legend(handles=[Patch(fc=C["sup"], label="supported"), Patch(fc=C["part"], label="partial"),
                       Patch(fc=C["no"], label="not supported"), Patch(fc=C["rev"], label="reversed (level > variance)"),
                       Patch(fc=C["ind"], label="underpowered / n.s. / n.d.")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=6.8)
    ax.set_title("Evidence matrix: where the derivative-variance signature is supported (thresholds in caption)", fontsize=8, pad=26, loc="left")
    save(fig, "F9_evidence_matrix")


# ----------------------------------------------------------------------------
# 표 CSV
# ----------------------------------------------------------------------------
def tables(P, out):
    (out / "tables").mkdir(exist_ok=True)
    e1 = read(P / "e1_group_auc.csv", quiet=True)
    if e1 is not None:
        x = e1[(e1["kind"] == "contrast") & (e1["item"] == "deriv_minus_level")].copy()
        x["ci_excludes_0"] = x["ci_lo"] > 0
        x.to_csv(out / "tables" / "T_E1_contrasts_long.csv", index=False)
        g = (x[x["block"] != "POOLED"].groupby("block")
             .agg(n_variants=("point", "size"), n_ci_gt0=("ci_excludes_0", "sum"),
                  min_delta=("point", "min"), max_delta=("point", "max")).reset_index())
        g.to_csv(out / "tables" / "T_E1_channel_summary.csv", index=False)
        print(g.to_string(index=False))
    e2 = read(P / "e2_scan_auc.csv", quiet=True)
    if e2 is not None:
        c = e2[e2["kind"] == "contrast"].copy()
        c["sign_class"] = np.where(c["ci_lo"] > 0, "variance>level", np.where(c["ci_hi"] < 0, "level>variance", "n.s."))
        c.to_csv(out / "tables" / "T_E2_contrasts.csv", index=False)
        print(c.groupby(["block", "sign_class"]).size().unstack(fill_value=0).to_string())
    for f in ["e4_summary.csv", "e5_h4_summary.csv", "e5_model_level_table.csv", "e8_stratified_placebo.csv", "t1_sample_flow.csv"]:
        d = read(P / f, quiet=True)
        if d is not None:
            d.to_csv(out / "tables" / f"T_{f}", index=False)
    print(f"  [저장] {out / 'tables'}")


def main():
    global OUT, TIFF
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="OPSSAT-AD_LEO data")
    ap.add_argument("--paper2-dir", default=None)
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tiff", action="store_true", help="600 dpi LZW TIFF도 저장")
    ap.add_argument("--only", nargs="*", default=None, help="예: F3 F4 F9 tables")
    a = ap.parse_args()
    base = Path(a.base_dir)
    P = Path(a.paper2_dir) if a.paper2_dir else base / "paper2_outputs"
    S = Path(a.stage_dir) if a.stage_dir else base / "causal_pipeline_outputs"
    OUT = Path(a.out_dir) if a.out_dir else base / "paper2_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    TIFF = a.tiff
    pelt = [("pen=5", base / "paper2_outputs_pelt5"), ("pen=10", base / "paper2_outputs_pelt10")]
    jobs = {
        "F1": lambda: fig_pipeline(P),
        "F2": lambda: fig_cases(P, S, base),
        "F3": lambda: fig_e1(P, "full", "F3_E1_forest"),
        "S1": lambda: fig_e1(P, "len_matched", "S1_E1_forest_length_matched"),
        "F4": lambda: fig_e2(P),
        "F5": lambda: (fig_e4_event(P), fig_e4_summary(P)),
        "F6": lambda: fig_e3(P, base, pelt),
        "F7": lambda: fig_triple(S),
        "F8": lambda: fig_e5(P),
        "F9": lambda: fig_matrix(P, S),
        "tables": lambda: tables(P, OUT),
    }
    for k, fn in jobs.items():
        if a.only and k not in a.only:
            continue
        print(f"\n=== {k} ===")
        try:
            fn()
        except Exception:
            print(f"  [오류] {k} 생성 실패 (나머지 계속):")
            traceback.print_exc()
    print(f"\n완료. 출력 폴더: {OUT.resolve()}")


if __name__ == "__main__":
    main()
