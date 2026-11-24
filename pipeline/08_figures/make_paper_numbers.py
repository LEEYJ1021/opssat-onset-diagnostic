#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_paper_numbers.py
논문에 들어가는 모든 수치·표·그림을 파이프라인 출력 CSV에서 '동적으로' 생성한다.
(하드코딩된 수치 없음. 파일이 없으면 해당 항목만 건너뜀)

실행 예:
  python make_paper_numbers.py --base_dir "$OPSSAT_BASE_DIR"
  python make_paper_numbers.py --base_dir "D:\\Research\\OPSSAT-AD_LEO data" --alpha 0.05 --dpi 300

산출물 (base_dir/paper_outputs/):
  numbers.json            본문·초록용 모든 수치
  paper_numbers.md        그대로 붙여 쓸 수 있는 결과 문장 + 수치 요약
  T1_data_channels.{csv,md,tex}
  T3_placebo_contrast.{csv,md,tex}
  T4_triple_verification.{csv,md,tex}
  T5_heterogeneity_wilson.{csv,md,tex}
  fig2c_onset_ratio.{pdf,png}
  fig5_placebo_contrast.{pdf,png}
  fig6_triple_verification.{pdf,png}
  T2a/T2b, T_precedence_*, T_persistence, T5b, T_loo, T6, T7, fig4, figS_persistence, fig7, fig8, fig9
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ 인자
ap = argparse.ArgumentParser()
ap.add_argument("--base_dir", required=True, help="segments.csv가 있는 폴더")
ap.add_argument("--out_sub", default="causal_pipeline_outputs", help="파이프라인 출력 하위 폴더")
ap.add_argument("--alpha", type=float, default=0.05, help="Bonferroni 보정 p 기준")
ap.add_argument("--min_n", type=int, default=5, help="판정 가능 최소 표본(검정 코드와 동일)")
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--tiers", default="high,medium", help="분석에 쓴 신뢰도 등급")
args = ap.parse_args()

BASE = Path(args.base_dir)
SRC = BASE / args.out_sub
OUT = BASE / "paper_outputs"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = args.alpha
TIERS = [t.strip() for t in args.tiers.split(",")]

FEATS = [("level_d_abs", "Level shift |d|", "level"),
         ("diff_logvar", "log var ratio (Δ)", "diff"),
         ("diff2_logvar", "log var ratio (Δ²)", "diff²")]

mpl_cfg = {"font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
           "axes.spines.top": False, "axes.spines.right": False,
           "pdf.fonttype": 42, "savefig.bbox": "tight"}
plt.rcParams.update(mpl_cfg)
C_ANOM, C_PLACEBO = "#1b6ca8", "#9a9a9a"

N = {}          # numbers.json
TEXT = []       # paper_numbers.md


def load(path, **kw):
    p = Path(path)
    if not p.exists():
        print(f"[건너뜀] 파일 없음: {p}")
        return None
    return pd.read_csv(p, **kw)


def fmt_p(p):
    if p is None or pd.isna(p):
        return "n/a"
    if p < 1e-16:
        return "<1e-16"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def stars(p):
    if pd.isna(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < ALPHA else "ns"


def fnum(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def save_table(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    cols = list(df.columns)
    md = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    (OUT / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("<", r"$<$")
    tex = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline",
           " & ".join(esc(c) for c in cols) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        tex.append(" & ".join(esc(v) for v in r.values) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[저장] {name}.(csv|md|tex)")


def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=args.dpi)
    plt.close(fig)
    print(f"[저장] {name}.(pdf|png)")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# ------------------------------------------------------------------ 로드
mw_full = load(SRC / "stage3_qexp_mannwhitney.csv")
mw_train = load(SRC / "stage3_qexp_mannwhitney_train_only.csv")
mw_test = load(SRC / "stage3_qexp_mannwhitney_test_only.csv")
tri = load(SRC / "stage3_triple_verification_summary.csv")
chsum = load(SRC / "stage3_qexp_channel_summary.csv")
sgp = load(SRC / "stage3_qexp_segment_pvalues.csv")
plc = load(SRC / "stage3_qexp_placebo_pool.csv")
draws = load(SRC / "stage2_mc_draws_raw.csv")
segc = load(SRC / "stage2_mc_segment_summary_corrected.csv")
seg = load(BASE / "segments.csv", usecols=["channel", "segment", "anomaly", "train", "value"])

# 채널 목록 = 분석 결과 파일에서 동적으로 (순서 유지)
ref = mw_full if mw_full is not None else (tri if tri is not None else chsum)
CHS = list(dict.fromkeys(ref["channel"])) if ref is not None else []
N["channels"] = CHS
short = lambda c: str(c).replace("CADC", "")

# ================================================================== T1 + onset 위치비율
if seg is not None and CHS:
    s = seg[seg["channel"].isin(CHS)]
    meta = (s.groupby(["channel", "segment"])
              .agg(anomaly=("anomaly", "max"), train=("train", "max"), n_all=("value", "size")).reset_index())
    n_anom_rows = (s[s["anomaly"] == 1].groupby(["channel", "segment"]).size().rename("n_points").reset_index())
    meta = meta.merge(n_anom_rows, on=["channel", "segment"], how="left")
    rel = None
    if segc is not None:
        rel = segc[segc["reliability_tier"].isin(TIERS)][["channel", "segment"]].drop_duplicates()
    rows = []
    for ch in CHS:
        m = meta[meta["channel"] == ch]
        a = m[m["anomaly"] == 1]
        r = {"channel": ch, "n_segments": len(m), "n_anomalous": len(a), "n_normal": len(m) - len(a),
             "anomaly_rate_%": fnum(100 * len(a) / max(len(m), 1), 1)}
        if rel is not None:
            ar = a.merge(rel, on=["channel", "segment"])
            r["n_reliable_all"] = len(ar)
            r["n_reliable_train"] = int((ar["train"] == 1).sum())
            r["n_reliable_test"] = int((ar["train"] == 0).sum())
        rows.append(r)
    T1 = pd.DataFrame(rows)
    save_table(T1, "T1_data_channels")
    N["n_segments_total_analysed"] = int(len(meta))
    N["n_anomalous_total_analysed"] = int((meta["anomaly"] == 1).sum())
    if rel is not None:
        N["n_reliable_total"] = int(T1["n_reliable_all"].sum())
        N["n_reliable_train"] = int(T1["n_reliable_train"].sum())
        N["n_reliable_test"] = int(T1["n_reliable_test"].sum())
    rows_tr = int((s["train"] == 1).sum()); rows_te = int((s["train"] == 0).sum())
    N["rows_total"] = int(len(s)); N["rows_train"] = rows_tr; N["rows_test"] = rows_te
    N["train_row_share_%"] = round(100 * rows_tr / max(len(s), 1), 1)

    # onset 위치비율
    if draws is not None and rel is not None:
        can = (draws.merge(rel, on=["channel", "segment"])
                    .groupby(["channel", "segment"])["onset_idx_sampled"].median().round().astype(int)
                    .rename("onset").reset_index())
        can = can.merge(meta[["channel", "segment", "n_points"]], on=["channel", "segment"])
        can["ratio"] = can["onset"] / can["n_points"]
        N["onset_ratio_mean"] = round(float(can["ratio"].mean()), 3)
        N["onset_ratio_median"] = round(float(can["ratio"].median()), 3)
        N["n_draws_per_segment"] = int(draws.groupby(["channel", "segment"]).size().median())
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        bins = np.linspace(0, 1, 21)
        for ch in CHS:
            ax.hist(can.loc[can["channel"] == ch, "ratio"], bins=bins, histtype="step", lw=1, label=short(ch))
        ax.axvline(can["ratio"].mean(), color="k", ls="--", lw=0.8)
        ax.text(can["ratio"].mean(), ax.get_ylim()[1] * 0.95, f" mean={can['ratio'].mean():.3f}", fontsize=6.5, va="top")
        ax.set_xlabel("Onset position / segment length"); ax.set_ylabel("Segments")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        save_fig(fig, "fig2c_onset_ratio")

# ================================================================== T3 + 피처별 요약
if mw_full is not None:
    T3 = mw_full.copy()
    T3["sig"] = T3["p_value_bonferroni"].apply(stars)
    T3o = pd.DataFrame({
        "channel": T3["channel"].map(short), "feature": T3["feature"].str.replace("_d_abs", "").str.replace("_logvar", ""),
        "n_anomaly": T3["n_anomaly"], "n_placebo": T3["n_placebo"],
        "median_anomaly": T3["median_anomaly"].map(fnum), "median_placebo": T3["median_placebo"].map(fnum),
        "p_bonf": T3["p_value_bonferroni"].map(fmt_p), "sig": T3["sig"]})
    save_table(T3o, "T3_placebo_contrast")

    N["n_tests_bonferroni"] = int(mw_full["p_value"].notna().sum())
    for f, lab, key in FEATS:
        d = mw_full[mw_full["feature"] == f]
        sig = d[d["p_value_bonferroni"] < ALPHA]
        N[f"{key}_n_channels_tested"] = int(d["p_value_bonferroni"].notna().sum())
        N[f"{key}_n_channels_significant"] = int(len(sig))
        N[f"{key}_p_bonf_max_among_sig"] = float(sig["p_value_bonferroni"].max()) if len(sig) else None
        N[f"{key}_p_bonf_min"] = float(d["p_value_bonferroni"].min())
        N[f"{key}_n_channels_anom_median_le_placebo"] = int((d["median_anomaly"] <= d["median_placebo"]).sum())

# ================================================================== T4 3중 검증
if tri is not None:
    t = tri.copy()
    for sl in ["full", "train", "test"]:
        p = pd.to_numeric(t[f"p_value_bonferroni_{sl}"], errors="coerce")
        t[f"sig_{sl}"] = np.where(p.isna(), np.nan, (p < ALPHA).astype(float))
    S = t[["sig_full", "sig_train", "sig_test"]]
    valid = S.notna().all(axis=1)
    agree = valid & (S.nunique(axis=1) == 1)
    t["status"] = np.where(~valid, "undecidable", np.where(agree, "agree", "disagree"))
    # n_anomaly(슬라이스별)
    def nmap(df):
        return {} if df is None else {(r.channel, r.feature): r.n_anomaly for r in df.itertuples()}
    n_tr, n_te = nmap(mw_train), nmap(mw_test)
    T4 = pd.DataFrame({
        "channel": t["channel"].map(short),
        "feature": t["feature"].str.replace("_d_abs", "").str.replace("_logvar", ""),
        "p_full": t["p_value_bonferroni_full"].map(fmt_p),
        "p_train": t["p_value_bonferroni_train"].map(fmt_p),
        "p_test": t["p_value_bonferroni_test"].map(fmt_p),
        "n_anom_train": [n_tr.get((c, f), np.nan) for c, f in zip(t["channel"], t["feature"])],
        "n_anom_test": [n_te.get((c, f), np.nan) for c, f in zip(t["channel"], t["feature"])],
        "status": t["status"]})
    save_table(T4, "T4_triple_verification")
    N["triple_n_combinations"] = int(len(t))
    N["triple_n_testable"] = int(valid.sum())
    N["triple_n_agree"] = int(agree.sum())
    N["triple_agree_%"] = round(100 * agree.sum() / max(valid.sum(), 1), 1)
    N["triple_disagree"] = [f"{short(c)}:{f}" for c, f in zip(t.loc[t["status"] == 'disagree', "channel"], t.loc[t["status"] == 'disagree', "feature"])]
    N["triple_undecidable"] = [f"{short(c)}:{f}" for c, f in zip(t.loc[t["status"] == 'undecidable', "channel"], t.loc[t["status"] == 'undecidable', "feature"])]
    # 불일치 조합의 p값 추이
    N["triple_disagree_detail"] = {f"{short(r.channel)}:{r.feature}": [r.p_value_bonferroni_full, r.p_value_bonferroni_train, r.p_value_bonferroni_test]
                                  for r in t[t["status"] == "disagree"].itertuples()}

    # ---- Fig 6
    cols = ["p_value_bonferroni_full", "p_value_bonferroni_train", "p_value_bonferroni_test"]
    M = -np.log10(t[cols].apply(pd.to_numeric, errors="coerce").clip(lower=1e-300)).to_numpy()
    fig, ax = plt.subplots(figsize=(3.4, 0.22 * len(t) + 0.9))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="auto", vmin=0)
    for (r, c), v in np.ndenumerate(M):
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, hatch="////", ec="#888", lw=0))
            ax.text(c, r, "n/a", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Full", "Train", "Test"])
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{short(c)} · {f.split('_')[0]}" for c, f in zip(t["channel"], t["feature"])])
    for r, st in enumerate(t["status"]):
        if st == "disagree":
            ax.text(2.62, r, "≠", va="center", fontsize=8, color="#b00020")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.08)
    cb.set_label(r"$-\log_{10}$ p (Bonferroni)")
    cb.ax.axhline(-np.log10(ALPHA), color="w", lw=1)
    save_fig(fig, "fig6_triple_verification")

# ================================================================== T5 이질성(세그먼트 단위 유의 비율, Wilson CI)
if chsum is not None:
    rows = []
    for _, r in chsum.iterrows():
        for f, lab, key in FEATS:
            nv, fr = r.get(f"{f}_n_valid", np.nan), r.get(f"{f}_frac_significant", np.nan)
            if pd.isna(nv) or pd.isna(fr):
                continue
            k = int(round(fr * nv)); lo, hi = wilson(k, int(nv))
            rows.append({"channel": short(r["channel"]), "feature": key, "n_valid": int(nv),
                         "frac_sig": fnum(fr, 3), "wilson_lo": fnum(lo, 3), "wilson_hi": fnum(hi, 3)})
    save_table(pd.DataFrame(rows), "T5_heterogeneity_wilson")
    N["frac_significant"] = {f"{short(r['channel'])}:{key}": (None if pd.isna(r.get(f'{f}_frac_significant')) else round(float(r[f'{f}_frac_significant']), 3))
                             for _, r in chsum.iterrows() for f, _, key in FEATS}

# ================================================================== Fig 5
if mw_full is not None and sgp is not None and plc is not None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for ax, (f, lab, key), tag in zip(axes, FEATS, "abc"):
        for i, ch in enumerate(CHS):
            a = sgp.loc[sgp["channel"] == ch, f"{f}_observed"].dropna() if f"{f}_observed" in sgp else pd.Series(dtype=float)
            b = plc.loc[plc["channel"] == ch, f].dropna()
            for v, c, dx in ((b, C_PLACEBO, -0.2), (a, C_ANOM, 0.2)):
                if len(v) >= args.min_n:
                    vp = ax.violinplot(v, positions=[i + dx], widths=0.34, showextrema=False)
                    for body in vp["bodies"]:
                        body.set_facecolor(c); body.set_alpha(0.35); body.set_edgecolor("none")
                    ax.hlines(np.median(v), i + dx - 0.12, i + dx + 0.12, color=c, lw=1.6)
            r = mw_full[(mw_full["channel"] == ch) & (mw_full["feature"] == f)]
            p = r["p_value_bonferroni"].iloc[0] if len(r) else np.nan
            ax.text(i, 1.02, stars(p), transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
            ax.text(i, -0.22, f"{len(a)}/{len(b)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color="#555")
        ax.set_xticks(range(len(CHS))); ax.set_xticklabels([short(c) for c in CHS])
        ax.set_ylabel(lab)
        ax.text(-0.14, 1.12, f"({tag})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C_PLACEBO, alpha=.5), plt.Rectangle((0, 0), 1, 1, fc=C_ANOM, alpha=.5)],
                   labels=["Placebo (normal)", "Anomalous"], frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    save_fig(fig, "fig5_placebo_contrast")

# ================================================================== [PART 2] 추가 표·그림 (실제 열 이름 기준)
ctype = {}
_prof = load(SRC / "channel_profiles_v2.csv")
if _prof is not None:
    ctype = dict(zip(_prof["channel"], _prof["channel_type"]))
TYPE_COLOR = {"float_noise_suspect": "#1a9988", "quantized": "#e08a1e"}
tcol = lambda ch: TYPE_COLOR.get(ctype.get(ch), "#555555")
infilt = lambda df: df[df["channel"].isin(CHS)] if (df is not None and "channel" in df.columns and CHS) else df
N["channel_types"] = {short(c): ctype.get(c) for c in CHS}
FNAMES = [("level", "level"), ("diff", "diff"), ("diff2", "diff²")]

# ---------------------------------------------------------------- T2 탐지 ablation
ab = load(SRC / "ablation_4combo_train_selection.csv")
if ab is not None:
    best = ab["train_macro_mcc"].idxmax()
    T2a = pd.DataFrame({"mixture": ab["use_mixture"], "forgetting": ab["use_forgetting"],
                        "train_macro_MCC": ab["train_macro_mcc"].map(lambda x: fnum(x, 3)),
                        "n_channels_scored": ab["n_scoring_channels"],
                        "selected": ["✓" if i == best else "" for i in ab.index]})
    save_table(T2a, "T2a_ablation_train")
    N["ablation_selected"] = {"mixture": bool(ab.loc[best, "use_mixture"]), "forgetting": bool(ab.loc[best, "use_forgetting"]),
                              "train_macro_mcc": float(ab.loc[best, "train_macro_mcc"]),
                              "n_scoring_channels": int(ab.loc[best, "n_scoring_channels"])}
te = load(SRC / "ablation_best_combo_test_eval.csv")
if te is not None:
    te2 = infilt(te)
    T2b = pd.DataFrame({"channel": te2["channel"].map(short), "n_anom": te2["n_anom"], "n_norm": te2["n_norm"],
                        "recall": te2["recall"].map(lambda x: fnum(x, 3)), "FA": te2["fa"].map(lambda x: fnum(x, 3)),
                        "MCC": te2["mcc"].map(lambda x: fnum(x, 3))})
    save_table(T2b, "T2b_detection_test")
    N["test_macro_mcc_final_channels"] = round(float(te2["mcc"].mean()), 3)
    N["test_recall_min"] = round(float(te2["recall"].min()), 3)
    N["test_recall_max"] = round(float(te2["recall"].max()), 3)
    N["test_recall_by_channel"] = {short(c): round(float(r), 3) for c, r in zip(te2["channel"], te2["recall"])}

# ---------------------------------------------------------------- 선행성 (표 + Fig 4)
pp = load(SRC / "stage2_precedence_test_pooled.csv")
pc = load(SRC / "stage2_precedence_test_by_channel.csv")
cm = load(SRC / "stage2_diagD_group_proportion_tests_cmh.csv")
ct = load(SRC / "stage2_precedence_crossing_times.csv")
if pp is not None:
    frac_cols = [c for c in pp.columns if c.startswith("frac_")]
    pp2 = pp.copy()
    pp2["frac_first_event"] = pp2[frac_cols].bfill(axis=1).iloc[:, 0]
    Tp = pd.DataFrame({"pair": pp2["pair"], "n_pairs(both crossed; ties included)": pp2["n_pairs"], "median_diff_sec": pp2["median_diff_sec"],
                       "frac_first_is_left": pp2["frac_first_event"].map(lambda x: fnum(x, 3)),
                       "p": pp2["p_value"].map(fmt_p), "p_bonf": pp2["p_bonferroni"].map(fmt_p)})
    save_table(Tp, "T_precedence_pooled")
    N["precedence_pooled"] = {r.pair: {"n_pairs": int(r.n_pairs), "p_bonf": None if pd.isna(r.p_bonferroni) else float(r.p_bonferroni),
                                       "frac_left_first": None if pd.isna(r.frac_first_event) else round(float(r.frac_first_event), 3)}
                              for r in pp2.itertuples()}
if pc is not None:
    pc2 = infilt(pc)
    Tc = pd.DataFrame({"channel": pc2["channel"].map(short), "pair": pc2["pair"], "n_pairs": pc2["n_pairs"],
                       "p": pc2["p_value"].map(fmt_p), "p_bonf": pc2["p_bonferroni"].map(fmt_p), "note": pc2["note"].fillna("")})
    save_table(Tc, "T_precedence_by_channel")
    lv = pc2[pc2["pair"] == "level vs diff"]
    N["precedence_by_channel_level_vs_diff_sig_channels"] = int((lv["p_bonferroni"] < ALPHA).sum())
    N["precedence_by_channel_level_vs_diff_tested"] = int(lv["p_bonferroni"].notna().sum())
if cm is not None:
    Tm = pd.DataFrame({"pair": cm["pair"], "n_strata": cm["n_strata"], "chi2": cm["chi2"].map(lambda x: fnum(x, 1)),
                       "OR_MH": cm["or_mh"].map(lambda x: fnum(x, 3)), "p_bonf": cm["p_value_bonferroni"].map(fmt_p)})
    save_table(Tm, "T_precedence_cmh")
    N["cmh"] = {r.pair: {"chi2": float(r.chi2), "or_mh": float(r.or_mh), "p_bonf": float(r.p_value_bonferroni)} for r in cm.itertuples()}
if ct is not None:
    c = infilt(ct)
    v = c[c["skip_reason"].isna()] if "skip_reason" in c.columns else c
    N["cross_n_segments"] = int(len(c)); N["cross_n_analyzable"] = int(len(v))
    for k in ["level", "diff", "diff2"]:
        N[f"cross_{k}_never_crossed"] = int(v[f"{k}_cross_sec"].isna().sum())
    both = v.dropna(subset=["level_cross_sec", "diff_cross_sec"])
    N["cross_pairs_level_and_diff_both"] = int(len(both))
    N["cross_pairs_diff_first"] = int((both["diff_cross_sec"] < both["level_cross_sec"]).sum())
    N["cross_pairs_tie"] = int((both["diff_cross_sec"] == both["level_cross_sec"]).sum())
    N["cross_pairs_level_first"] = int((both["diff_cross_sec"] > both["level_cross_sec"]).sum())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    rng_j = np.random.default_rng(0)
    for i, ch in enumerate(CHS):
        d = both[both["channel"] == ch]
        lag = (d["diff_cross_sec"] - d["level_cross_sec"]).to_numpy()
        ax.scatter(i + rng_j.uniform(-0.22, 0.22, len(lag)), lag, s=7, color=tcol(ch), alpha=0.6, lw=0)
        if len(lag):
            ax.hlines(np.median(lag), i - 0.3, i + 0.3, color="k", lw=1.4)
        ax.text(i, 1.02, f"n={len(lag)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(len(CHS))); ax.set_xticklabels([short(c) for c in CHS])
    ax.set_ylabel("t(diff) − t(level)  [s]  (<0: diff first)")
    ax.text(-0.18, 1.12, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = axes[1]
    if cm is not None:
        y = np.arange(len(cm))[::-1]
        ax.scatter(cm["or_mh"], y, color=C_ANOM, s=22, zorder=3)
        ax.axvline(1, color="#888", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yticks(y); ax.set_yticklabels(cm["pair"], fontsize=6)
        for yy, r in zip(y, cm.itertuples()):
            ax.text(1.02, yy, f"{stars(r.p_value_bonferroni)}  χ²={r.chi2:.1f}", transform=ax.get_yaxis_transform(), fontsize=6, va="center")
        ax.set_xlabel("Mantel–Haenszel OR (channel-stratified)")
        ax.text(-0.18, 1.12, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig4_precedence")

# ---------------------------------------------------------------- 지속성(transient) 프로파일
tp = load(SRC / "stage2_diagA_diff_temporal_profile.csv")
if tp is not None:
    tp = infilt(tp)
    pat = lambda s: s.astype(str).str.split("(").str[0].str.strip()
    rows = []
    for key, _ in FNAMES:
        vc = pat(tp[f"{key}_pattern"]).value_counts()
        for pname, n_ in vc.items():
            rows.append({"feature": key, "pattern": pname, "n": int(n_), "share_%": round(100 * n_ / vc.sum(), 1)})
            N[f"persist_{key}_{pname}_%"] = round(100 * n_ / vc.sum(), 1)
        dec = vc[[p_ for p_ in vc.index if "판정불가" not in p_ and "undecid" not in p_.lower()]]
        if dec.sum() > 0 and "transient" in dec.index:
            N[f"persist_{key}_transient_of_decidable_%"] = round(100 * dec["transient"] / dec.sum(), 1)
    Tper = pd.DataFrame(rows)
    save_table(Tper, "T_persistence")
    pats = list(dict.fromkeys(Tper["pattern"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    left = np.zeros(len(FNAMES))
    pal = ["#1b6ca8", "#e08a1e", "#9a9a9a", "#1a9988"]
    for j, pname in enumerate(pats):
        vals = np.array([Tper[(Tper.feature == k) & (Tper.pattern == pname)]["share_%"].sum() for k, _ in FNAMES])
        ax.barh(range(len(FNAMES)), vals, left=left, color=pal[j % 4], label=pname)
        left += vals
    ax.set_yticks(range(len(FNAMES))); ax.set_yticklabels([l for _, l in FNAMES]); ax.invert_yaxis()
    ax.set_xlabel("Share of anomalous segments (%)"); ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save_fig(fig, "figS_persistence")

# ---------------------------------------------------------------- 이질성 메타분석 (표 + Fig 7)
mp = load(SRC / "stage2_meta_frac_significant_pooled.csv")
mbc = load(SRC / "stage2_meta_frac_significant_by_channel.csv")
loo = load(SRC / "stage2_meta_loo_influential_flags_v2.csv")
inv = lambda x: np.sin(np.clip(x, 0, np.pi / 2)) ** 2     # 각변환 역변환 (yi_prop 열과 일치 확인됨)
if mp is not None:
    T5b = pd.DataFrame({"feature": mp["feature"], "k": mp["k"], "pooled_prop": mp["pooled_proportion"].map(lambda x: fnum(x, 3)),
                        "CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(mp["ci_low_proportion"], mp["ci_high_proportion"])],
                        "I2_%": mp["I2"].map(lambda x: fnum(x, 1)), "p_Q": mp["p_Q"].map(fmt_p), "tau2": mp["tau2"].map(lambda x: fnum(x, 3))})
    save_table(T5b, "T5b_meta_pooled")
    N["meta_I2_%"] = {r.feature: round(float(r.I2), 1) for r in mp.itertuples()}
    N["meta_pooled_prop"] = {r.feature: round(float(r.pooled_proportion), 3) for r in mp.itertuples()}
if mp is not None and mbc is not None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, (key, lab), tag in zip(axes, FNAMES, "abc"):
        d = infilt(mbc[mbc["feature"] == key])
        for i, r in enumerate(d.itertuples()):
            ax.plot([inv(r.ci_low), inv(r.ci_high)], [i, i], color=tcol(r.channel), lw=1.2)
            ax.scatter(r.yi_prop, i, color=tcol(r.channel), s=14, zorder=3)
        pr = mp[mp["feature"] == key]
        if len(pr):
            r = pr.iloc[0]; y0 = len(d) + 0.3
            ax.fill([r.ci_low_proportion, r.pooled_proportion, r.ci_high_proportion, r.pooled_proportion],
                    [y0, y0 + 0.28, y0, y0 - 0.28], color="k")
            ax.text(0.98, 0.04, f"I²={r.I2:.0f}%", transform=ax.transAxes, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(d))); ax.set_yticklabels([short(c) for c in d["channel"]])
        ax.set_xlim(0, 1); ax.set_xlabel(f"{lab}: segments significant (prop.)")
        ax.text(-0.1, 1.06, f"({tag})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig7_heterogeneity")
if loo is not None:
    infl = loo[loo["influential_v2"].astype(str) == "True"]
    N["loo_influential"] = [f"{r.metric}|{r.feature}|{short(r.excluded_channel)}" for r in infl.itertuples()]
    Tl = pd.DataFrame({"metric": loo["metric"], "feature": loo["feature"], "excluded": loo["excluded_channel"].map(short),
                       "I2_drop": loo["I2_drop_from_full"].map(lambda x: fnum(x, 1)),
                       "rel_shift": loo["relative_shift"].map(lambda x: fnum(x, 3)), "influential": loo["influential_v2"]})
    save_table(Tl, "T_loo")

# ---------------------------------------------------------------- 모델군 일치도 (표 + Fig 8)
ag = load(SRC / "modelclass_agreement_summary.csv")
fs = load(SRC / "modelclass_family_summary.csv")
fi = load(SRC / "modelclass_feature_importance.csv")
bs = load(SRC / "modelclass_bootstrap_stability.csv")
if ag is not None:
    a = ag.iloc[0]
    N["modelclass"] = {"n_models": int(a["n_models"]), "kendalls_W": round(float(a["kendalls_W"]), 3),
                       "chi2": round(float(a["chi2_stat"]), 2), "chi2_p": float(a["chi2_p_value"]),
                       "n_level_not_top": int(a["n_models_level_not_top"]),
                       "binom_p": float(a["binomial_p_value(level_not_top_gt_chance)"])}
if bs is not None:
    N["modelclass_bootstrap"] = {r.model: {"n_boot": int(r.n_boot), "frac": float(r.frac_diff_or_diff2_beats_level)} for r in bs.itertuples()}
if fs is not None:
    kw = ["Linear", "Probab", "Instance", "Kernel", "Tree", "Convol", "Recurr", "Attention", "State"]
    def _ord(f):
        for i, k in enumerate(kw):
            if k.lower() in str(f).lower():
                return i
        return len(kw)
    fs2 = fs.assign(_o=fs["family"].map(_ord)).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    T6 = pd.DataFrame({"family": fs2["family"], "n_models": fs2["n_models"], "level": fs2["mean_level"].map(lambda x: fnum(x, 3)),
                       "diff": fs2["mean_diff"].map(lambda x: fnum(x, 3)), "diff2": fs2["mean_diff2"].map(lambda x: fnum(x, 3)),
                       "diff+diff2 > level": fs2["diff_plus_diff2_beats_level"]})
    save_table(T6, "T6_modelclass_families")
    N["modelclass_families_total"] = int(len(fs2))
    N["modelclass_families_diff_beats_level"] = int(fs2["diff_plus_diff2_beats_level"].astype(str).eq("True").sum())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.22 * len(fs2) + 1.2), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]; y = np.arange(len(fs2)); left = np.zeros(len(fs2))
    for col, lab, colr in (("mean_level", "level", "#9a9a9a"), ("mean_diff", "diff", "#1b6ca8"), ("mean_diff2", "diff²", "#e08a1e")):
        ax.barh(y, fs2[col], left=left, color=colr, label=lab); left += fs2[col].to_numpy()
    ax.set_yticks(y); ax.set_yticklabels(fs2["family"], fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("Mean attribution share"); ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.text(-0.5, 1.06, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = axes[1]
    if fi is not None:
        pv = fi.pivot_table(index="model", columns="feature", values="importance_share")
        rk = pv.rank(axis=1, ascending=False, method="min")["level"]
        cnt = [int((rk == k).sum()) for k in (1, 2, 3)]
        N["modelclass_level_rank_counts_recomputed"] = {"rank1": cnt[0], "rank2": cnt[1], "rank3": cnt[2], "n_models": int(len(rk))}
        ax.bar(["1st", "2nd", "3rd"], cnt, color=["#b00020", "#9a9a9a", "#9a9a9a"])
        for i, v_ in enumerate(cnt):
            ax.text(i, v_ + 0.1, str(v_), ha="center", fontsize=7)
        ax.set_ylabel("Models"); ax.set_xlabel("Rank of 'level' among 3 features")
        if ag is not None:
            ax.text(0.5, 0.92, f"W={a['kendalls_W']:.3f}\nbinomial p={fmt_p(a['binomial_p_value(level_not_top_gt_chance)'])}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
        ax.text(-0.25, 1.06, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig8_modelclass")

# ---------------------------------------------------------------- 경보 임계값 tau* (표 + Fig 9)
ts = load(SRC / "stage4_v7_8_final_summary.csv")
if ts is not None:
    ts = infilt(ts).reset_index(drop=True)
    c1, c2 = "unbounded_plateau(non_identifiable)", "near_degenerate_plateau(practically_non_identifiable)"
    ident = ~(ts[c1].astype(str).eq("True") | ts[c2].astype(str).eq("True")) if (c1 in ts and c2 in ts) else pd.Series([np.nan] * len(ts))
    collapsed = (ts["tau_star_boot_ci_low"] == ts["tau_star_boot_ci_high"])
    T7 = pd.DataFrame({"channel": ts["channel"].map(short), "n_anomaly": ts["n_anomaly"],
                       "tau*_Youden": ts["tau_star_youden"].map(lambda x: f"{x:.6f}"),
                       "recall": ts["recall"].map(lambda x: fnum(x, 3)), "FA": ts["fa_rate"].map(lambda x: fnum(x, 3)),
                       "J": ts["youden_j"].map(lambda x: fnum(x, 3)),
                       "tau* boot 95% CI": [f"[{a:.6f}, {b:.6f}]" for a, b in zip(ts["tau_star_boot_ci_low"], ts["tau_star_boot_ci_high"])],
                       "CI_collapsed": collapsed.map({True: "yes", False: ""}),
                       "CVaR_identifiable": ident.map({True: "yes", False: "no"}),
                       "small_sample": ts["small_sample_flag"]})
    save_table(T7, "T7_tau_star")
    N["tau_star"] = {short(r.channel): {"tau": float(r.tau_star_youden), "ci": [float(r.tau_star_boot_ci_low), float(r.tau_star_boot_ci_high)],
                                        "recall": round(float(r.recall), 3), "fa": round(float(r.fa_rate), 3)} for r in ts.itertuples()}
    N["tau_ci_collapsed_channels"] = [short(c) for c, f in zip(ts["channel"], collapsed) if f]
    N["cvar_identifiable_channels"] = [short(c) for c, f in zip(ts["channel"], ident) if f is True or f == True]
    N["small_sample_channels"] = [short(c) for c, f in zip(ts["channel"], ts["small_sample_flag"]) if str(f) == "True"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]; y = np.arange(len(ts))
    for i, r in enumerate(ts.itertuples()):
        col = tcol(r.channel)
        ax.plot([r.tau_star_boot_ci_low, r.tau_star_boot_ci_high], [i, i], color=col, lw=1.4)
        ax.scatter(r.tau_star_youden, i, color=col, s=20, zorder=3, marker="o" if not collapsed[i] else "D")
        ax.text(1.02, i, ("● CVaR-identifiable" if ident[i] is True or ident[i] == True else "○ not identifiable") + ("  | CI collapsed" if collapsed[i] else ""),
                transform=ax.get_yaxis_transform(), fontsize=5.5, va="center")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([short(c) for c in ts["channel"]]); ax.invert_yaxis()
    ax.set_xlabel(r"$\tau^*$ (Youden), bootstrap 95% CI")
    ax.text(-0.18, 1.06, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = axes[1]
    for r in ts.itertuples():
        col = tcol(r.channel)
        ax.errorbar(r.fa_rate, r.recall, xerr=[[r.fa_rate - r.fa_boot_ci_low], [r.fa_boot_ci_high - r.fa_rate]],
                    yerr=[[r.recall - r.recall_boot_ci_low], [r.recall_boot_ci_high - r.recall]], fmt="o", ms=4, color=col, lw=0.8, capsize=1.5)
        ax.annotate(short(r.channel), (r.fa_rate, r.recall), textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set_xlabel("False-alarm rate"); ax.set_ylabel("Recall")
    ax.text(-0.18, 1.06, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig9_tau_star")

# ================================================================== 문장 자동 생성
L = TEXT.append
L("# 논문 수치 요약 (자동 생성 — 원본 CSV 기준, 수동 수정 금지)\n")
L("## 데이터")
for k in ["channels", "n_segments_total_analysed", "n_anomalous_total_analysed", "n_reliable_total", "n_reliable_train",
          "n_reliable_test", "rows_total", "rows_train", "rows_test", "train_row_share_%", "n_draws_per_segment",
          "onset_ratio_mean", "onset_ratio_median"]:
    if k in N:
        L(f"- {k}: {N[k]}")
if "n_reliable_total" in N:
    L(f"\n> Data 절: 분석 대상은 {len(CHS)}개 채널의 신뢰도({'+'.join(TIERS)}) 이상 이상 세그먼트 {N['n_reliable_total']}개"
      f"(train {N.get('n_reliable_train')}, test {N.get('n_reliable_test')})이다. 행 기준 train 비율은 {N.get('train_row_share_%')}%이다.")
if "onset_ratio_mean" in N:
    L(f"> onset 위치비율 평균은 {N['onset_ratio_mean']}이다(세그먼트 뒷부분 쏠림 → Limitations에 라벨링 한계와 함께 명시).")

if mw_full is not None:
    L("\n## 결과 R4 — 위약 대조 (Bonferroni, 검정 수 = %d)" % N["n_tests_bonferroni"])
    for f, lab, key in FEATS:
        L(f"- {key}: 유의 {N[f'{key}_n_channels_significant']}/{N[f'{key}_n_channels_tested']} 채널, "
          f"p_bonf 최소 {fmt_p(N[f'{key}_p_bonf_min'])}, 유의 채널 중 최대 "
          f"{fmt_p(N[f'{key}_p_bonf_max_among_sig']) if N[f'{key}_p_bonf_max_among_sig'] is not None else 'n/a'}, "
          f"이상 중앙값 ≤ 위약 중앙값 채널 {N[f'{key}_n_channels_anom_median_le_placebo']}개")
    sig_feats = [k for _, _, k in FEATS if N[f"{k}_n_channels_significant"] == N[f"{k}_n_channels_tested"] and N[f"{k}_n_channels_tested"] > 0]
    nosig = [k for _, _, k in FEATS if N[f"{k}_n_channels_significant"] == 0]
    L(f"\n> 문장: Against the placebo pool, {', '.join(sig_feats) if sig_feats else '(none)'} variance changes were significant in all tested channels "
      f"(adjusted p ≤ {fmt_p(max([N[f'{k}_p_bonf_max_among_sig'] for k in sig_feats if N[f'{k}_p_bonf_max_among_sig'] is not None], default=np.nan))}), "
      f"whereas {', '.join(nosig) if nosig else '(none)'} showed no significant difference in any channel.")

if tri is not None:
    L("\n## 결과 R6 — 3중 검증")
    L(f"- 조합 {N['triple_n_combinations']}개 중 판정 가능 {N['triple_n_testable']}개, 완전 일치 {N['triple_n_agree']}개 ({N['triple_agree_%']}%)")
    L(f"- 불일치: {N['triple_disagree'] or '없음'} / 판정불가: {N['triple_undecidable'] or '없음'}")
    for k, v in N["triple_disagree_detail"].items():
        L(f"- 불일치 {k}: p_bonf(full, train, test) = {', '.join(fmt_p(x) for x in v)}  (점진 악화 여부를 본문에 서술)")
    L(f"\n> 문장: Of {N['triple_n_testable']} testable channel×feature combinations, {N['triple_n_agree']} "
      f"({N['triple_agree_%']}%) agreed across full, train-only and test-only slices; {len(N['triple_undecidable'])} could not be tested (n<{args.min_n}) "
      f"and are reported as undecidable rather than non-significant.")

if "frac_significant" in N:
    L("\n## 결과 R5 — 세그먼트 단위 유의 비율")
    for k, v in N["frac_significant"].items():
        L(f"- {k}: {v}")


if "ablation_selected" in N:
    sel = N["ablation_selected"]
    L("\n## 결과 R1 — 탐지·ablation")
    L(f"- 선택 조합: mixture={sel['mixture']}, forgetting={sel['forgetting']} (train macro MCC={sel['train_macro_mcc']:.3f}, 점수 산출 채널 수={sel['n_scoring_channels']})")
    if "test_macro_mcc_final_channels" in N:
        L(f"- test 매크로 MCC(최종 {len(CHS)}채널)={N['test_macro_mcc_final_channels']}, recall 범위 {N['test_recall_min']}–{N['test_recall_max']}")
        L("> 주의: ablation 점수 산출 채널 수와 최종 분석 채널 수가 다르면 본문에 명시. recall이 낮은 채널은 '탐지 성능이 아니라 시그니처 분석이 본 논문의 초점'임을 서술.")
if "precedence_pooled" in N:
    L("\n## 결과 R2 — 선행성")
    for k, v in N["precedence_pooled"].items():
        L(f"- pooled Wilcoxon [{k}]: n_pairs(양쪽 교차, 동률 포함)={v['n_pairs']}, p_bonf={fmt_p(v['p_bonf'])}, 왼쪽 사건이 먼저인 비율={v['frac_left_first']}")
    if "precedence_by_channel_level_vs_diff_tested" in N:
        L(f"- 채널별 level vs diff: {N['precedence_by_channel_level_vs_diff_sig_channels']}/{N['precedence_by_channel_level_vs_diff_tested']} 채널만 보정 후 유의 (표본 작음 → 선행성 주장은 pooled 수준으로 한정)")
if "cmh" in N:
    for k, v in N["cmh"].items():
        L(f"- CMH [{k}]: chi2={v['chi2']:.1f}, OR_MH={v['or_mh']:.3f}, p_bonf={fmt_p(v['p_bonf'])}")
    L("> 주의: CMH 표는 'group proportion test'(stage2_diagD)입니다. 비교 군의 정의(이상 vs 정상 대조 등)를 코드에서 확인해 Methods에 그대로 서술하고, 해석 문장은 그 정의에 맞추세요.")
if "cross_n_analyzable" in N:
    L(f"- 교차시각: 분석 가능 세그먼트 {N['cross_n_analyzable']}개 중 level 미교차 {N['cross_level_never_crossed']}, diff 미교차 {N['cross_diff_never_crossed']}, diff² 미교차 {N['cross_diff2_never_crossed']}; "
      f"level·diff 모두 교차 {N['cross_pairs_level_and_diff_both']}쌍 중 diff 먼저 {N['cross_pairs_diff_first']}, 동률 {N['cross_pairs_tie']}, level 먼저 {N['cross_pairs_level_first']}")
if any(k.startswith("persist_") for k in N):
    L("\n## 결과 R3 — 지속성 (판정불가 비율을 함께 보고할 것; 'of_decidable'은 판정불가 제외 기준)")
    for k, v in N.items():
        if k.startswith("persist_"): L(f"- {k}: {v}")
if "meta_I2_%" in N:
    L("\n## 결과 R5 — 메타분석 (세그먼트 내부 pre/post 유의 비율 기준)")
    for f_, v in N["meta_I2_%"].items():
        L(f"- {f_}: pooled 비율={N['meta_pooled_prop'][f_]}, I²={v}%")
    L(f"- LOO 영향 채널: {N.get('loo_influential') or '없음'}")
    L("> 주의: 이 값은 위약 대조 이전의 내부 비교입니다. level의 내부 유의 비율이 높고(위약 대조에서는 비유의) diff/diff²의 pooled 비율은 낮다는 점을 혼동하지 않게 서술하세요.")
if "modelclass" in N:
    m_ = N["modelclass"]
    L("\n## 결과 R7 — 모델군 일치")
    L(f"- 모델 {m_['n_models']}개, level이 최상위가 아닌 모델 {m_['n_level_not_top']}개 (이항 p={fmt_p(m_['binom_p'])}), Kendall's W={m_['kendalls_W']}, chi2={m_['chi2']} (p={fmt_p(m_['chi2_p'])})")
    if "modelclass_families_total" in N:
        L(f"- diff+diff² 합이 level보다 큰 모델군: {N['modelclass_families_diff_beats_level']}/{N['modelclass_families_total']}")
    if "modelclass_level_rank_counts_recomputed" in N:
        rc = N["modelclass_level_rank_counts_recomputed"]
        L(f"- 중요도 표로 재계산한 level 순위: 1위 {rc['rank1']}, 2위 {rc['rank2']}, 3위 {rc['rank3']} (요약 CSV와 불일치 시 점검)")
    for k, v in N.get("modelclass_bootstrap", {}).items():
        L(f"- 부트스트랩 {k}: {v['frac']*100:.0f}% (n_boot={v['n_boot']})")
if "tau_star" in N:
    L("\n## 결과 R8 — 경보 임계값")
    for ch, v in N["tau_star"].items():
        L(f"- {ch}: tau*={v['tau']}, CI=[{v['ci'][0]}, {v['ci'][1]}], recall={v['recall']}, FA={v['fa']}")
    L(f"- CVaR 식별 가능 채널: {N['cvar_identifiable_channels'] or '없음'} / CI가 한 점으로 붕괴한 채널: {N['tau_ci_collapsed_channels'] or '없음'} / 소표본: {N['small_sample_channels'] or '없음'}")
    L("> 주의: tau* CI 폭이 0인 채널은 정밀하다는 뜻이 아니라 임계값 격자가 거칠다는 뜻일 수 있으므로 본문에 '격자 해상도 한계'로 서술.")

(OUT / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding="utf-8")
(OUT / "paper_numbers.md").write_text("\n".join(TEXT), encoding="utf-8")
print(f"\n[완료] {OUT}")