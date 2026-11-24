#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper 2 pipeline: derivative-variance signature of anomalies (OPS-SAT / OPSSAT-AD)
==================================================================================
단일 CLI 스크립트. 서브커맨드(step):

  prep       샘플 흐름표(T1) + 세그먼트 피처표 + 분석 동결 파일(analysis_plan_freeze.json)
  e1         onset 무관 세그먼트 수준 판별 (그룹 AUC, ΔAUC, cluster bootstrap) + 선택편향 표
  e2         onset 무관 scan 통계량 (모든 pivot 탐색, surrogate 귀무로 길이 보정)
  e3         onset 정의 4종(A:BOCPD-KF, B:PELT, C:level-CUSUM, D:level-BOCPD) 위약 대조 반복
  e4         이상 세그먼트 내부 위약(onset-locking) + 이벤트 스터디 그림
  e5         모델군 재설계 (그룹 제거 ΔAUC, 단일 표현 학습, 정규화 2종, cluster bootstrap)
  e8         정상 하위라벨(a2/a3/a4) 및 sampling 층화 위약 대조
  ext-build  외부 시계열(point-wise label) -> 세그먼트 long CSV 변환 (E6 준비)
  ext-run    외부 세그먼트 CSV에 E1+E2 동결 코드 그대로 적용 (E6)
  all        prep,e1,e2,e4,e8,e3,e5 순차 실행 (--skip-e5 로 e5 제외)

예)
  python paper2_pipeline.py all --base-dir "$OPSSAT_BASE_DIR"
  python paper2_pipeline.py e1  --base-dir ... --n-boot 2000
  python paper2_pipeline.py all --base-dir ... --quick          # 스모크 테스트(축소 설정)

입력(기본값): BASE/segments.csv, BASE/causal_pipeline_outputs/stage2_mc_draws_raw.csv,
              BASE/causal_pipeline_outputs/stage2_mc_segment_summary_corrected.csv
출력: BASE/paper2_outputs/*.csv, *.png, analysis_plan_freeze.json

주의(기존 코드 대비 의도적 변경)
  * E2: 순환이동(circular shift)은 scan 통계량에 부적합(이음매 인공물) -> 위상무작위화(phase
    randomization) surrogate를 기본 귀무로 사용. --e2-null block 으로 블록 순열도 가능.
  * E5: 기존 딥러닝 코드는 early-stopping 을 test fold 의 val-loss 로 했음(누수). 여기서는 train fold
    내부 그룹 hold-out 으로 교체.
  * E5: SHAP/IG 중요도 대신 모든 모델에 동일 지표(그룹 제거 재학습 AUC, ΔAUC)를 사용.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

BURN_IN = 5
FEATS3 = ["level_d_abs", "diff_logvar", "diff2_logvar"]
A = None      # argparse Namespace (전역 설정)
OUT = None    # 출력 디렉터리


# =============================================================================
# 공통 유틸
# =============================================================================
def sec(t):
    print("\n" + "=" * 90 + "\n" + t + "\n" + "=" * 90, flush=True)


def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"[저장] {p}  ({len(df)} rows)", flush=True)


def det_seed(*parts):
    key = "|".join(repr(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def fast_auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_auc(y, scores, contrasts, n_boot, rng):
    """층화 bootstrap. scores: name->array. contrasts: [(name, a, b)] = AUC(a)-AUC(b).
    반환: {name: (point, lo, hi)} (AUC와 contrast 모두)"""
    y = np.asarray(y).astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    names = list(scores)
    pt = {k: fast_auc(y, scores[k]) for k in names}
    out = {}
    if len(pos) < 2 or len(neg) < 2:
        for k in names:
            out[k] = (pt[k], np.nan, np.nan)
        for c, a, b in contrasts:
            out[c] = (pt[a] - pt[b], np.nan, np.nan)
        return out
    yb = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    acc = {k: [] for k in names}
    accc = {c: [] for c, _, _ in contrasts}
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        a_ = {k: fast_auc(yb, scores[k][idx]) for k in names}
        for k in names:
            acc[k].append(a_[k])
        for c, a, b in contrasts:
            accc[c].append(a_[a] - a_[b])
    for k in names:
        out[k] = (pt[k], float(np.nanpercentile(acc[k], 2.5)), float(np.nanpercentile(acc[k], 97.5)))
    for c, a, b in contrasts:
        out[c] = (pt[a] - pt[b], float(np.nanpercentile(accc[c], 2.5)), float(np.nanpercentile(accc[c], 97.5)))
    return out


def cluster_boot_contrasts(y, groups, preds, contrasts, n_boot, rng):
    """그룹(cluster) bootstrap. preds: name->OOF 예측. contrasts: [(name,a,b)]."""
    y = np.asarray(y).astype(bool)
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
    pt = {c: fast_auc(y, preds[a]) - fast_auc(y, preds[b]) for c, a, b in contrasts}
    acc = {c: [] for c, _, _ in contrasts}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        yb = y[idx]
        if yb.sum() < 2 or (~yb).sum() < 2:
            continue
        cache = {}
        for c, a, b in contrasts:
            for k in (a, b):
                if k not in cache:
                    cache[k] = fast_auc(yb, preds[k][idx])
            acc[c].append(cache[a] - cache[b])
    return {c: (pt[c],
                float(np.nanpercentile(acc[c], 2.5)) if acc[c] else np.nan,
                float(np.nanpercentile(acc[c], 97.5)) if acc[c] else np.nan) for c, _, _ in contrasts}


# =============================================================================
# 데이터 적재
# =============================================================================
def build_store(seg):
    cols = seg.columns
    sort_cols = ["channel", "segment"] + (["timestamp"] if "timestamp" in cols else [])
    seg = seg.sort_values(sort_cols, kind="stable")
    store = {}
    for (ch, sid), g in seg.groupby(["channel", "segment"], sort=False):
        store[(ch, int(sid))] = dict(
            values=g["value"].to_numpy(float),
            anomaly=int(g["anomaly"].iloc[0]),
            train=int(g["train"].iloc[0]) if "train" in cols else -1,
            sampling=float(g["sampling"].iloc[0]) if "sampling" in cols else 1.0,
            label=str(g["label"].iloc[0]) if "label" in cols else
            ("anomaly" if int(g["anomaly"].iloc[0]) == 1 else "nominal"),
        )
    return store


def load_segments(path, channels=None):
    seg = pd.read_csv(path)
    if "timestamp" in seg.columns:
        try:
            seg["timestamp"] = pd.to_datetime(seg["timestamp"])
        except Exception:
            pass
    if channels is not None:
        seg = seg[seg["channel"].isin(channels)]
    return seg.reset_index(drop=True)


def load_canonical(stage_dir):
    pd_ = Path(stage_dir) / "stage2_mc_draws_raw.csv"
    pc_ = Path(stage_dir) / "stage2_mc_segment_summary_corrected.csv"
    if not pd_.exists() or not pc_.exists():
        return None
    draws = pd.read_csv(pd_)
    corr = pd.read_csv(pc_)
    rel = corr[corr["reliability_tier"].isin(["high", "medium"])][["channel", "segment"]]
    co = (draws.merge(rel, on=["channel", "segment"])
          .groupby(["channel", "segment"])["onset_idx_sampled"].median().round().astype(int))
    return {(ch, int(s)): int(v) for (ch, s), v in co.items()}


# =============================================================================
# 세그먼트 피처 (onset 무관) — OPSSAT 와 외부 데이터에 동일 적용
# =============================================================================
GROUPS = {
    "level": ["mean", "skew", "kurtosis", "half_shift", "abs_slope"],
    "disp": ["var", "range", "iqr"],
    "deriv": ["diff_var", "diff2_var", "diff_peaks", "diff2_peaks", "rough1", "rough2"],
    "cov": ["sampling", "len", "duration"],
}
ALL_FEATS = [f for g in GROUPS.values() for f in g]


def _npeaks(x):
    if len(x) < 3:
        return 0
    pt = np.ptp(x)
    if pt <= 0:
        return 0
    pk, _ = find_peaks(x, prominence=0.1 * pt)
    return int(len(pk))


def seg_features(v, sampling):
    n = len(v)
    d1 = np.diff(v)
    d2 = np.diff(d1)
    var = float(np.var(v, ddof=1))
    sd = math.sqrt(var)
    f = {"len": n, "sampling": sampling, "duration": n * sampling, "mean": float(v.mean())}
    f["skew"] = float(stats.skew(v)) if sd > 0 else 0.0
    f["kurtosis"] = float(stats.kurtosis(v)) if sd > 0 else 0.0
    h = n // 2
    a, b = v[:h], v[h:]
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) if h > 1 and len(b) > 1 else 0.0
    denom = pooled if pooled > 0 else (sd if sd > 0 else 1.0)
    f["half_shift"] = float(abs(b.mean() - a.mean()) / denom)
    slope = np.polyfit(np.arange(n), v, 1)[0]
    f["abs_slope"] = float(abs(slope) * n / (sd if sd > 0 else 1.0))
    f["var"] = var
    f["range"] = float(np.ptp(v))
    f["iqr"] = float(np.subtract(*np.percentile(v, [75, 25])))
    dv1 = float(np.var(d1, ddof=1)) if len(d1) > 1 else np.nan
    dv2 = float(np.var(d2, ddof=1)) if len(d2) > 1 else np.nan
    f["diff_var"], f["diff2_var"] = dv1, dv2
    f["diff_peaks"], f["diff2_peaks"] = _npeaks(d1), _npeaks(d2)
    f["rough1"] = dv1 / (var + 1e-300)
    f["rough2"] = dv2 / (dv1 + 1e-300)
    return f


def build_table(store):
    rows = []
    for (ch, sid), r in store.items():
        if len(r["values"]) < 8:
            continue
        f = seg_features(r["values"], r["sampling"])
        f.update(channel=ch, segment=sid, anomaly=r["anomaly"], train=r["train"], label=r["label"])
        rows.append(f)
    t = pd.DataFrame(rows)
    for c in ALL_FEATS:
        t[c] = t[c].replace([np.inf, -np.inf], np.nan)
        t[c] = t.groupby("channel")[c].transform(lambda s: s.fillna(s.median()))
        t[c] = t[c].fillna(0.0)
        t["rk_" + c] = t.groupby("channel")[c].rank(pct=True)
    return t


def select_channels(t, min_pos, min_neg):
    g = t.groupby("channel")["anomaly"].agg(["sum", "count"])
    g["neg"] = g["count"] - g["sum"]
    ok = g[(g["sum"] >= min_pos) & (g["neg"] >= min_neg)].index.tolist()
    return sorted(ok), g


# =============================================================================
# 효과 크기 (기존 §3 로직과 동일)
# =============================================================================
def cohens_d_abs(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sd = np.sqrt((va + vb) / 2.0)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return float(abs(np.mean(a) - np.mean(b)) / sd)


def log_var_ratio(pre, post):
    if len(pre) < A.min_w or len(post) < A.min_w:
        return np.nan
    vp, vq = np.var(pre, ddof=1), np.var(post, ddof=1)
    if vp <= 0 or vq <= 0 or not np.isfinite(vp) or not np.isfinite(vq):
        return np.nan
    return float(np.log(vq / vp))


def compute_effects(v, piv):
    out = {"level_d_abs": np.nan, "diff_logvar": np.nan, "diff2_logvar": np.nan}
    out["level_d_abs"] = cohens_d_abs(v[piv:], v[BURN_IN:piv])
    d1 = np.diff(v)
    s1 = max(piv - 1, 0)
    out["diff_logvar"] = log_var_ratio(d1[max(BURN_IN - 1, 0):s1], d1[s1:])
    d2 = np.diff(d1)
    s2 = max(piv - 2, 0)
    out["diff2_logvar"] = log_var_ratio(d2[max(BURN_IN - 2, 0):s2], d2[s2:])
    return out


def valid_pivot_range(n):
    return BURN_IN + A.min_w, n - A.min_w


def make_index(store):
    idx = {}
    for key, r in store.items():
        idx.setdefault((key[0], r["anomaly"]), []).append(key)
    return idx


def placebo_rows(store, idx, ch, ratio_pool, rng, k=None, cap=None, filt=None):
    k = k or A.k_placebo
    cap = cap or A.max_normal
    keys = list(idx.get((ch, 0), []))
    if filt is not None:
        keys = [kk for kk in keys if filt(store[kk])]
    if len(keys) > cap:
        keys = [keys[i] for i in rng.choice(len(keys), cap, replace=False)]
    rows = []
    for key in keys:
        r = store[key]
        v = r["values"]
        n = len(v)
        lo, hi = valid_pivot_range(n)
        if hi < lo:
            continue
        for _ in range(k):
            piv = int(np.clip(int(round(rng.choice(ratio_pool) * n)), lo, hi))
            rows.append(dict(channel=ch, segment=key[1], pivot=piv, sampling=r["sampling"],
                             label=r["label"], **compute_effects(v, piv)))
    return pd.DataFrame(rows)


def anomaly_effects(store, onsets, channels):
    rows, ratios = [], {c: [] for c in channels}
    for (ch, sid), onset in onsets.items():
        if ch not in channels:
            continue
        r = store.get((ch, sid))
        if r is None or r["anomaly"] != 1:
            continue
        v = r["values"]
        n = len(v)
        lo, hi = valid_pivot_range(n)
        if not (lo <= onset <= hi):
            continue
        rows.append(dict(channel=ch, segment=sid, onset=onset, n_points=n, sampling=r["sampling"],
                         **compute_effects(v, onset)))
        ratios[ch].append(onset / n)
    return pd.DataFrame(rows), ratios


def mw_rows(obs, pl, channels, tags, strata_col=None):
    rows = []
    for ch in channels:
        o = obs[obs["channel"] == ch] if len(obs) else obs
        p = pl[pl["channel"] == ch] if len(pl) else pl
        for f in FEATS3:
            a = o[f].dropna().values if len(o) else np.array([])
            b = p[f].dropna().values if len(p) else np.array([])
            row = dict(channel=ch, feature=f, n_anomaly=len(a), n_placebo=len(b), **tags)
            if len(a) < 5 or len(b) < 5:
                row.update(p_value=np.nan, auc_mw=np.nan, median_anomaly=np.nan,
                           median_placebo=np.nan, note="판정불가(n<5)")
            else:
                u, pv = stats.mannwhitneyu(a, b, alternative="greater")
                row.update(p_value=float(pv), auc_mw=float(u / (len(a) * len(b))),
                           median_anomaly=float(np.median(a)), median_placebo=float(np.median(b)), note="")
            rows.append(row)
    return rows


def add_bonf(df, group_cols=None):
    df = df.copy()
    if group_cols:
        n = df.groupby(group_cols)["p_value"].transform(lambda s: max(int(s.notna().sum()), 1))
    else:
        n = max(int(df["p_value"].notna().sum()), 1)
    df["p_bonf"] = (df["p_value"] * n).clip(upper=1.0)
    return df


# =============================================================================
# prep
# =============================================================================
def step_prep(store, table, canon):
    sec("PREP: 샘플 흐름표(T1) + 피처표 + 분석 동결")
    chans, g = select_channels(table, A.min_pos, A.min_neg)
    flow = table.groupby("channel").agg(n_segments=("anomaly", "size"), n_anomaly=("anomaly", "sum"))
    flow["n_normal"] = flow["n_segments"] - flow["n_anomaly"]
    if "train" in table and (table["train"] >= 0).any():
        flow["n_train"] = table.groupby("channel")["train"].sum()
    if canon is not None:
        det = pd.Series([k[0] for k in canon], name="channel").value_counts()
        flow["n_anomaly_with_onset(reliable)"] = det
        flow["onset_coverage"] = flow["n_anomaly_with_onset(reliable)"] / flow["n_anomaly"].replace(0, np.nan)
    flow["included_E1_E2"] = flow.index.isin(chans)
    flow = flow.reset_index()
    print(flow.to_string(index=False))
    save(flow, "t1_sample_flow.csv")
    save(table, "p2_segment_features.csv")

    try:
        gh = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        gh = None
    import sklearn
    freeze = dict(
        frozen_at=datetime.now().isoformat(timespec="seconds"), git_commit=gh,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(A).items()},
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                      scipy=__import__("scipy").__version__, sklearn=sklearn.__version__),
        e1e2_channels=chans,
        hypotheses={
            "H1": "onset-free segment AUC: deriv group > level group, CI(deltaAUC)>0 in >=75% of analyzable channels",
            "H2": "canonical onset log-var-ratio > random pivots of the same anomalous segment",
            "H3": "E3 onset definitions (A,B,C,D): deriv features keep the same sign as definition A",
            "H4": "single-representation: diff-only and diff2-only AUC >= level-only AUC in >=80% of models",
            "H5": "external dataset reproduces H1",
        },
        note="Freeze declared AFTER extensive exploratory analysis; separate confirmatory/exploratory in the paper.")
    (OUT / "analysis_plan_freeze.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[저장] {OUT/'analysis_plan_freeze.json'}")


# =============================================================================
# E1 : onset 무관 세그먼트 수준 판별
# =============================================================================
SETS = {"cov": ["cov"], "level": ["level"], "disp": ["disp"], "deriv": ["deriv"],
        "level+disp": ["level", "disp"], "all": ["level", "disp", "deriv"]}
E1_CONTRASTS = [("deriv_minus_level", "deriv", "level"),
                ("deriv_minus_disp", "deriv", "disp"),
                ("all_minus_level+disp", "all", "level+disp")]


def set_cols(s):
    return ["rk_" + f for g in SETS[s] for f in GROUPS[g]]


def e1_models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=3, random_state=A.seed),
    }


def oof_scores(X, y, strat, make_model, n_repeats, seed):
    from sklearn.model_selection import RepeatedStratifiedKFold
    y = np.asarray(y).astype(int)
    k = max(2, min(5, int(y.sum()), int((1 - y).sum())))
    st = np.asarray(strat)
    _, cnt = np.unique(st, return_counts=True)
    if cnt.min() < k:
        st = y
    acc, c = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed).split(X, st):
        m = make_model()
        m.fit(X[tr], y[tr])
        acc[te] += m.predict_proba(X[te])[:, 1]
        c[te] += 1
    return acc / np.maximum(c, 1)


def match_len(d, rng):
    d = d.copy()
    try:
        d["_bin"] = pd.qcut(d["len"], q=4, duplicates="drop", labels=False)
    except Exception:
        return d
    keep = []
    for _, g in d.groupby("_bin"):
        pos, neg = g[g["anomaly"] == 1], g[g["anomaly"] == 0]
        if len(pos) == 0:
            continue
        m = min(len(neg), 3 * len(pos))
        keep.append(pos)
        keep.append(neg.iloc[rng.choice(len(neg), m, replace=False)] if m > 0 else neg.iloc[:0])
    return pd.concat(keep).drop(columns="_bin") if keep else d.iloc[:0]


def e1_block(name, d, tag, models, rows, uni_rows):
    y = d["anomaly"].to_numpy(int)
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos < A.min_class or nneg < A.min_class:
        return
    rng = np.random.default_rng(det_seed(name, tag, A.seed))
    strat = (d["channel"].astype(str) + "_" + d["anomaly"].astype(str)).to_numpy() if name == "POOLED" else y

    # 단변량 AUC
    scores = {f: d["rk_" + f].to_numpy() for f in ALL_FEATS}
    res = boot_auc(y, scores, [], A.n_boot, rng)
    grp_of = {f: g for g, fs in GROUPS.items() for f in fs}
    for f, (p, lo, hi) in res.items():
        uni_rows.append(dict(tag=tag, block=name, feature=f, group=grp_of[f], auc=p, ci_lo=lo, ci_hi=hi,
                             oriented_auc=max(p, 1 - p), n_pos=npos, n_neg=nneg))

    has_split = "train" in d and set(d["train"].unique()) >= {0, 1}
    for mname, mk in models.items():
        # (1) OOF
        sc = {s: oof_scores(d[set_cols(s)].to_numpy(), y, strat, mk, A.n_repeats, A.seed) for s in SETS}
        res = boot_auc(y, sc, E1_CONTRASTS, A.n_boot, rng)
        for item, (p, lo, hi) in res.items():
            rows.append(dict(tag=tag, block=name, model=mname, evaluation="oof_cv", item=item,
                             kind="contrast" if item in {c for c, _, _ in E1_CONTRASTS} else "auc",
                             point=p, ci_lo=lo, ci_hi=hi, n_pos=npos, n_neg=nneg))
        # (2) 공식 train/test 분할
        if has_split:
            tr, te = (d["train"] == 1).to_numpy(), (d["train"] == 0).to_numpy()
            ytr, yte = y[tr], y[te]
            if yte.sum() >= 3 and (1 - yte).sum() >= 3 and ytr.sum() >= 3 and (1 - ytr).sum() >= 3:
                sc2 = {}
                for s in SETS:
                    X = d[set_cols(s)].to_numpy()
                    m = mk()
                    m.fit(X[tr], ytr)
                    sc2[s] = m.predict_proba(X[te])[:, 1]
                res = boot_auc(yte, sc2, E1_CONTRASTS, A.n_boot, rng)
                for item, (p, lo, hi) in res.items():
                    rows.append(dict(tag=tag, block=name, model=mname, evaluation="official_test", item=item,
                                     kind="contrast" if item in {c for c, _, _ in E1_CONTRASTS} else "auc",
                                     point=p, ci_lo=lo, ci_hi=hi, n_pos=int(yte.sum()), n_neg=int((1 - yte).sum())))


def step_e1(store, table, canon, prefix="", force_channels=None):
    sec(f"{prefix.upper() or ''}E1: onset 무관 세그먼트 수준 판별")
    chans, g = select_channels(table, A.min_pos, A.min_neg)
    if force_channels:
        chans = force_channels
    print(f"분석 채널({len(chans)}): {chans}\n제외 채널: {sorted(set(table['channel']) - set(chans))}")
    models = e1_models()
    rows, uni = [], []
    for tag in ["full", "len_matched"]:
        parts = []
        for ch in chans:
            d = table[table["channel"] == ch]
            if tag == "len_matched":
                d = match_len(d, np.random.default_rng(det_seed(ch, "match", A.seed)))
            parts.append(d)
            e1_block(ch, d, tag, models, rows, uni)
        e1_block("POOLED", pd.concat(parts), tag, models, rows, uni)
    res = pd.DataFrame(rows)
    save(res, f"{prefix}e1_group_auc.csv")
    save(pd.DataFrame(uni), f"{prefix}e1_univariate_auc.csv")

    # H1 판정
    sub = res[(res["tag"] == "full") & (res["item"] == "deriv_minus_level") & (res["block"] != "POOLED")]
    print("\n[H1 판정] deriv_minus_level (AUC) — CI 하한 > 0 인 채널 수")
    h1 = []
    thr = math.ceil(0.75 * max(len(chans), 1))
    for (mname, ev), gg in sub.groupby(["model", "evaluation"]):
        n_ok = int((gg["ci_lo"] > 0).sum())
        h1.append(dict(model=mname, evaluation=ev, n_channels=len(gg), n_ci_above_0=n_ok,
                       threshold=thr, h1_pass=bool(n_ok >= thr and ev == "oof_cv")))
        print(f"  {mname:7s} {ev:14s}: {n_ok}/{len(gg)} (기준 {thr})  "
              f"mean ΔAUC={gg['point'].mean():+.3f}")
    save(pd.DataFrame(h1), f"{prefix}e1_h1_verdict.csv")

    # 선택편향: onset 검출/미검출 이상 세그먼트 비교
    if canon is not None and not prefix:
        rows2 = []
        det_keys = set(canon)
        an = table[(table["anomaly"] == 1) & table["channel"].isin(chans)].copy()
        an["detected"] = [(c, int(s)) in det_keys for c, s in zip(an["channel"], an["segment"])]
        for ch in chans:
            a_ = an[an["channel"] == ch]
            for f in ALL_FEATS:
                x, z = a_[a_["detected"]][f].values, a_[~a_["detected"]][f].values
                if len(x) >= 5 and len(z) >= 5:
                    p = stats.mannwhitneyu(x, z, alternative="two-sided").pvalue
                    rows2.append(dict(channel=ch, feature=f, group=grp(f), n_detected=len(x), n_undetected=len(z),
                                      median_detected=np.median(x), median_undetected=np.median(z), p_two_sided=p))
        save(pd.DataFrame(rows2), "e1_selection_bias_detected_vs_undetected.csv")


def grp(f):
    for g, fs in GROUPS.items():
        if f in fs:
            return g


# =============================================================================
# E2 : scan 통계량
# =============================================================================
def scan_logvar(x, minw):
    n = len(x)
    if n < 2 * minw + 1:
        return np.nan
    sd = x.std()
    if sd <= 0:
        return np.nan
    x = (x - x.mean()) / sd
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    ks = np.arange(minw, n - minw + 1)
    s1, s2 = c1[ks - 1], c2[ks - 1]
    m = n - ks
    vp = (s2 - s1 ** 2 / ks) / (ks - 1)
    q1, q2 = c1[-1] - s1, c2[-1] - s2
    vq = (q2 - q1 ** 2 / m) / (m - 1)
    return float(np.max(np.log((np.maximum(vq, 0) + 1e-3) / (np.maximum(vp, 0) + 1e-3))))


def scan_level(y, minw, detrend):
    n = len(y)
    if n < 2 * minw + 1:
        return np.nan
    if detrend:
        t = np.arange(n)
        y = y - np.polyval(np.polyfit(t, y, 1), t)
    sd = y.std()
    if sd <= 0:
        return np.nan
    y = (y - y.mean()) / sd
    c1, c2 = np.cumsum(y), np.cumsum(y * y)
    ks = np.arange(minw, n - minw + 1)
    s1, s2 = c1[ks - 1], c2[ks - 1]
    m = n - ks
    vp = np.maximum((s2 - s1 ** 2 / ks) / (ks - 1), 0)
    q1, q2 = c1[-1] - s1, c2[-1] - s2
    vq = np.maximum((q2 - q1 ** 2 / m) / (m - 1), 0)
    pooled = np.sqrt((vp + vq) / 2) + 1e-6
    return float(np.max(np.abs(q1 / m - s1 / ks) / pooled))


def phase_surrogate(x, rng):
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(X)))
    ph[0] = 1
    if n % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(np.abs(X) * ph, n) + x.mean()


def block_perm(x, rng, b=None):
    n = len(x)
    b = b or max(5, n // 10)
    blocks = [x[i:i + b] for i in range(0, n, b)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def scan_all(v, minw):
    d1 = np.diff(v)
    d2 = np.diff(d1)
    return {"LV": (scan_logvar(d1, minw), d1), "LV2": (scan_logvar(d2, minw), d2),
            "LS": (scan_level(v, minw, False), v), "LSdt": (scan_level(v, minw, True), v)}


def step_e2(store, table, prefix=""):
    sec(f"{prefix.upper() or ''}E2: onset 무관 scan 통계량 (null={A.e2_null}, K={A.e2_k})")
    chans, _ = select_channels(table, A.min_pos, A.min_neg)
    rows = []
    sur = phase_surrogate if A.e2_null == "phase" else block_perm
    for (ch, sid), r in store.items():
        if ch not in chans or len(r["values"]) < 2 * A.min_w + 6:
            continue
        v = r["values"]
        rng = np.random.default_rng(det_seed(ch, sid, "e2", A.seed))
        obs = scan_all(v, A.min_w)
        row = dict(channel=ch, segment=sid, anomaly=r["anomaly"], train=r["train"], len=len(v))
        nulls = {k: [] for k in obs}
        for _ in range(A.e2_k):
            for k, (o, series) in obs.items():
                if not np.isfinite(o):
                    continue
                s = sur(series, rng)
                if k in ("LV", "LV2"):
                    nulls[k].append(scan_logvar(s, A.min_w))
                else:
                    nulls[k].append(scan_level(s, A.min_w, k == "LSdt"))
        for k, (o, _) in obs.items():
            nv = np.array([x for x in nulls[k] if np.isfinite(x)])
            row[k] = o
            if np.isfinite(o) and len(nv) >= 5:
                row[k + "_z"] = (o - nv.mean()) / (nv.std() + 1e-9)
                row[k + "_p"] = (1 + (nv >= o).sum()) / (len(nv) + 1)
            else:
                row[k + "_z"], row[k + "_p"] = np.nan, np.nan
        rows.append(row)
    seg_df = pd.DataFrame(rows)
    save(seg_df, f"{prefix}e2_scan_segments.csv")

    contr = [("LV_minus_LS", "LV_z", "LS_z"), ("LV_minus_LSdt", "LV_z", "LSdt_z"),
             ("LV2_minus_LS", "LV2_z", "LS_z"), ("LV2_minus_LSdt", "LV2_z", "LSdt_z"),
             ("LV_raw_minus_LS_raw", "LV", "LS"), ("LV2_raw_minus_LSdt_raw", "LV2", "LSdt")]
    out = []
    for ch in chans + ["POOLED"]:
        d = seg_df if ch == "POOLED" else seg_df[seg_df["channel"] == ch]
        d = d.dropna(subset=["LV_z", "LV2_z", "LS_z", "LSdt_z"])
        y = d["anomaly"].to_numpy(int)
        if y.sum() < A.min_class or (1 - y).sum() < A.min_class:
            continue
        rng = np.random.default_rng(det_seed(ch, "e2boot", A.seed))
        sc = {k: d[k].to_numpy() for k in ["LV_z", "LV2_z", "LS_z", "LSdt_z", "LV", "LV2", "LS", "LSdt"]}
        sc["len"] = d["len"].to_numpy(float)
        res = boot_auc(y, sc, contr, A.n_boot, rng)
        cn = {c for c, _, _ in contr}
        for item, (p, lo, hi) in res.items():
            out.append(dict(block=ch, item=item, kind="contrast" if item in cn else "auc", point=p,
                            ci_lo=lo, ci_hi=hi, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                            n_dropped_nan=int(len(seg_df[seg_df['channel'] == ch]) - len(d)) if ch != "POOLED" else np.nan))
    o = pd.DataFrame(out)
    save(o, f"{prefix}e2_scan_auc.csv")
    if len(o):
        print(o[(o["kind"] == "contrast") & (o["item"].isin(["LV_minus_LS", "LV_minus_LSdt"]))]
              .to_string(index=False))


# =============================================================================
# E3 : onset 정의 4종
# =============================================================================
def baseline_z(v, minw):
    n = len(v)
    b = max(minw, n // 5)
    mu, sd = v[:b].mean(), v[:b].std()
    if sd <= 0:
        sd = v.std() if v.std() > 0 else 1.0
    return (v - mu) / sd


def onset_cusum(v, minw, k=0.5, h=5.0):
    z = baseline_z(v, minw)
    n = len(v)
    b = max(minw, n // 5)
    sp = sn = 0.0
    sp0 = sn0 = b
    for t in range(b, n):
        if sp == 0:
            sp0 = t
        if sn == 0:
            sn0 = t
        sp = max(0.0, sp + z[t] - k)
        sn = max(0.0, sn - z[t] - k)
        if sp > h:
            return int(sp0)
        if sn > h:
            return int(sn0)
    return None


def bocpd_mrl(x, hazard=1 / 250., mu0=0., k0=1., a0=1., b0=1., kmax=40.):
    n = len(x)
    probs = np.array([1.0])
    mu, kap, al, be = np.array([mu0]), np.array([k0]), np.array([a0]), np.array([b0])
    mrl = np.zeros(n, int)
    for t in range(n):
        xt = x[t]
        pp = np.nan_to_num(stats.t.pdf(xt, df=2 * al, loc=mu, scale=np.sqrt(be * (kap + 1) / (al * kap))),
                           nan=1e-12, posinf=1e-12)
        new = np.concatenate([[np.sum(probs * pp * hazard)], probs * pp * (1 - hazard)])
        tot = new.sum()
        if tot <= 0 or not np.isfinite(tot):
            new = np.zeros_like(new)
            new[0] = 1.0
        else:
            new = new / tot
        probs = new
        mrl[t] = int(np.argmax(probs))
        nmu = np.concatenate([[mu0], (kap * mu + xt) / (kap + 1)])
        nkap = np.concatenate([[k0], kap + 1])
        nal = np.concatenate([[a0], al + 0.5])
        nbe = np.concatenate([[b0], be + kap * (xt - mu) ** 2 / (2 * (kap + 1))])
        over = nkap > kmax
        if over.any():
            s = np.where(over, kmax / nkap, 1.0)
            nal = np.maximum(np.where(over, nal * s, nal), 1e-3)
            nbe = np.where(over, nbe * s, nbe)
            nkap = np.where(over, kmax, nkap)
        mu, kap, al, be = nmu, nkap, nal, nbe
    return mrl


def onset_bocpd_level(v, minw):
    z = baseline_z(v, minw)
    mrl = bocpd_mrl(z)
    for t in range(1, len(z)):
        if mrl[t] <= 2 and mrl[t - 1] > 5:
            return int(t - mrl[t])
    return None


def onset_pelt(v, minw):
    import ruptures as rpt
    n = len(v)
    if v.std() <= 0:
        return None
    z = ((v - v.mean()) / v.std()).reshape(-1, 1)
    bk = rpt.Pelt(model="l2", min_size=minw, jump=1).fit(z).predict(pen=A.pelt_pen * np.log(n))
    bk = [b for b in bk if b < n]
    if not bk:
        return None
    pts = [0] + bk + [n]
    best, bj = None, -1
    for i in range(1, len(pts) - 1):
        a_, b_ = v[pts[i - 1]:pts[i]], v[pts[i]:pts[i + 1]]
        j = abs(b_.mean() - a_.mean()) / (z.std() * v.std() + 1e-300) if False else abs(b_.mean() - a_.mean()) / (v.std() + 1e-300)
        if j > bj:
            best, bj = pts[i], j
    return int(best)


def step_e3(store, idx, canon):
    sec("E3: onset 정의 4종 반복 (A=KF+BOCPD, B=PELT, C=level-CUSUM, D=level-BOCPD)")
    if canon is None:
        print("[중단] stage2 산출물(canonical onset) 없음")
        return
    chans = A.e3_channels
    defs = {"A_KF_BOCPD": None, "C_level_CUSUM": onset_cusum, "D_level_BOCPD": onset_bocpd_level}
    try:
        import ruptures  # noqa
        defs["B_PELT"] = onset_pelt
    except ImportError:
        print("[안내] ruptures 미설치 -> B(PELT) 건너뜀 (pip install ruptures)")
    all_rows, cov_rows = [], []
    for dname, fn in defs.items():
        t0 = time.time()
        onsets = {}
        if fn is None:
            onsets = {k: v for k, v in canon.items() if k[0] in chans}
            n_anom = {c: len(idx.get((c, 1), [])) for c in chans}
        else:
            n_anom = {c: 0 for c in chans}
            for ch in chans:
                for key in idx.get((ch, 1), []):
                    n_anom[ch] += 1
                    try:
                        o = fn(store[key]["values"], A.min_w)
                    except Exception:
                        o = None
                    if o is not None:
                        onsets[key] = o
        obs, ratios = anomaly_effects(store, onsets, chans)
        rng = np.random.default_rng(det_seed(dname, A.seed))
        allr = np.concatenate([np.array(r) for r in ratios.values() if len(r)]) if any(len(r) for r in ratios.values()) else np.array([0.5])
        pls = []
        for ch in chans:
            pool = np.array(ratios[ch]) if len(ratios[ch]) >= 5 else allr
            pls.append(placebo_rows(store, idx, ch, pool, rng))
        pl = pd.concat(pls) if pls else pd.DataFrame()
        rows = mw_rows(obs, pl, chans, {"definition": dname})
        all_rows += rows
        for ch in chans:
            cov_rows.append(dict(definition=dname, channel=ch, n_anomaly_segments=n_anom[ch],
                                 n_valid_onset=int((obs["channel"] == ch).sum()) if len(obs) else 0))
        print(f"[{dname}] 유효 onset {len(obs)}개 ({time.time()-t0:.0f}s)")
    res = add_bonf(pd.DataFrame(all_rows), ["definition"])
    save(res, "e3_onset_definitions_mw.csv")
    cov = pd.DataFrame(cov_rows)
    cov["detection_rate"] = cov["n_valid_onset"] / cov["n_anomaly_segments"].replace(0, np.nan)
    save(cov, "e3_onset_detection_coverage.csv")

    # H3 요약
    ref = res[res["definition"] == "A_KF_BOCPD"].set_index(["channel", "feature"])
    summ = []
    for dname in res["definition"].unique():
        r = res[res["definition"] == dname].copy()
        r["diff_sign"] = np.sign(r["median_anomaly"] - r["median_placebo"])
        dd = r[r["feature"].isin(["diff_logvar", "diff2_logvar"])].dropna(subset=["median_anomaly"])
        frac_pos = float((dd["diff_sign"] > 0).mean()) if len(dd) else np.nan
        sig = float((dd["p_bonf"] < 0.05).mean()) if len(dd) else np.nan
        lv = r[(r["feature"] == "level_d_abs")].dropna(subset=["median_anomaly"])
        lv_sig = float((lv["p_bonf"] < 0.05).mean()) if len(lv) else np.nan
        summ.append(dict(definition=dname, n_deriv_tests=len(dd), frac_deriv_positive=frac_pos,
                         frac_deriv_significant=sig, frac_level_significant=lv_sig,
                         sign_reversed=bool(frac_pos < 0.5) if np.isfinite(frac_pos) else np.nan))
    s = pd.DataFrame(summ)
    print(s.to_string(index=False))
    n_rev = int(s["sign_reversed"].fillna(False).sum())
    print(f"\n[H3] 부호 반전 정의 수: {n_rev} (2종 이상이면 H3 기각)")
    save(s, "e3_h3_summary.csv")


# =============================================================================
# E4 : onset-locking + 이벤트 스터디
# =============================================================================
def logvar_profile(v, onset, offsets, w):
    d = np.diff(v)
    base = d[BURN_IN:max(onset - 1, BURN_IN)]
    if len(base) < A.min_w:
        return None
    bv = np.var(base, ddof=1)
    if bv <= 0:
        return None
    out = np.full(len(offsets), np.nan)
    for i, o in enumerate(offsets):
        s = onset - 1 + o
        e = s + w
        if s < 0 or e > len(d):
            continue
        out[i] = np.log((np.var(d[s:e], ddof=1) + 1e-12 * bv) / bv)
    return out


def agg_profile(M, rng, n_boot=300):
    M = np.asarray(M)
    med = np.nanmedian(M, axis=0)
    bs = []
    for _ in range(n_boot):
        bs.append(np.nanmedian(M[rng.integers(0, len(M), len(M))], axis=0))
    bs = np.array(bs)
    return med, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)


def step_e4(store, idx, canon):
    sec("E4: 이상 세그먼트 내부 위약 (onset-locking) + 이벤트 스터디")
    if canon is None:
        print("[중단] canonical onset 없음")
        return
    chans = sorted({k[0] for k in canon})
    rng = np.random.default_rng(A.seed)
    obs_df, ratios = anomaly_effects(store, canon, chans)
    seg_rows = []
    for _, r in obs_df.iterrows():
        ch, sid, onset = r["channel"], int(r["segment"]), int(r["onset"])
        v = store[(ch, sid)]["values"]
        lo, hi = valid_pivot_range(len(v))
        piv = rng.integers(lo, hi + 1, size=A.e4_k)
        null = pd.DataFrame([compute_effects(v, int(p)) for p in piv])
        for f in FEATS3:
            nv = null[f].dropna().values
            if not np.isfinite(r[f]) or len(nv) < 5:
                continue
            seg_rows.append(dict(channel=ch, segment=sid, feature=f, observed=r[f],
                                 null_median=float(np.median(nv)),
                                 p_emp=float((1 + (nv >= r[f]).sum()) / (len(nv) + 1)),
                                 percentile=float((nv < r[f]).mean())))
    sd = pd.DataFrame(seg_rows)
    save(sd, "e4_within_segment_null.csv")

    summ = []
    for ch in chans:
        pl = placebo_rows(store, idx, ch, np.array(ratios[ch]) if ratios[ch] else np.array([0.5]), rng)
        for f in FEATS3:
            g = sd[(sd["channel"] == ch) & (sd["feature"] == f)]
            if len(g) < 5:
                continue
            try:
                wp = stats.wilcoxon(g["observed"] - g["null_median"], alternative="greater").pvalue
            except Exception:
                wp = np.nan
            summ.append(dict(channel=ch, feature=f, n=len(g), frac_p_lt_05=float((g["p_emp"] < 0.05).mean()),
                             median_percentile=float(g["percentile"].median()), wilcoxon_p=wp,
                             median_observed=float(g["observed"].median()),
                             median_internal_null=float(g["null_median"].median()),
                             median_normal_placebo=float(pl[f].median()) if len(pl) else np.nan))
    s = pd.DataFrame(summ)
    print(s.to_string(index=False))
    save(s, "e4_summary.csv")
    print("\n해석: median_internal_null 이 median_normal_placebo 보다 훨씬 크고 median_observed 와 비슷하면\n"
          "      'onset 국소 급증'이 아니라 '이상 세그먼트의 전반적 거칠기'로 재서술해야 함.")

    # 이벤트 스터디
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    offs = list(range(-A.e4_pre, A.e4_post + 1))
    prof_rows = []
    for ch in chans:
        Ma, Mr, Mp = [], [], []
        for _, r in obs_df[obs_df["channel"] == ch].iterrows():
            v = store[(ch, int(r["segment"]))]["values"]
            pa = logvar_profile(v, int(r["onset"]), offs, A.e4_w)
            if pa is not None:
                Ma.append(pa)
            lo, hi = valid_pivot_range(len(v))
            ps = [logvar_profile(v, int(p), offs, A.e4_w) for p in rng.integers(lo, hi + 1, size=5)]
            ps = [p for p in ps if p is not None]
            if ps:
                Mr.append(np.nanmean(ps, axis=0))
        keys = list(idx.get((ch, 0), []))
        keys = [keys[i] for i in rng.choice(len(keys), min(len(keys), A.max_normal), replace=False)] if keys else []
        pool = np.array(ratios[ch]) if ratios[ch] else np.array([0.5])
        for key in keys:
            v = store[key]["values"]
            lo, hi = valid_pivot_range(len(v))
            if hi < lo:
                continue
            piv = int(np.clip(int(round(rng.choice(pool) * len(v))), lo, hi))
            p = logvar_profile(v, piv, offs, A.e4_w)
            if p is not None:
                Mp.append(p)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, M, col in [("anomaly @ canonical onset", Ma, "tab:red"),
                             ("anomaly @ random pivots (internal)", Mr, "tab:orange"),
                             ("normal placebo", Mp, "tab:blue")]:
            if len(M) < 3:
                continue
            med, lo_, hi_ = agg_profile(M, rng)
            ax.plot(offs, med, color=col, label=f"{name} (n={len(M)})")
            ax.fill_between(offs, lo_, hi_, color=col, alpha=0.2)
            for o, a_, b_, c_ in zip(offs, med, lo_, hi_):
                prof_rows.append(dict(channel=ch, curve=name, offset=o, median=a_, lo=b_, hi=c_, n=len(M)))
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("offset from pivot (ticks)")
        ax.set_ylabel(f"log(var diff[w={A.e4_w}] / pre-onset var)")
        ax.set_title(f"{ch} — event-study profile")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"e4_event_study_{ch}.png", dpi=130)
        plt.close(fig)
    save(pd.DataFrame(prof_rows), "e4_event_study_profiles.csv")


# =============================================================================
# E8 : 하위 라벨 / sampling 층화
# =============================================================================
def step_e8(store, idx, canon):
    sec("E8: 정상 하위라벨(a2/a3/a4) 및 sampling 층화 위약 대조")
    if canon is None:
        print("[중단] canonical onset 없음")
        return
    chans = A.e3_channels
    obs, ratios = anomaly_effects(store, canon, chans)
    rng = np.random.default_rng(A.seed)
    rows = []
    for ch in chans:
        pool = np.array(ratios[ch]) if ratios[ch] else np.array([0.5])
        labels = sorted({store[k]["label"] for k in idx.get((ch, 0), [])})
        samps = sorted({store[k]["sampling"] for k in idx.get((ch, 0), [])})
        pl_all = placebo_rows(store, idx, ch, pool, rng)
        rows += mw_rows(obs, pl_all, [ch], {"stratum_type": "ALL", "stratum": "ALL"})
        for lb in labels:
            pl = placebo_rows(store, idx, ch, pool, rng, filt=lambda r, lb=lb: r["label"] == lb)
            rows += mw_rows(obs, pl, [ch], {"stratum_type": "normal_label", "stratum": lb})
        for sp in samps:
            o_s = obs[obs["sampling"] == sp]
            pl = placebo_rows(store, idx, ch, pool, rng, filt=lambda r, sp=sp: r["sampling"] == sp)
            rows += mw_rows(o_s, pl, [ch], {"stratum_type": "sampling", "stratum": str(sp)})
    res = add_bonf(pd.DataFrame(rows))
    save(res, "e8_stratified_placebo.csv")
    print(res[res["feature"] != "level_d_abs"][["channel", "stratum_type", "stratum", "feature", "n_anomaly",
                                                "n_placebo", "median_anomaly", "median_placebo", "p_bonf"]]
          .to_string(index=False))
    print("\n해석: 특정 하위 유형에서만 효과가 나오면 운영모드 교란을 의심. 모든 층에서 같은 방향이면 안심.")


# =============================================================================
# E5 : 모델군 재설계
# =============================================================================
E5_SETS = {"level": ["level_d_abs"], "diff": ["diff_logvar"], "diff2": ["diff2_logvar"],
           "deriv": ["diff_logvar", "diff2_logvar"], "all": FEATS3}
E5_CONTRASTS = [("diff_minus_level", "diff", "level"), ("diff2_minus_level", "diff2", "level"),
                ("deriv_minus_level", "deriv", "level"), ("all_minus_deriv(level_added)", "all", "deriv"),
                ("all_minus_level(deriv_added)", "all", "level")]


def build_window(v, piv, half, mode):
    n = len(v)
    lo, hi = piv - half, piv + half
    if lo - 2 < 0 or hi > n:
        return None
    if mode == "global_robust":
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        s = mad if mad > 0 else (v.std() if v.std() > 0 else 1.0)
        v = (v - med) / s
    d1 = np.diff(v)
    d2 = np.diff(d1)
    chs = [v[lo:hi], d1[lo - 1:hi - 1], d2[lo - 2:hi - 2]]
    if mode == "window_z":
        chs = [(c - c.mean()) / (c.std() + 1e-8) for c in chs]
    return np.stack(chs).astype(np.float32)


def assign_folds(groups, labels, k, rng):
    fold = {}
    for lab in (0, 1):
        gs = sorted({g for g, l in zip(groups, labels) if l == lab})
        for i, g in enumerate(rng.permutation(gs)):
            fold[g] = i % k
    return np.array([fold[g] for g in groups])


def build_e5_dataset(store, idx, canon):
    rng = np.random.default_rng(A.seed)
    chans = A.e3_channels
    obs, ratios = anomaly_effects(store, canon, chans)
    recs = []
    for _, r in obs.iterrows():
        recs.append(dict(channel=r["channel"], segment=int(r["segment"]), label=1, pivot=int(r["onset"]),
                         **{f: r[f] for f in FEATS3}))
    for ch in chans:
        pool = np.array(ratios[ch]) if ratios[ch] else np.array([0.5])
        pl = placebo_rows(store, idx, ch, pool, rng, k=A.k_placebo)
        for _, r in pl.iterrows():
            recs.append(dict(channel=ch, segment=int(r["segment"]), label=0, pivot=int(r["pivot"]),
                             **{f: r[f] for f in FEATS3}))
    df = pd.DataFrame(recs)
    df["group"] = df["channel"] + "_" + df["segment"].astype(str)
    df = df.dropna(subset=FEATS3).reset_index(drop=True)
    return df


def tab_zoo():
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    s = A.seed
    z = [("LogReg_L2", "Linear", lambda: LogisticRegression(class_weight="balanced", max_iter=3000)),
         ("LogReg_L1", "Linear(sparse)", lambda: LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000)),
         ("GaussianNB", "Probabilistic", lambda: GaussianNB()),
         ("kNN", "Instance-based", lambda: KNeighborsClassifier(n_neighbors=15)),
         ("SVM_RBF", "Kernel", lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=s)),
         ("RandomForest", "Tree(bagging)", lambda: RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=s, n_jobs=-1)),
         ("ExtraTrees", "Tree(extra)", lambda: ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=s, n_jobs=-1)),
         ("GradBoost", "Tree(boosting)", lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=s))]
    try:
        from xgboost import XGBClassifier
        z.append(("XGBoost", "Tree(boosting)", lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=s, verbosity=0)))
    except ImportError:
        print("[안내] xgboost 없음 -> 건너뜀")
    try:
        from lightgbm import LGBMClassifier
        z.append(("LightGBM", "Tree(boosting)", lambda: LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=s, verbosity=-1)))
    except ImportError:
        print("[안내] lightgbm 없음 -> 건너뜀")
    if A.e5_include_mlp:
        from sklearn.neural_network import MLPClassifier
        z.append(("MLP_shallow", "NeuralNet(shallow)", lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=s)))
    return z


def define_torch_models():
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(s, c_in, h=32):
            super().__init__()
            s.net = nn.Sequential(nn.Conv1d(c_in, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            s.head = nn.Linear(h, 1)

        def forward(s, x):
            return s.head(s.net(x).squeeze(-1)).squeeze(-1)

    class TCNBlock(nn.Module):
        def __init__(s, ci, co, dil):
            super().__init__()
            s.pad = 4 * dil
            s.conv = nn.Conv1d(ci, co, 5, padding=s.pad, dilation=dil)
            s.res = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

        def forward(s, x):
            o = s.conv(x)[:, :, :-s.pad]
            return torch.relu(o + s.res(x))

    class TCN(nn.Module):
        def __init__(s, c_in, h=32):
            super().__init__()
            s.b = nn.Sequential(TCNBlock(c_in, h, 1), TCNBlock(h, h, 2), TCNBlock(h, h, 4))
            s.head = nn.Linear(h, 1)

        def forward(s, x):
            return s.head(s.b(x).mean(-1)).squeeze(-1)

    class BiRNN(nn.Module):
        def __init__(s, c_in, h=32, kind="lstm"):
            super().__init__()
            s.r = (nn.LSTM if kind == "lstm" else nn.GRU)(c_in, h, batch_first=True, bidirectional=True)
            s.head = nn.Linear(2 * h, 1)

        def forward(s, x):
            o, _ = s.r(x.permute(0, 2, 1))
            return s.head(o.mean(1)).squeeze(-1)

    class BiLSTM(BiRNN):
        def __init__(s, c_in):
            super().__init__(c_in, kind="lstm")

    class BiGRU(BiRNN):
        def __init__(s, c_in):
            super().__init__(c_in, kind="gru")

    class TinyTransformer(nn.Module):
        def __init__(s, c_in, d=32, nh=4, nl=2, max_len=200):
            super().__init__()
            s.proj = nn.Linear(c_in, d)
            pe = torch.zeros(max_len, d)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            s.register_buffer("pe", pe.unsqueeze(0))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nh, 64, batch_first=True), nl)
            s.head = nn.Linear(d, 1)

        def forward(s, x):
            h = s.proj(x.permute(0, 2, 1)) + s.pe[:, :x.size(2)]
            return s.head(s.enc(h).mean(1)).squeeze(-1)

    class LightMamba(nn.Module):
        """순수 PyTorch 선택적 SSM 근사(공식 mamba_ssm 아님)."""
        def __init__(s, c_in, d=32, ds=16):
            super().__init__()
            s.pi = nn.Linear(c_in, d)
            s.A_log = nn.Parameter(torch.randn(d, ds) * 0.1)
            s.B, s.C = nn.Linear(d, ds), nn.Linear(d, ds)
            s.dl = nn.Linear(d, d)
            s.head = nn.Linear(d, 1)
            s.ds = ds

        def forward(s, x):
            u = s.pi(x.permute(0, 2, 1))
            Bsz, T, D = u.shape
            dl = torch.nn.functional.softplus(s.dl(u))
            Am = -torch.exp(s.A_log)
            Bc, Cc = s.B(u), s.C(u)
            h = torch.zeros(Bsz, D, s.ds, device=u.device)
            ys = []
            for t in range(T):
                dA = torch.exp(dl[:, t].unsqueeze(-1) * Am.unsqueeze(0))
                dBu = dl[:, t].unsqueeze(-1) * Bc[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
                h = dA * h + dBu
                ys.append((h * Cc[:, t].unsqueeze(1)).sum(-1))
            return s.head(torch.stack(ys, 1).mean(1)).squeeze(-1)

    return {"CNN1D": ("Convolutional", CNN1D), "TCN": ("Convolutional(dilated)", TCN),
            "BiLSTM": ("Recurrent", BiLSTM), "BiGRU": ("Recurrent", BiGRU),
            "TinyTransformer": ("Attention", TinyTransformer), "LightMamba": ("State-space(SSM)", LightMamba)}


def train_oof_deep(Cls, X, y, groups, folds, epochs, seed):
    import torch
    import torch.nn as nn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        te = np.where(folds == f)[0]
        tr_all = np.where(folds != f)[0]
        rng = np.random.default_rng(seed + f)
        ug = np.unique(groups[tr_all])
        vg = set(rng.permutation(ug)[:max(1, int(0.15 * len(ug)))])
        is_val = np.array([g in vg for g in groups[tr_all]])
        tr, va = tr_all[~is_val], tr_all[is_val]
        if len(te) < 5 or len(tr) < 20 or y[tr].sum() < 3:
            continue
        torch.manual_seed(seed + f)
        model = Cls(X.shape[1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pw = torch.tensor([(1 - y[tr]).sum() / max(y[tr].sum(), 1)], dtype=torch.float32, device=dev)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt = torch.tensor(X[tr], device=dev)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
        Xv = torch.tensor(X[va], device=dev)
        yv = torch.tensor(y[va], dtype=torch.float32, device=dev)
        best, bs, pat = np.inf, None, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(tr), device=dev)
            for i in range(0, len(tr), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                crit(model(Xt[b]), yt[b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(Xv), yv).item() if len(va) else 0.0
            if vl < best - 1e-4:
                best, pat = vl, 0
                bs = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= 6:
                    break
        if bs:
            model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(torch.tensor(X[te], device=dev))).cpu().numpy()
    return oof


def step_e5(store, idx, canon):
    sec("E5: 모델군 재설계 (그룹 제거 ΔAUC, 단일 표현 학습, 정규화 2종, cluster bootstrap)")
    if canon is None:
        print("[중단] canonical onset 없음")
        return
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    ds = build_e5_dataset(store, idx, canon)
    y = ds["label"].to_numpy(int)
    groups = ds["group"].to_numpy()
    print(f"데이터셋: {len(ds)}행 (양성 {int(y.sum())}, 음성 {int((1-y).sum())}), 그룹 {len(set(groups))}개")
    save(ds[["channel", "segment", "label", "pivot"] + FEATS3], "e5_dataset.csv")
    rng = np.random.default_rng(A.seed)
    folds = assign_folds(groups, y, 5, rng)

    results = []   # model,family,rep,set,auc,prauc
    preds = {}     # (rep, model) -> {set: oof}
    # ---- 표 모델
    for mname, fam, mk in tab_zoo():
        print(f"[tab] {mname}", flush=True)
        P = {}
        for sname, cols in E5_SETS.items():
            X = ds[cols].to_numpy(float)
            oof = np.full(len(y), np.nan)
            for f in range(5):
                tr, te = folds != f, folds == f
                sc = StandardScaler().fit(X[tr])
                try:
                    m = mk()
                    m.fit(sc.transform(X[tr]), y[tr])
                    oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
                except Exception as e:
                    print(f"   fold {f} 실패: {e}")
            ok = ~np.isnan(oof)
            P[sname] = np.where(ok, oof, 0.5)
            results.append(dict(model=mname, family=fam, rep="tabular", set=sname,
                                auc=fast_auc(y[ok], oof[ok]), pr_auc=float(average_precision_score(y[ok], oof[ok])),
                                n=int(ok.sum())))
        preds[("tabular", mname)] = (P, fam)

    # ---- 시퀀스 모델
    try:
        import torch  # noqa
        have_torch = True
    except ImportError:
        have_torch = False
        print("[경고] torch 없음 -> 시퀀스 모델 건너뜀")
    if have_torch:
        zoo = define_torch_models()
        wanted = [m for m in A.e5_seq_models.split(",") if m in zoo]
        for mode in ["window_z", "global_robust"]:
            Xs, keep = [], []
            for i, r in ds.iterrows():
                w = build_window(store[(r["channel"], int(r["segment"]))]["values"], int(r["pivot"]), A.seq_half, mode)
                if w is not None:
                    Xs.append(w)
                    keep.append(i)
            Xs = np.stack(Xs)
            if mode == "global_robust":
                m_, s_ = Xs.mean(axis=(0, 2), keepdims=True), Xs.std(axis=(0, 2), keepdims=True) + 1e-8
                Xs = np.clip((Xs - m_) / s_, -10, 10).astype(np.float32)
            keep = np.array(keep)
            ys, gs, fs = y[keep], groups[keep], folds[keep]
            chsel = {"level": [0], "diff": [1], "diff2": [2], "deriv": [1, 2], "all": [0, 1, 2]}
            print(f"[seq:{mode}] 윈도우 {len(Xs)}개")
            for mname in wanted:
                fam, Cls = zoo[mname]
                P = {}
                for sname, ci in chsel.items():
                    t0 = time.time()
                    oof = train_oof_deep(Cls, np.ascontiguousarray(Xs[:, ci, :]), ys, gs, fs, A.e5_epochs, A.seed)
                    ok = ~np.isnan(oof)
                    P[sname] = np.where(ok, oof, 0.5)
                    results.append(dict(model=mname, family=fam, rep=f"seq_{mode}", set=sname,
                                        auc=fast_auc(ys[ok], oof[ok]),
                                        pr_auc=float(average_precision_score(ys[ok], oof[ok])) if ok.sum() > 10 else np.nan,
                                        n=int(ok.sum())))
                    print(f"   {mname:16s} {sname:6s} AUC={results[-1]['auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                preds[(f"seq_{mode}", mname)] = (P, fam, ys, gs)
    res = pd.DataFrame(results)
    save(res, "e5_auc_by_model_set.csv")

    # ---- cluster bootstrap 대비
    rows = []
    for key, val in preds.items():
        rep, mname = key
        if rep == "tabular":
            P, fam = val
            yy, gg = y, groups
        else:
            P, fam, yy, gg = val
        r_ = np.random.default_rng(det_seed(rep, mname, A.seed))
        cb = cluster_boot_contrasts(yy, gg, P, E5_CONTRASTS, A.e5_boot, r_)
        for c, (p, lo, hi) in cb.items():
            rows.append(dict(rep=rep, model=mname, family=fam, contrast=c, delta_auc=p, ci_lo=lo, ci_hi=hi))
    cdf = pd.DataFrame(rows)
    save(cdf, "e5_contrasts_cluster_boot.csv")

    # ---- H4 요약
    from scipy.stats import binomtest
    piv = res.pivot_table(index=["rep", "model", "family"], columns="set", values="auc").reset_index()
    piv["strict_pass"] = (piv["diff"] >= piv["level"]) & (piv["diff2"] >= piv["level"])
    piv["lenient_pass"] = (piv["diff"] >= piv["level"]) | (piv["diff2"] >= piv["level"])
    piv["deriv_ge_level"] = piv["deriv"] >= piv["level"]
    save(piv, "e5_model_level_table.csv")
    h4 = []
    for rep, g in piv.groupby("rep"):
        variants = [("all_models", g)]
        if A.e5_include_mlp:
            variants = [("all_models_incl_MLP", g), ("excl_MLP", g[g["model"] != "MLP_shallow"])]
        for vname, gg in variants:
            n = len(gg)
            if n == 0:
                continue
            for crit in ["strict_pass", "lenient_pass", "deriv_ge_level"]:
                k = int(gg[crit].sum())
                ci = binomtest(k, n).proportion_ci(method="exact")
                h4.append(dict(rep=rep, variant=vname, criterion=crit, n_models=n, n_pass=k, frac=k / n,
                               ci_lo=ci.low, ci_hi=ci.high, h4_pass_80pct=bool(crit == "strict_pass" and k / n >= 0.8)))
    h4 = pd.DataFrame(h4)
    print(h4.to_string(index=False))
    save(h4, "e5_h4_summary.csv")
    print("\n주의: 모델별 예외(예: LightMamba)는 e5_model_level_table.csv 에서 개별 확인하고 본문에서 논의할 것.")


# =============================================================================
# E6 : 외부 데이터
# =============================================================================
def step_ext_build():
    sec("EXT-BUILD: 외부 시계열 -> 세그먼트 long CSV")
    s = pd.read_csv(A.ext_series)
    need = {"channel", "value", "label"}
    assert need <= set(s.columns), f"필요 컬럼 {need}"
    rng = np.random.default_rng(A.seed)
    out, sid = [], 0
    for ch, g in s.groupby("channel", sort=False):
        v = g["value"].to_numpy(float)
        lab = g["label"].to_numpy(int)
        n = len(v)
        d = np.diff(np.r_[0, lab, 0])
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        lens, blocked = [], np.zeros(n, bool)
        for a, b in zip(st, en):
            lo, hi = max(0, a - A.ext_ctx), min(n, b + A.ext_ctx)
            blocked[max(0, a - 2 * A.ext_ctx):min(n, b + 2 * A.ext_ctx)] = True
            if hi - lo < A.ext_minlen:
                continue
            sid += 1
            lens.append(hi - lo)
            out.append(pd.DataFrame(dict(channel=ch, segment=sid, timestamp=np.arange(hi - lo), value=v[lo:hi], anomaly=1, sampling=1.0)))
        if not lens:
            continue
        used = blocked.copy()
        got, target = 0, A.ext_normal_ratio * len(lens)
        for _ in range(int(target) * 30):
            if got >= target:
                break
            L = int(rng.choice(lens))
            if L >= n:
                continue
            a = int(rng.integers(0, n - L + 1))
            if used[a:a + L].any():
                continue
            used[a:a + L] = True
            sid += 1
            got += 1
            out.append(pd.DataFrame(dict(channel=ch, segment=sid, timestamp=np.arange(L), value=v[a:a + L], anomaly=0, sampling=1.0)))
        print(f"[{ch}] 이상 세그먼트 {len(lens)}개, 정상 {got}개")
    res = pd.concat(out)
    p = Path(A.ext_out)
    res.to_csv(p, index=False)
    print(f"[저장] {p}  ({len(res)} rows)")


def step_ext_run():
    sec("EXT-RUN: 외부 세그먼트에 E1+E2 동결 코드 적용 (H5)")
    seg = pd.read_csv(A.ext_segments)
    store = build_store(seg)
    table = build_table(store)
    chans, g = select_channels(table, A.min_pos, A.min_neg)
    print(g)
    save(table, "ext_segment_features.csv")
    step_e1(store, table, None, prefix="ext_")
    step_e2(store, table, prefix="ext_")


# =============================================================================
# main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prep", "e1", "e2", "e3", "e4", "e5", "e8", "ext-build", "ext-run", "all"])
    p.add_argument("--base-dir", default=".")
    p.add_argument("--segments", default=None)
    p.add_argument("--stage-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-e5", action="store_true")
    p.add_argument("--min-w", type=int, default=5)
    p.add_argument("--min-pos", type=int, default=20, help="E1/E2 채널 선정: 최소 이상 세그먼트")
    p.add_argument("--min-neg", type=int, default=20)
    p.add_argument("--min-class", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--e2-k", type=int, default=50)
    p.add_argument("--e2-null", choices=["phase", "block"], default="phase")
    p.add_argument("--e3-channels", default="CADC0872,CADC0873,CADC0874,CADC0888,CADC0894")
    p.add_argument("--pelt-pen", type=float, default=3.0)
    p.add_argument("--k-placebo", type=int, default=5)
    p.add_argument("--max-normal", type=int, default=300)
    p.add_argument("--e4-k", type=int, default=20)
    p.add_argument("--e4-pre", type=int, default=15)
    p.add_argument("--e4-post", type=int, default=40)
    p.add_argument("--e4-w", type=int, default=5)
    p.add_argument("--e5-boot", type=int, default=500)
    p.add_argument("--e5-epochs", type=int, default=50)
    p.add_argument("--e5-seq-models", default="CNN1D,TCN,BiLSTM,BiGRU,TinyTransformer,LightMamba")
    p.add_argument("--e5-include-mlp", action="store_true")
    p.add_argument("--seq-half", type=int, default=15)
    p.add_argument("--ext-series", default=None)
    p.add_argument("--ext-out", default="ext_segments.csv")
    p.add_argument("--ext-segments", default=None)
    p.add_argument("--ext-ctx", type=int, default=50)
    p.add_argument("--ext-minlen", type=int, default=40)
    p.add_argument("--ext-normal-ratio", type=float, default=5.0)
    return p.parse_args()


def main():
    global A, OUT
    A = parse_args()
    A.e3_channels = [c.strip() for c in A.e3_channels.split(",") if c.strip()]
    if A.quick:
        A.n_boot, A.n_repeats, A.e2_k, A.e4_k = 200, 1, 15, 8
        A.e5_boot, A.e5_epochs, A.k_placebo, A.max_normal = 100, 6, 2, 60
    base = Path(A.base_dir)
    seg_path = Path(A.segments) if A.segments else base / "segments.csv"
    stage_dir = Path(A.stage_dir) if A.stage_dir else base / "causal_pipeline_outputs"
    OUT = Path(A.out_dir) if A.out_dir else base / "paper2_outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"step={A.step}  base={base}  out={OUT}  quick={A.quick}")

    if A.step == "ext-build":
        return step_ext_build()
    if A.step == "ext-run":
        return step_ext_run()

    sec("데이터 적재")
    seg = load_segments(seg_path)
    store = build_store(seg)
    idx = make_index(store)
    table = build_table(store)
    canon = load_canonical(stage_dir)
    print(f"세그먼트 {len(store)}개, 피처표 {table.shape}, canonical onset "
          f"{'없음' if canon is None else len(canon)}개")

    steps = {"prep": lambda: step_prep(store, table, canon),
             "e1": lambda: step_e1(store, table, canon),
             "e2": lambda: step_e2(store, table),
             "e3": lambda: step_e3(store, idx, canon),
             "e4": lambda: step_e4(store, idx, canon),
             "e5": lambda: step_e5(store, idx, canon),
             "e8": lambda: step_e8(store, idx, canon)}
    order = ["prep", "e1", "e2", "e4", "e8", "e3", "e5"] if A.step == "all" else [A.step]
    if A.step == "all" and A.skip_e5:
        order.remove("e5")
    for s in order:
        t0 = time.time()
        steps[s]()
        print(f"--- {s} 완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
