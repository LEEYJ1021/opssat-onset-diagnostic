"""rev_common.py — 리비전 공통 유틸 (numpy/pandas/scipy만 사용)"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

import os
BASE = Path(os.environ.get("OPSSAT_BASE_DIR", "."))
STAGE = BASE / "causal_pipeline_outputs"
OUT = Path(os.environ.get("OPSSAT_OUT_DIR", BASE / "revision_outputs")); OUT.mkdir(parents=True, exist_ok=True)
CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]
BURN_IN, MIN_W = 5, 5
REPS = ["level", "d1", "d2"]
# lv : ln(var_post/var_pre)               (분산비, 식 (2))
# ms : |mean_post-mean_pre| / pooled SD   (평균이동, 식 (1))
# msg: |mean_post-mean_pre| / SD_pre      (pre-SD 표준화: post 분산 증가에 둔감)
STAT_COLS = [f"{r}_{s}" for r in REPS for s in ("lv", "ms", "msg")]


def load_store():
    seg = pd.read_csv(BASE / "segments.csv",
                      usecols=["channel", "segment", "timestamp", "value", "sampling", "anomaly"])
    seg = seg[seg.channel.isin(CHANNELS)]
    seg = seg.sort_values(["channel", "segment", "timestamp"], kind="stable")
    store = {}
    for (ch, sid), g in seg.groupby(["channel", "segment"], sort=False):
        store[(ch, int(sid))] = dict(v=g["value"].to_numpy(float),
                                     anomaly=int(g["anomaly"].iloc[0]),
                                     sampling=float(g["sampling"].iloc[0]))
    return store


def load_onsets(normal=False):
    """(onset 중앙값 dict, high/medium draw 표, 평가된 세그먼트 집합)"""
    sfx = "_NORMAL" if normal else ""
    draws = pd.read_csv(STAGE / f"stage2_mc_draws_raw{sfx}.csv",
                        usecols=["channel", "segment", "draw", "onset_idx_sampled"])
    corr = pd.read_csv(STAGE / f"stage2_mc_segment_summary_corrected{sfx}.csv",
                       usecols=["channel", "segment", "reliability_tier"])
    evaluated = {(c, int(s)) for c, s in zip(corr.channel, corr.segment)}
    rel = corr[corr.reliability_tier.isin(["high", "medium"])][["channel", "segment"]]
    d = draws.merge(rel, on=["channel", "segment"])
    med = d.groupby(["channel", "segment"])["onset_idx_sampled"].median().round().astype(int)
    return {(c, int(s)): int(v) for (c, s), v in med.items()}, d, evaluated


# ---------------- 통계량 ----------------
def _lv(pre, post):
    if len(pre) < MIN_W or len(post) < MIN_W:
        return np.nan
    vp, vq = np.var(pre, ddof=1), np.var(post, ddof=1)
    if not (vp > 0 and vq > 0 and np.isfinite(vp) and np.isfinite(vq)):
        return np.nan
    return float(np.log(vq / vp))


def _ms(pre, post):
    if len(pre) < 2 or len(post) < 2:
        return np.nan, np.nan
    vp, vq = np.var(pre, ddof=1), np.var(post, ddof=1)
    dm = abs(post.mean() - pre.mean())
    sd = np.sqrt((vp + vq) / 2)
    pooled = dm / sd if (sd > 0 and np.isfinite(sd)) else np.nan
    glass = dm / np.sqrt(vp) if vp > 0 else np.nan
    return float(pooled), float(glass)


def _windows(v, piv, h=None):
    n = len(v)
    d1 = np.diff(v); d2 = np.diff(d1)
    if h is None:   # 논문 설정: burn-in 5 이후 ~ pivot / pivot ~ 끝
        return {"level": (v[BURN_IN:piv], v[piv:]),
                "d1": (d1[BURN_IN - 1:piv - 1], d1[piv - 1:]),
                "d2": (d2[BURN_IN - 2:piv - 2], d2[piv - 2:])}
    if piv - h < 0 or piv + h > n:   # 고정 윈도우 ±h
        return None
    return {"level": (v[piv - h:piv], v[piv:piv + h]),
            "d1": (d1[piv - h:piv - 1], d1[piv - 1:piv + h - 1]),
            "d2": (d2[piv - h:piv - 2], d2[piv - 2:piv + h - 2])}


def stats_at(v, piv, h=None):
    out = {c: np.nan for c in STAT_COLS}
    n = len(v)
    if piv < BURN_IN + MIN_W or piv > n - MIN_W:
        return out
    W = _windows(v, piv, h)
    if W is None:
        return out
    for rep, (pre, post) in W.items():
        out[f"{rep}_lv"] = _lv(pre, post)
        out[f"{rep}_ms"], out[f"{rep}_msg"] = _ms(pre, post)
    return out


# ---------------- 표 생성 ----------------
def onset_table(store, onsets, anomaly, h=None, channel=None):
    rows = []
    for (ch, sid), on in onsets.items():
        r = store.get((ch, sid))
        if r is None or r["anomaly"] != anomaly or (channel and ch != channel):
            continue
        v = r["v"]; n = len(v)
        if not (BURN_IN + MIN_W <= on <= n - MIN_W):
            continue
        rows.append(dict(channel=ch, segment=sid, pivot=on, n=n, ratio=on / n, **stats_at(v, on, h)))
    return pd.DataFrame(rows)


def placebo_table(store, ch, pool, k, seed, h=None, cap=300, allowed=None):
    rng = np.random.default_rng(seed)
    keys = sorted(kk for kk, r in store.items()
                  if kk[0] == ch and r["anomaly"] == 0 and (allowed is None or kk in allowed))
    if len(keys) > cap:
        keys = [keys[i] for i in rng.choice(len(keys), cap, replace=False)]
    rows = []
    for key in keys:
        v = store[key]["v"]; n = len(v)
        lo = BURN_IN + MIN_W if h is None else max(BURN_IN + MIN_W, h)
        hi = n - MIN_W if h is None else n - max(MIN_W, h)
        if hi < lo:
            continue
        for _ in range(k):
            piv = int(np.clip(int(round(rng.choice(pool) * n)), lo, hi))
            rows.append(dict(channel=ch, segment=key[1], pivot=piv, n=n, ratio=piv / n,
                             **stats_at(v, piv, h)))
    return pd.DataFrame(rows)


# ---------------- 효과크기 / cluster bootstrap ----------------
def auc_A(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    r = stats.rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def _groups(ids, x):
    u, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(x[order], np.cumsum(np.bincount(inv))[:-1])


def contrast_row(a_df, b_df, col, n_boot=1000, seed=0):
    """A=P(a>b) + 세그먼트 단위 cluster bootstrap 95% CI + 양측 MW p(기술용)."""
    res = dict(n_a=0, n_b=0, A=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_two_sided=np.nan)
    if not len(a_df) or not len(b_df) or col not in a_df or col not in b_df:
        return res
    xa = a_df[col].to_numpy(float); xb = b_df[col].to_numpy(float)
    ia = a_df["segment"].to_numpy(); ib = b_df["segment"].to_numpy()
    ma, mb = np.isfinite(xa), np.isfinite(xb)
    xa, ia, xb, ib = xa[ma], ia[ma], xb[mb], ib[mb]
    res.update(n_a=len(xa), n_b=len(xb))
    if len(xa) < 3 or len(xb) < 3:
        return res
    res["A"] = auc_A(xa, xb)
    res["p_two_sided"] = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
    ga, gb = _groups(ia, xa), _groups(ib, xb)
    rng = np.random.default_rng(seed); est = []
    for _ in range(n_boot):
        a = np.concatenate([ga[i] for i in rng.integers(0, len(ga), len(ga))])
        b = np.concatenate([gb[i] for i in rng.integers(0, len(gb), len(gb))])
        est.append(auc_A(a, b))
    est = np.array(est); est = est[np.isfinite(est)]
    if len(est):
        res["ci_lo"], res["ci_hi"] = [float(q) for q in np.percentile(est, [2.5, 97.5])]
    return res