import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev4_estimator_matrix.py
추정기 {KF-BOCPD(분산 민감), PELT-l2(평균 민감), level-CUSUM, level-BOCPD} 를
이상/정상 구간 양쪽에 똑같이 적용하여
  (1) 발화율  (2) true vs pseudo-onset  (3) true vs random  (4) pseudo vs random
을 9개 통계량으로 비교. 위치 층화(0.45-0.75)와 공통 길이 범위 민감도 포함.
E3(paper2_pipeline.py)의 추정기 구현을 그대로 복사함."""
import argparse, time
from scipy import stats as sps
from rev_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=500)
ap.add_argument("--estimators", default="PELT3,PELT5,CUSUM,BOCPD_level")
ap.add_argument("--recompute", action="store_true")
args = ap.parse_args()

# ---------------- E3 추정기 (원본 복사) ----------------
def baseline_z(v, minw):
    b = max(minw, len(v) // 5)
    mu, sd = v[:b].mean(), v[:b].std()
    if sd <= 0:
        sd = v.std() if v.std() > 0 else 1.0
    return (v - mu) / sd

def onset_cusum(v, minw=MIN_W, k=0.5, h=5.0):
    z = baseline_z(v, minw); n = len(v); b = max(minw, n // 5)
    sp = sn = 0.0; sp0 = sn0 = b
    for t in range(b, n):
        if sp == 0: sp0 = t
        if sn == 0: sn0 = t
        sp = max(0.0, sp + z[t] - k); sn = max(0.0, sn - z[t] - k)
        if sp > h: return int(sp0)
        if sn > h: return int(sn0)
    return None

def bocpd_mrl(x, hazard=1 / 250., mu0=0., k0=1., a0=1., b0=1., kmax=40.):
    n = len(x); probs = np.array([1.0])
    mu, kap, al, be = np.array([mu0]), np.array([k0]), np.array([a0]), np.array([b0])
    mrl = np.zeros(n, int)
    for t in range(n):
        xt = x[t]
        pp = np.nan_to_num(sps.t.pdf(xt, df=2 * al, loc=mu, scale=np.sqrt(be * (kap + 1) / (al * kap))),
                           nan=1e-12, posinf=1e-12)
        new = np.concatenate([[np.sum(probs * pp * hazard)], probs * pp * (1 - hazard)])
        tot = new.sum()
        if tot <= 0 or not np.isfinite(tot):
            new = np.zeros_like(new); new[0] = 1.0
        else:
            new = new / tot
        probs = new; mrl[t] = int(np.argmax(probs))
        nmu = np.concatenate([[mu0], (kap * mu + xt) / (kap + 1)])
        nkap = np.concatenate([[k0], kap + 1]); nal = np.concatenate([[a0], al + 0.5])
        nbe = np.concatenate([[b0], be + kap * (xt - mu) ** 2 / (2 * (kap + 1))])
        over = nkap > kmax
        if over.any():
            s = np.where(over, kmax / nkap, 1.0)
            nal = np.maximum(np.where(over, nal * s, nal), 1e-3)
            nbe = np.where(over, nbe * s, nbe); nkap = np.where(over, kmax, nkap)
        mu, kap, al, be = nmu, nkap, nal, nbe
    return mrl

def onset_bocpd_level(v, minw=MIN_W):
    z = baseline_z(v, minw); mrl = bocpd_mrl(z)
    for t in range(1, len(z)):
        if mrl[t] <= 2 and mrl[t - 1] > 5:
            return int(t - mrl[t])
    return None

def onset_pelt(v, minw=MIN_W, pen=3.0):
    import ruptures as rpt
    n = len(v)
    if v.std() <= 0: return None
    z = ((v - v.mean()) / v.std()).reshape(-1, 1)
    bk = rpt.Pelt(model="l2", min_size=minw, jump=1).fit(z).predict(pen=pen * np.log(n))
    bk = [b for b in bk if b < n]
    if not bk: return None
    pts = [0] + bk + [n]; best, bj = None, -1
    for i in range(1, len(pts) - 1):
        a_, b_ = v[pts[i - 1]:pts[i]], v[pts[i]:pts[i + 1]]
        j = abs(b_.mean() - a_.mean()) / (v.std() + 1e-300)
        if j > bj: best, bj = pts[i], j
    return int(best)

EST = {"PELT3": lambda v: onset_pelt(v, pen=3.0), "PELT5": lambda v: onset_pelt(v, pen=5.0),
       "CUSUM": onset_cusum, "BOCPD_level": onset_bocpd_level}

# ---------------- 데이터 / 정상 세그먼트 고정 ----------------
store = load_store()
rng0 = np.random.default_rng(42)
norm_keys = {}
for ch in CHANNELS:
    ks = sorted(k for k, r in store.items() if k[0] == ch and r["anomaly"] == 0)
    if len(ks) > 300:
        ks = [ks[i] for i in sorted(rng0.choice(len(ks), 300, replace=False))]
    norm_keys[ch] = set(ks)
anom_keys = {ch: sorted(k for k, r in store.items() if k[0] == ch and r["anomaly"] == 1) for ch in CHANNELS}

# ---------------- 온셋 계산 (캐시) ----------------
cache = OUT / "rev4_onsets.csv"
cached = pd.read_csv(cache) if (cache.exists() and not args.recompute) else pd.DataFrame(
    columns=["estimator", "channel", "segment", "anomaly", "n", "onset"])
for name in args.estimators.split(","):
    if name in set(cached.estimator):
        continue
    t0 = time.time(); rows = []
    for ch in CHANNELS:
        todo = [(k, 1) for k in anom_keys[ch]] + [(k, 0) for k in sorted(norm_keys[ch])]
        for key, an in todo:
            v = store[key]["v"]
            try: o = EST[name](v)
            except Exception: o = None
            rows.append(dict(estimator=name, channel=ch, segment=key[1], anomaly=an, n=len(v),
                             onset=np.nan if o is None else o))
    cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    cached.to_csv(cache, index=False)
    print(f"[onset] {name} 완료 ({time.time() - t0:.0f}s)")

def to_dict(df):
    d = df.dropna(subset=["onset"])
    return {(c, int(s)): int(o) for c, s, o in zip(d.channel, d.segment, d.onset)}

# ---------------- 추정기별 (true, pseudo, 정상집합, 이상 총수) ----------------
sets = {}
t_on, _, ev_a = load_onsets(False); p_on, _, ev_n = load_onsets(True)
sets["KF_BOCPD(원본)"] = dict(
    true=t_on, pseudo=p_on,
    normals={ch: {k for k in ev_n if k[0] == ch} for ch in CHANNELS},
    n_anom={ch: len(anom_keys[ch]) for ch in CHANNELS})
for name in args.estimators.split(","):
    c = cached[cached.estimator == name]
    sets[name] = dict(true=to_dict(c[c.anomaly == 1]), pseudo=to_dict(c[c.anomaly == 0]),
                      normals=norm_keys, n_anom={ch: len(anom_keys[ch]) for ch in CHANNELS})

# ---------------- 1. 발화율 ----------------
rows = []
for est, S in sets.items():
    for ch in CHANNELS:
        na, nn = S["n_anom"][ch], len(S["normals"][ch])
        fa = sum(1 for k in S["true"] if k[0] == ch)
        fn = sum(1 for k in S["pseudo"] if k[0] == ch)
        rows.append(dict(estimator=est, channel=ch, n_anom=na, fired_anom=fa, rate_anom=round(fa / max(na, 1), 3),
                         n_norm=nn, fired_norm=fn, rate_norm=round(fn / max(nn, 1), 3)))
fr = pd.DataFrame(rows); fr.to_csv(OUT / "rev4_firing_rates.csv", index=False)
print("\n[1] 추정기별 발화율 (이상 vs 정상)"); print(fr.to_string(index=False))

# ---------------- 2. 대조 ----------------
rows = []
for est, S in sets.items():
    for ch in CHANNELS:
        T = onset_table(store, S["true"], 1, channel=ch)
        P = onset_table(store, S["pseudo"], 0, channel=ch)
        if len(T) < 3:
            continue
        R = placebo_table(store, ch, T.ratio.to_numpy(), 5, 42, allowed=S["normals"][ch])
        variants = {"all": (T, P, R)}
        pos = lambda d: d[(d.ratio >= .45) & (d.ratio <= .75)] if len(d) else d
        variants["pos_0.45-0.75"] = (pos(T), pos(P), pos(R))
        if len(P) >= 3:
            lo, hi = max(T.n.min(), P.n.min()), min(T.n.max(), P.n.max())
            lim = lambda d: d[(d.n >= lo) & (d.n <= hi)] if len(d) else d
            variants["common_length"] = (lim(T), lim(P), lim(R))
        for vn, (t, p, r) in variants.items():
            for cn, a, b in [("true_vs_pseudo", t, p), ("true_vs_random", t, r), ("pseudo_vs_random", p, r)]:
                for col in STAT_COLS:
                    rows.append(dict(estimator=est, channel=ch, variant=vn, contrast=cn, stat=col,
                                     **contrast_row(a, b, col, args.n_boot, seed=3)))
res = pd.DataFrame(rows); res.to_csv(OUT / "rev4_estimator_matrix.csv", index=False)

show = ["d1_lv", "d2_lv", "level_lv", "level_ms", "level_msg"]
for vn in ["all", "pos_0.45-0.75", "common_length"]:
    for cn in ["true_vs_pseudo", "true_vs_random"]:
        sub = res[(res.variant == vn) & (res.contrast == cn) & res.stat.isin(show)]
        if sub.empty: continue
        A_ = sub.pivot_table(index=["estimator", "channel"], columns="stat", values="A")[show].round(2)
        nb = sub[sub.stat == "d1_lv"].set_index(["estimator", "channel"])[["n_a", "n_b"]]
        print(f"\n[2] {vn} | {cn}  (A=P(앞>뒤); n_a=true onset 수, n_b=비교군 수)")
        print(A_.join(nb).to_string())