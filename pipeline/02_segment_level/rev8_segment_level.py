import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from rev_common import BASE, OUT
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--n-boot", type=int, default=300)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick: args.reps, args.n_boot = 1, 30

df = pd.read_csv(BASE / "paper2_outputs" / "p2_segment_features.csv").replace([np.inf, -np.inf], np.nan)
LEVEL = ["mean", "skew", "kurtosis", "half_shift", "abs_slope"]
DISP = ["var", "range", "iqr"]
DERIV = ["diff_var", "diff2_var", "diff_peaks", "diff2_peaks", "rough1", "rough2"]
COV = ["len", "duration", "sampling"]
SETS = {"cov_only": COV, "level": LEVEL, "disp": DISP, "deriv": DERIV,
        "provided4": ["diff_var", "diff2_var", "diff_peaks", "diff2_peaks"],
        "level+disp": LEVEL + DISP, "all": LEVEL + DISP + DERIV, "all+cov": LEVEL + DISP + DERIV + COV}
allc = sorted({c for v in SETS.values() for c in v})
for c in allc:
    df[c] = df.groupby("channel")[c].transform(lambda s: s.fillna(s.median())).fillna(0)
PAIRS = [("deriv", "level"), ("provided4", "level"), ("all", "level+disp"), ("deriv", "cov_only"), ("level+disp", "cov_only")]
MODELS = {"logreg": lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
          "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=4)}

def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 2 or n0 < 2: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def run_cv(X, y, mk, seed, reps):
    sc = np.zeros(len(y)); aucs = []
    for r in range(reps):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            m = make_pipeline(QuantileTransformer(n_quantiles=max(10, min(100, len(tr))), output_distribution="normal"), mk())
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(auc(y, oof)); sc += oof
    return sc / reps, float(np.mean(aucs))

def boot(y, S, nb, seed=0):
    rng = np.random.default_rng(seed); n = len(y); R = {k: [] for k in S}; D = {p: [] for p in PAIRS}
    for _ in range(nb):
        i = rng.integers(0, n, n); yy = y[i]
        if yy.sum() < 3 or (1 - yy).sum() < 3: continue
        a = {k: auc(yy, S[k][i]) for k in S}
        for k in S: R[k].append(a[k])
        for p in PAIRS: D[p].append(a[p[0]] - a[p[1]])
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return {k: q(v) for k, v in R.items()}, {p: q(v) for p, v in D.items()}

def match_len(d, rng):
    d = d.copy()
    try: d["_b"] = pd.qcut(d["len"], 4, labels=False, duplicates="drop")
    except Exception: return d.iloc[:0]
    keep = []
    for _, g in d.groupby("_b"):
        p, q = g[g.anomaly == 1], g[g.anomaly == 0]; k = min(len(p), len(q))
        if k: keep += [p.sample(k, random_state=int(rng.integers(1e9))), q.sample(k, random_state=int(rng.integers(1e9)))]
    return pd.concat(keep) if keep else d.iloc[:0]

rng = np.random.default_rng(0); rows = []
for ch in sorted(df.channel.unique()):
    d0 = df[df.channel == ch]
    strata = {"all": d0, "len_matched": match_len(d0, rng)}
    for s in sorted(d0.sampling.unique()): strata[f"sampling={s:g}"] = d0[d0.sampling == s]
    for sn, sub in strata.items():
        y = sub.anomaly.to_numpy(int); npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos < 10 or nneg < 10: continue
        for mn, mk in MODELS.items():
            S, A = {}, {}
            for k, cols in SETS.items(): S[k], A[k] = run_cv(sub[cols].to_numpy(float), y, mk, 42, args.reps)
            ci, dci = boot(y, S, args.n_boot)
            for k in SETS: rows.append(dict(channel=ch, stratum=sn, model=mn, kind="auc", name=k, value=A[k], lo=ci[k][0], hi=ci[k][1], n_pos=npos, n_neg=nneg))
            for p in PAIRS: rows.append(dict(channel=ch, stratum=sn, model=mn, kind="delta", name=f"{p[0]}-{p[1]}", value=A[p[0]] - A[p[1]], lo=dci[p][0], hi=dci[p][1], n_pos=npos, n_neg=nneg))
        print(f"{ch} {sn} n_pos={npos} n_neg={nneg} done", flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT / "rev8_segment_level.csv", index=False)
for mn in MODELS:
    for sn in ["all", "len_matched"]:
        t = res[(res.kind == "auc") & (res.model == mn) & (res.stratum == sn)]
        if t.empty: continue
        print(f"\n[AUC] model={mn} stratum={sn}")
        pv = t.pivot_table(index="channel", columns="name", values="value")[list(SETS)].round(3)
        print(pv.join(t.groupby("channel")[["n_pos", "n_neg"]].first()).to_string())
        t = res[(res.kind == "delta") & (res.model == mn) & (res.stratum == sn)].copy()
        t["txt"] = t.value.round(3).astype(str) + " [" + t.lo.round(2).astype(str) + "," + t.hi.round(2).astype(str) + "]"
        print(f"[delta AUC, 95% bootstrap CI] model={mn} stratum={sn}")
        print(t.pivot(index="channel", columns="name", values="txt").to_string())
