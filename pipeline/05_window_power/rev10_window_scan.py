import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
import argparse, warnings
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from rev_common import *
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
HS = [8, 10, 15, 20, 30, 50, None]; ESTS = ["PELT3", "CUSUM"]
if args.quick: args.seeds, HS, ESTS = 1, [15, None], ["CUSUM"]
store = load_store(); cache = pd.read_csv(OUT / "rev4_onsets.csv")
SETS = {"level_all": [c for c in STAT_COLS if c.startswith("level_")], "d1_all": [c for c in STAT_COLS if c.startswith("d1_")],
        "d2_all": [c for c in STAT_COLS if c.startswith("d2_")], "all9": STAT_COLS}
MODELS = {"logreg": lambda s: LogisticRegression(class_weight="balanced", max_iter=3000),
          "rf": lambda s: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=s, n_jobs=4)}

def auc_y(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 2 or n0 < 2: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def assign_folds(groups, labels, k, rng):
    fold = {}
    for lab in (0, 1):
        gs = sorted({g for g, l in zip(groups, labels) if l == lab})
        for i, g in enumerate(rng.permutation(gs)): fold[g] = i % k
    return np.array([fold[g] for g in groups])
def onsets(est, an):
    c = cache[(cache.estimator == est) & (cache.anomaly == an)].dropna(subset=["onset"])
    return {(a, int(s)): int(o) for a, s, o in zip(c.channel, c.segment, c.onset)}
def build(true_on, pseudo_on, h, cl):
    parts = []
    for ch in CHANNELS:
        T = onset_table(store, true_on, 1, h=h, channel=ch); P = onset_table(store, pseudo_on, 0, h=h, channel=ch)
        if len(T) < 5 or len(P) < 5: continue
        if cl:
            lo, hi = max(T.n.min(), P.n.min()), min(T.n.max(), P.n.max())
            T = T[(T.n >= lo) & (T.n <= hi)]; P = P[(P.n >= lo) & (P.n <= hi)]
            if len(T) < 5 or len(P) < 5: continue
        T["label"] = 1; P["label"] = 0; parts += [T, P]
    if not parts: return None
    d = pd.concat(parts, ignore_index=True); d["group"] = d.channel + "_" + d.segment.astype(str)
    return d.dropna(subset=STAT_COLS).reset_index(drop=True)

rows = []
for est in ESTS:
    t_on, p_on = onsets(est, 1), onsets(est, 0)
    for cl in (False, True):
        for h in HS:
            d = build(t_on, p_on, h, cl)
            hs = "full" if h is None else str(h); vn = "common_length" if cl else "all"
            if d is None or d.label.sum() < 30 or (1 - d.label).sum() < 30:
                print(f"{est} {vn} h={hs}: 표본 부족, 건너뜀"); continue
            y = d.label.to_numpy(int); g = d.group.to_numpy()
            for seed in range(args.seeds):
                folds = assign_folds(g, y, 5, np.random.default_rng(seed))
                for mn, mk in MODELS.items():
                    for sn, cols in SETS.items():
                        X = d[cols].to_numpy(float); oof = np.full(len(y), np.nan)
                        for f in range(5):
                            tr, te = folds != f, folds == f
                            sc = StandardScaler().fit(X[tr]); m = mk(seed).fit(sc.transform(X[tr]), y[tr])
                            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
                        rows.append(dict(est=est, variant=vn, h=hs, seed=seed, model=mn, set=sn, auc=auc_y(y, oof), n_pos=int(y.sum()), n_neg=int((1 - y).sum())))
            print(f"{est} {vn} h={hs}: n_pos={int(y.sum())} n_neg={int((1-y).sum())} 완료", flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT / "rev10_window_scan.csv", index=False)
g = res.groupby(["model", "est", "variant", "h", "set"]).auc.agg(["mean", "std"]).reset_index()
g["txt"] = g["mean"].round(3).astype(str) + "±" + g["std"].fillna(0).round(3).astype(str)
npn = res.groupby(["est", "variant", "h"])[["n_pos", "n_neg"]].first()
order = {"8": 0, "10": 1, "15": 2, "20": 3, "30": 4, "50": 5, "full": 6}
for mn in MODELS:
    t = g[g.model == mn].pivot(index=["est", "variant", "h"], columns="set", values="txt")[list(SETS)].join(npn)
    t = t.reset_index(); t["o"] = t.h.map(order); t = t.sort_values(["est", "variant", "o"]).drop(columns="o")
    print(f"\n[window scan] model={mn} (평균±시드 표준편차, {args.seeds}시드)"); print(t.to_string(index=False))
