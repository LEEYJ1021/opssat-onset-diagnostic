import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
import argparse, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats as sps
from rev_common import *
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n-inj", type=int, default=60); ap.add_argument("--no-pelt", action="store_true"); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick: args.n_inj, args.no_pelt = 8, True
HS = [15, 30, None]; TOL = 5
KINDS = {"null": [0], "mean_shift": [2, 5, 10], "noise_var": [2, 3, 5], "slope": [0.05, 0.15, 0.5]}

def baseline_z(v, minw):
    b = max(minw, len(v) // 5); mu, sd = v[:b].mean(), v[:b].std()
    if sd <= 0: sd = v.std() if v.std() > 0 else 1.0
    return (v - mu) / sd
def onset_cusum(v, minw=MIN_W, k=0.5, h=5.0):
    z = baseline_z(v, minw); n = len(v); b = max(minw, n // 5); sp = sn = 0.0; sp0 = sn0 = b
    for t in range(b, n):
        if sp == 0: sp0 = t
        if sn == 0: sn0 = t
        sp = max(0.0, sp + z[t] - k); sn = max(0.0, sn - z[t] - k)
        if sp > h: return int(sp0)
        if sn > h: return int(sn0)
    return None
def onset_pelt(v, minw=MIN_W, pen=3.0):
    import ruptures as rpt
    n = len(v)
    if v.std() <= 0: return None
    z = ((v - v.mean()) / v.std()).reshape(-1, 1)
    bk = [b for b in rpt.Pelt(model="l2", min_size=minw, jump=1).fit(z).predict(pen=pen * np.log(n)) if b < n]
    if not bk: return None
    pts = [0] + bk + [n]; best, bj = None, -1
    for i in range(1, len(pts) - 1):
        j = abs(v[pts[i]:pts[i + 1]].mean() - v[pts[i - 1]:pts[i]].mean()) / (v.std() + 1e-300)
        if j > bj: best, bj = pts[i], j
    return int(best)
EST = {"CUSUM": onset_cusum}
if not args.no_pelt: EST["PELT3"] = onset_pelt

def noise_sig(v):
    d = np.diff(v); s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    return s if s > 0 else np.std(d) / np.sqrt(2)
def inject(v, t0, kind, mag, sig, rng):
    w = v.copy(); idx = np.arange(len(v)); post = idx >= t0
    if kind == "mean_shift": w[post] += mag * sig
    elif kind == "noise_var": w[post] += rng.normal(0, np.sqrt(mag ** 2 - 1) * sig, post.sum())
    elif kind == "slope": w[post] += mag * sig * (idx[post] - t0)
    return w
def run_est(fn, w):
    try: return fn(w)
    except Exception: return None

store = load_store(); rng = np.random.default_rng(42)
pos_src, neg_src = [], []
for ch in CHANNELS:
    ks = sorted(k for k, r in store.items() if k[0] == ch and r["anomaly"] == 0 and len(r["v"]) >= 80 and noise_sig(r["v"]) > 0)
    ks = [ks[i] for i in rng.permutation(len(ks))]; h2 = len(ks) // 2
    pos_src += ks[:h2][:args.n_inj]; neg_src += ks[h2:]
print(f"주입 원천 세그먼트 {len(pos_src)}개, 음성 세그먼트 {len(neg_src)}개", flush=True)

neg = {h: [] for h in HS}; bl = {e: [] for e in EST}
for k in neg_src:
    v = store[k]["v"]; n = len(v); piv = int(round(rng.uniform(.4, .6) * n))
    for h in HS: neg[h].append(dict(channel=k[0], **stats_at(v, piv, h)))
    for e, fn in EST.items(): bl[e].append(run_est(fn, v) is not None)
print("음성(무변화) 세그먼트에서 추정기 발화율:", {e: round(float(np.mean(v)), 3) for e, v in bl.items()}, flush=True)

pos = {}; rec = []
for kind, mags in KINDS.items():
    for mag in mags:
        rows = {h: [] for h in HS}; hits = {e: [] for e in EST}; fired = {e: [] for e in EST}
        for k in pos_src:
            v = store[k]["v"]; n = len(v); t0 = int(round(rng.uniform(.4, .6) * n)); sig = noise_sig(v)
            w = inject(v, t0, kind, mag, sig, rng)
            for h in HS: rows[h].append(dict(channel=k[0], **stats_at(w, t0, h)))
            for e, fn in EST.items():
                o = run_est(fn, w); fired[e].append(o is not None); hits[e].append(o is not None and abs(o - t0) <= TOL)
        for h in HS: pos[(kind, mag, h)] = rows[h]
        for e in EST: rec.append(dict(kind=kind, mag=mag, est=e, fire_rate=np.mean(fired[e]), hit_rate_within5=np.mean(hits[e]), n=len(pos_src)))
        print(f"{kind} {mag} 완료", flush=True)

def cv_auc(X, y):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)).fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    from scipy.stats import rankdata
    r = rankdata(oof); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

out = []
for (kind, mag, h), rws in pos.items():
    P = pd.DataFrame(rws); N = pd.DataFrame(neg[h]); row = dict(kind=kind, mag=mag, h="full" if h is None else h)
    for c in ["level_ms", "level_lv", "d1_lv", "d2_lv"]: row[f"A_{c}"] = auc_A(P[c], N[c])
    d = pd.concat([P.assign(label=1), N.assign(label=0)]).dropna(subset=STAT_COLS)
    row["n_pos"] = int(d.label.sum()); row["n_neg"] = int((1 - d.label).sum())
    row["cv_auc_all9"] = cv_auc(d[STAT_COLS].to_numpy(float), d.label.to_numpy(int)) if row["n_pos"] >= 10 and row["n_neg"] >= 10 else np.nan
    out.append(row)
res = pd.DataFrame(out); res.to_csv(OUT / "rev11_injection_auc.csv", index=False)
rc = pd.DataFrame(rec); rc.to_csv(OUT / "rev11_estimator_recovery.csv", index=False)
for h in ["15", "30", "full"]:
    print(f"\n[주입 검출력] 오라클 pivot, 윈도우 h={h} (A=P(주입>무변화), null 행이 0.5 근처여야 정상)")
    print(res[res.h.astype(str) == h].drop(columns="h").round(3).to_string(index=False))
print("\n[추정기 회복] fire_rate=발화율, hit_rate_within5=진짜 변화점 ±5 tick 이내로 찍은 비율"); print(rc.round(3).to_string(index=False))
