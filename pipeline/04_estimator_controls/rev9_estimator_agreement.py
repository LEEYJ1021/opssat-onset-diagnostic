import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
from itertools import combinations
from scipy.stats import spearmanr
from rev_common import *
store = load_store(); cache = pd.read_csv(OUT / "rev4_onsets.csv")
true_on, _, _ = load_onsets(False)
ons = {"KF_BOCPD": true_on}
for e in cache.estimator.unique():
    c = cache[(cache.estimator == e) & (cache.anomaly == 1)].dropna(subset=["onset"])
    ons[e] = {(a, int(s)): int(o) for a, s, o in zip(c.channel, c.segment, c.onset)}
rng = np.random.default_rng(0)

def agree(oa, ob, n, tol=5):
    obs = float(np.mean(np.abs(oa - ob) <= tol))
    rb = ob / n; null = np.mean([np.mean(np.abs(oa - rng.permutation(rb) * n) <= tol) for _ in range(200)])
    rho = spearmanr(oa / n, ob / n).correlation if len(oa) > 3 else np.nan
    return obs, float(null), float(rho), float(np.median(np.abs(oa - ob) / n))

rows = []
for A_, B_ in combinations(ons, 2):
    pool = dict(oa=[], ob=[], n=[]); inter_t = union_t = 0
    for ch in CHANNELS:
        ka = {k for k in ons[A_] if k[0] == ch}; kb = {k for k in ons[B_] if k[0] == ch}
        com = sorted(ka & kb); inter_t += len(com); union_t += len(ka | kb)
        if len(com) < 5:
            rows.append(dict(pair=f"{A_}|{B_}", channel=ch, n_common=len(com), jaccard=len(com) / max(len(ka | kb), 1))); continue
        n = np.array([len(store[k]["v"]) for k in com]); oa = np.array([ons[A_][k] for k in com]); ob = np.array([ons[B_][k] for k in com])
        o, nu, rho, md = agree(oa, ob, n)
        rows.append(dict(pair=f"{A_}|{B_}", channel=ch, n_common=len(com), jaccard=len(com) / len(ka | kb), within5=o, null_within5=nu, rho_pos=rho, med_abs_rel_diff=md))
        pool["oa"] += list(oa); pool["ob"] += list(ob); pool["n"] += list(n)
    if len(pool["oa"]) >= 5:
        o, nu, rho, md = agree(np.array(pool["oa"]), np.array(pool["ob"]), np.array(pool["n"]))
        rows.append(dict(pair=f"{A_}|{B_}", channel="ALL", n_common=len(pool["oa"]), jaccard=inter_t / max(union_t, 1), within5=o, null_within5=nu, rho_pos=rho, med_abs_rel_diff=md))
res = pd.DataFrame(rows); res.to_csv(OUT / "rev9_estimator_agreement.csv", index=False)
print("[ALL 채널 합산] within5=두 추정기 onset이 5 tick 이내인 비율, null_within5=세그먼트를 섞었을 때의 우연 기대값")
print(res[res.channel == "ALL"].round(3).to_string(index=False))
print("\n[채널별]"); print(res[res.channel != "ALL"].round(3).to_string(index=False))
