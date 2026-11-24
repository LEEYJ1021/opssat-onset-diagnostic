import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev3_synthetic.py — 추정기(KF local-linear-trend + BOCPD-NIG, 원본과 동일 규칙)의 선택 효과를 합성 신호로 측정"""
import argparse
from scipy import stats as sps
from rev_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--n-sim", type=int, default=300); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
N, T0 = args.n, args.n // 2

def kf_z(y, r, q):
    A = np.array([[1., 1.], [0., 1.]]); Q = np.array([[0., 0.], [0., q]])
    x = np.zeros(2); P = np.eye(2) * 10.; z = np.empty(len(y))
    for t, yt in enumerate(y):
        xp = A @ x; Pp = A @ P @ A.T + Q
        S = Pp[0, 0] + r; nu = yt - xp[0]
        K = Pp[:, 0] / S; x = xp + K * nu; P = Pp - np.outer(K, Pp[0])
        z[t] = nu / np.sqrt(max(S, 1e-8))
    return z

def bocpd(x, hazard=1 / 250., kmax=40.):
    n = len(x); R = [np.array([1.0])]
    mu, kap, al, be = np.array([0.]), np.array([1.]), np.array([1.]), np.array([1.])
    mrl = np.zeros(n, int)
    for t in range(n):
        xt = x[t]
        pp = np.nan_to_num(sps.t.pdf(xt, df=2 * al, loc=mu, scale=np.sqrt(be * (kap + 1) / (al * kap))),
                           nan=1e-12, posinf=1e-12)
        pr = R[t]
        new = np.concatenate([[np.sum(pr * pp * hazard)], pr * pp * (1 - hazard)])
        s = new.sum()
        new = new / s if (s > 0 and np.isfinite(s)) else np.eye(len(new))[0]
        R.append(new); mrl[t] = int(np.argmax(new))
        nmu = np.concatenate([[0.], (kap * mu + xt) / (kap + 1)])
        nkap = np.concatenate([[1.], kap + 1]); nal = np.concatenate([[1.], al + .5])
        nbe = np.concatenate([[1.], be + kap * (xt - mu) ** 2 / (2 * (kap + 1))])
        over = nkap > kmax
        if over.any():
            sc = np.where(over, kmax / nkap, 1.0)
            nal = np.maximum(np.where(over, nal * sc, nal), 1e-3); nbe = np.where(over, nbe * sc, nbe)
            nkap = np.where(over, kmax, nkap)
        mu, kap, al, be = nmu, nkap, nal, nbe
    for t in range(1, n):
        if mrl[t] <= 2 and mrl[t - 1] > 5:
            row = R[t + 1]; k = min(len(row), 30)
            post = {int(r): float(row[r]) for r in range(k) if row[r] > 1e-6}
            return t, post
    return None, None

def estimate_onset(y, r, q, rng, n_draws=200):
    conf, post = bocpd(kf_z(y, r, q))
    if post is None:
        return None
    rl = np.array(list(post.keys())); p = np.array(list(post.values())); p /= p.sum()
    cand = np.clip(conf - rl, BURN_IN, len(y) - 1)
    d = cand[rng.choice(len(cand), n_draws, p=p)]
    d = d[(d >= BURN_IN + MIN_W) & (d <= len(y) - MIN_W)]
    return int(round(np.median(d))) if len(d) else None

def gen(kind, rng, sig=1.0, rw=0.1):
    inc = rng.normal(0, rw, N); noise = rng.normal(0, sig, N)
    post = np.arange(N) >= T0
    if kind == "mean_shift":   noise = noise + 4 * sig * post
    if kind == "slope_change": inc = inc + 0.05 * post
    if kind == "noise_var_x3": noise = noise * np.where(post, 3., 1.)
    if kind == "increment_var_x5": inc = inc * np.where(post, 5., 1.)
    return np.cumsum(inc) + noise

rng = np.random.default_rng(args.seed)
cal = [gen("null", rng) for _ in range(50)]
r = np.var(np.concatenate([np.diff(c) for c in cal])) / 2
q = np.var(np.concatenate([np.diff(c, 2) for c in cal])) / 6

kinds = ["null", "mean_shift", "slope_change", "noise_var_x3", "increment_var_x5"]
rows = []
for kind in kinds:
    for i in range(args.n_sim):
        y = gen(kind, rng); on = estimate_onset(y, r, q, rng)
        base = dict(kind=kind, sim=i, fired=on is not None, onset=on if on is not None else -1,
                    err=(on - T0) if on is not None else np.nan)
        rows.append({**base, "pivot_type": "estimated", **(stats_at(y, on) if on else {c: np.nan for c in STAT_COLS})})
        for _ in range(5):   # 같은 신호의 random pivot (위치 0.3–0.8)
            pv = int(rng.integers(int(.3 * N), int(.8 * N)))
            rows.append({**base, "pivot_type": "random", **stats_at(y, pv)})
df = pd.DataFrame(rows); df.to_csv(OUT / "rev3_synthetic_raw.csv", index=False)

fire = df[df.pivot_type == "estimated"].groupby("kind").agg(
    fire_rate=("fired", "mean"), err_med=("err", "median"), err_abs_med=("err", lambda s: s.abs().median()))
print("\n[1] 발화율과 위치 오차(진짜 변화점 = %d)" % T0); print(fire.round(3).to_string())

null_rand = df[(df.kind == "null") & (df.pivot_type == "random")]
null_est = df[(df.kind == "null") & (df.pivot_type == "estimated") & df.fired]
out = []
for kind in kinds:
    est = df[(df.kind == kind) & (df.pivot_type == "estimated") & df.fired]
    for col in STAT_COLS:
        out.append(dict(kind=kind, stat=col, n_est=len(est),
                        A_vs_null_random=auc_A(est[col], null_rand[col]),
                        A_vs_null_pseudo=auc_A(est[col], null_est[col]) if len(null_est) > 5 else np.nan))
res = pd.DataFrame(out); res.to_csv(OUT / "rev3_synthetic_A.csv", index=False)
print("\n[2] 추정 onset의 A vs 변화 없는 신호의 random pivot (선택 효과 포함)")
print(res.pivot(index="kind", columns="stat", values="A_vs_null_random").round(2).to_string())
print("\n[3] 추정 onset의 A vs 변화 없는 신호에서 같은 추정기가 찍은 pseudo-onset")
print(res.pivot(index="kind", columns="stat", values="A_vs_null_pseudo").round(2).to_string())