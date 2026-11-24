import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev2: 3표현 × 3통계량 대칭 비교, 양측 A + cluster CI, pivot 수/시드/고정윈도우 민감도, 200 draw 전파"""
import argparse
from rev_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=1000)
args = ap.parse_args()
store = load_store()
true_on, draws, _ = load_onsets(False)
_, _, ev_n = load_onsets(True)
ON = {h: onset_table(store, true_on, 1, h) for h in (None, 15, 30)}

# ---------- 1. 설정별 A + CI ----------
configs = [("rest_k5", None, 5), ("rest_k1", None, 1), ("fix15_k5", 15, 5), ("fix30_k5", 30, 5)]
rows = []
for ch in CHANNELS:
    pool = ON[None][ON[None].channel == ch].ratio.to_numpy()
    for name, h, k in configs:
        T = ON[h][ON[h].channel == ch]
        P = placebo_table(store, ch, pool, k, 42, h=h, allowed=ev_n)
        for col in STAT_COLS:
            rows.append(dict(channel=ch, config=name, stat=col, **contrast_row(T, P, col, args.n_boot, seed=2)))
t1 = pd.DataFrame(rows); t1.to_csv(OUT / "rev2_symmetric_A_ci.csv", index=False)
m = t1[t1.config == "rest_k5"]
print("\n[1] rest_k5: A(onset>placebo) [95% cluster CI]")
m = m.assign(txt=m.A.round(2).astype(str) + " [" + m.ci_lo.round(2).astype(str) + "," + m.ci_hi.round(2).astype(str) + "]")
print(m.pivot(index="channel", columns="stat", values="txt").to_string())
print("\n[1b] 양측 p (기술용, 분산·평균이동 모두)")
print(m.pivot(index="channel", columns="stat", values="p_two_sided").map(lambda x: f"{x:.1e}").to_string())

# ---------- 2. 시드 민감도 (k=1, 5; 20 seeds) ----------
rows = []
for ch in CHANNELS:
    pool = ON[None][ON[None].channel == ch].ratio.to_numpy()
    T = ON[None][ON[None].channel == ch]
    for k in (1, 5):
        for seed in range(20):
            P = placebo_table(store, ch, pool, k, seed, allowed=ev_n)
            for col in STAT_COLS:
                rows.append(dict(channel=ch, k=k, seed=seed, stat=col, A=auc_A(T[col], P[col])))
t2 = pd.DataFrame(rows)
s2 = t2.groupby(["channel", "k", "stat"]).A.agg(["min", "mean", "max"]).round(3).reset_index()
s2.to_csv(OUT / "rev2_seed_sensitivity.csv", index=False)
print("\n[2] 시드 20회 A 범위 (d1_lv, level_lv, level_ms)")
print(s2[s2.stat.isin(["d1_lv", "level_lv", "level_ms"])].to_string(index=False))

# ---------- 3. 200 draw 전파 (placebo 고정) ----------
rows = []
for ch in CHANNELS:
    pool = ON[None][ON[None].channel == ch].ratio.to_numpy()
    P = placebo_table(store, ch, pool, 5, 42, allowed=ev_n)
    for d in sorted(draws.draw.unique()):
        sub = draws[(draws.draw == d) & (draws.channel == ch)]
        onsets_d = {(c, int(s)): int(i) for c, s, i in zip(sub.channel, sub.segment, sub.onset_idx_sampled)}
        T = onset_table(store, onsets_d, 1, channel=ch)
        for col in STAT_COLS:
            rows.append(dict(channel=ch, draw=d, stat=col, A=auc_A(T[col], P[col]) if len(T) else np.nan))
t3 = pd.DataFrame(rows)
s3 = t3.groupby(["channel", "stat"]).A.quantile([.025, .5, .975]).unstack().round(3).reset_index()
s3.to_csv(OUT / "rev2_draw_propagation.csv", index=False)
print("\n[3] draw 간 A 분포 (onset 사후 불확실성만 반영; 추정기 선택 불확실성은 E3 참조)")
print(s3[s3.stat.isin(["d1_lv", "level_ms"])].to_string(index=False))