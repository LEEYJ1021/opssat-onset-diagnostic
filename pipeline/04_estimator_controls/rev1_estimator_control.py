import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev1: (1) 길이별 발화율  (2) true onset vs 같은 추정기의 pseudo-onset  (3) placebo 음수 중앙값 진단"""
import argparse
from rev_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=1000)
args = ap.parse_args()
store = load_store()
true_on, _, ev_a = load_onsets(False)
pseudo_on, _, ev_n = load_onsets(True)

# ---------- 1. 길이별 발화율 (이상 vs 정상) ----------
seg = pd.DataFrame([dict(channel=c, segment=s, anomaly=r["anomaly"], n=len(r["v"]))
                    for (c, s), r in store.items()])
seg = seg[[((c, s) in (ev_a if a == 1 else ev_n)) for c, s, a in zip(seg.channel, seg.segment, seg.anomaly)]].copy()
seg["fired"] = [((c, s) in (true_on if a == 1 else pseudo_on)) for c, s, a in zip(seg.channel, seg.segment, seg.anomaly)]
seg["len_bin"] = pd.cut(seg.n, [0, 50, 100, 200, 400, 10**9], right=False,
                        labels=["<50", "50-99", "100-199", "200-399", ">=400"])
t1 = (seg.groupby(["channel", "anomaly", "len_bin"], observed=True)
         .agg(n_seg=("fired", "size"), n_fired=("fired", "sum")).reset_index())
t1["rate"] = (t1.n_fired / t1.n_seg).round(3)
t1["group"] = t1.anomaly.map({1: "anomalous", 0: "normal"})
t1.drop(columns="anomaly").to_csv(OUT / "rev1_firing_by_length.csv", index=False)
print("\n[1] 길이별 발화율"); print(t1.drop(columns="anomaly").to_string(index=False))

# ---------- 2. true onset vs pseudo-onset vs random placebo ----------
rows = []
for ch in ["CADC0874", "CADC0888", "CADC0894"]:
    T = onset_table(store, true_on, 1, channel=ch)
    P = onset_table(store, pseudo_on, 0, channel=ch)
    pool = T.ratio.to_numpy()
    R = placebo_table(store, ch, pool, 5, 42, allowed=ev_n)
    lo_n, hi_n = max(T.n.min(), P.n.min()), min(T.n.max(), P.n.max())
    for variant in ["all", "common_length_support"]:
        t, p, r = T, P, R
        if variant != "all":
            t, p, r = [d[(d.n >= lo_n) & (d.n <= hi_n)] for d in (T, P, R)]
        for name, a, b in [("true_vs_pseudo", t, p), ("pseudo_vs_random", p, r), ("true_vs_random", t, r)]:
            for col in STAT_COLS:
                rows.append(dict(channel=ch, variant=variant, contrast=name, stat=col,
                                 **contrast_row(a, b, col, args.n_boot, seed=1)))
c2 = pd.DataFrame(rows); c2.to_csv(OUT / "rev1_pseudo_onset_control.csv", index=False)
show = c2[(c2.variant == "all") & (c2.stat.isin(["level_lv", "level_ms", "level_msg", "d1_lv", "d2_lv"]))]
print("\n[2] 추정기 통제 (A=P(앞>뒤)); 874는 pseudo n=4라 기술용")
print(show.pivot_table(index=["channel", "contrast"], columns="stat", values="A").round(2).to_string())

# ---------- 3. placebo R1 음수 중앙값 진단 (정방향 vs 시간역전) ----------
rows = []
for ch in CHANNELS:
    pool = onset_table(store, true_on, 1, channel=ch).ratio.to_numpy()
    rng = np.random.default_rng(42)
    for (c, s), r in sorted(store.items()):
        if c != ch or r["anomaly"] != 0 or (c, s) not in ev_n:
            continue
        v = r["v"]; n = len(v)
        for _ in range(5):
            piv = int(np.clip(int(round(rng.choice(pool) * n)), BURN_IN + MIN_W, n - MIN_W))
            f = stats_at(v, piv); b = stats_at(v[::-1].copy(), n - piv)
            rows.append(dict(channel=ch, ratio=piv / n, n=n,
                             d1_fwd=f["d1_lv"], d1_rev=b["d1_lv"], d2_fwd=f["d2_lv"], d2_rev=b["d2_lv"]))
d3 = pd.DataFrame(rows)
d3["ratio_bin"] = pd.cut(d3.ratio, [0, .3, .45, .6, .75, 1.0])
print("\n[3] placebo 로그분산비 중앙값: 정방향 vs 시간역전 (역전 시 부호가 뒤집히면 세그먼트 시작/끝 과도응답 의심)")
print(d3.groupby("channel")[["d1_fwd", "d1_rev", "d2_fwd", "d2_rev"]].median().round(2).to_string())
print(d3.groupby(["channel", "ratio_bin"], observed=True)["d1_fwd"].median().unstack().round(2).to_string())
d3.to_csv(OUT / "rev1_placebo_negative_diag.csv", index=False)