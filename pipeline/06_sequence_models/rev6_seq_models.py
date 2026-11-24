import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev6_seq_models.py — 시퀀스 모델: 채널 단독/제거 재학습 × 정규화 2종 × 음성 설정 2종
모델/학습 루프는 paper2_pipeline.py(define_torch_models, train_oof_deep)와 동일."""
import argparse, math, time, warnings
import torch, torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve
from rev_common import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--estimators", default="KF_BOCPD,PELT3,CUSUM")
ap.add_argument("--models", default="CNN1D,TCN,BiLSTM,BiGRU,TinyTransformer,LightMamba")
ap.add_argument("--modes", default="window_z,global_robust")
ap.add_argument("--neg", default="random", choices=["random", "pseudo"])
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--half", type=int, default=15)
ap.add_argument("--n-boot", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--gpu", type=int, default=0)
args = ap.parse_args()
SEED = args.seed; SFX = "" if args.neg == "random" else "_pseudo"
dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)

# ---------------- 모델 (원본 복사) ----------------
class CNN1D(nn.Module):
    def __init__(s, c_in, h=32):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(c_in, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.head = nn.Linear(h, 1)
    def forward(s, x): return s.head(s.net(x).squeeze(-1)).squeeze(-1)

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
    def forward(s, x): return s.head(s.b(x).mean(-1)).squeeze(-1)

class BiRNN(nn.Module):
    def __init__(s, c_in, h=32, kind="lstm"):
        super().__init__()
        s.r = (nn.LSTM if kind == "lstm" else nn.GRU)(c_in, h, batch_first=True, bidirectional=True)
        s.head = nn.Linear(2 * h, 1)
    def forward(s, x):
        o, _ = s.r(x.permute(0, 2, 1))
        return s.head(o.mean(1)).squeeze(-1)

class BiLSTM(BiRNN):
    def __init__(s, c_in): super().__init__(c_in, kind="lstm")
class BiGRU(BiRNN):
    def __init__(s, c_in): super().__init__(c_in, kind="gru")

class TinyTransformer(nn.Module):
    def __init__(s, c_in, d=32, nh=4, nl=2, max_len=200):
        super().__init__()
        s.proj = nn.Linear(c_in, d)
        pe = torch.zeros(max_len, d); pos = torch.arange(0, max_len).unsqueeze(1).float()
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
        s.pi = nn.Linear(c_in, d); s.A_log = nn.Parameter(torch.randn(d, ds) * 0.1)
        s.B, s.C = nn.Linear(d, ds), nn.Linear(d, ds); s.dl = nn.Linear(d, d)
        s.head = nn.Linear(d, 1); s.ds = ds
    def forward(s, x):
        u = s.pi(x.permute(0, 2, 1)); Bsz, T, D = u.shape
        dl = torch.nn.functional.softplus(s.dl(u)); Am = -torch.exp(s.A_log)
        Bc, Cc = s.B(u), s.C(u)
        h = torch.zeros(Bsz, D, s.ds, device=u.device); ys = []
        for t in range(T):
            dA = torch.exp(dl[:, t].unsqueeze(-1) * Am.unsqueeze(0))
            dBu = dl[:, t].unsqueeze(-1) * Bc[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA * h + dBu
            ys.append((h * Cc[:, t].unsqueeze(1)).sum(-1))
        return s.head(torch.stack(ys, 1).mean(1)).squeeze(-1)

ZOO = {"CNN1D": CNN1D, "TCN": TCN, "BiLSTM": BiLSTM, "BiGRU": BiGRU, "TinyTransformer": TinyTransformer, "LightMamba": LightMamba}
MODELS = [m for m in args.models.split(",") if m in ZOO]

def train_oof_deep(Cls, X, y, groups, folds, epochs, seed):
    oof = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        te = np.where(folds == f)[0]; tr_all = np.where(folds != f)[0]
        rng = np.random.default_rng(seed + f)
        ug = np.unique(groups[tr_all]); vg = set(rng.permutation(ug)[:max(1, int(0.15 * len(ug)))])
        is_val = np.array([g in vg for g in groups[tr_all]])
        tr, va = tr_all[~is_val], tr_all[is_val]
        if len(te) < 5 or len(tr) < 20 or y[tr].sum() < 3: continue
        torch.manual_seed(seed + f)
        model = Cls(X.shape[1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pw = torch.tensor([(1 - y[tr]).sum() / max(y[tr].sum(), 1)], dtype=torch.float32, device=dev)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt = torch.tensor(X[tr], device=dev); yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
        Xv = torch.tensor(X[va], device=dev); yv = torch.tensor(y[va], dtype=torch.float32, device=dev)
        best, bs, pat = np.inf, None, 0
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(tr), device=dev)
            for i in range(0, len(tr), 64):
                b = perm[i:i + 64]; opt.zero_grad(); crit(model(Xt[b]), yt[b]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(Xv), yv).item() if len(va) else 0.0
            if vl < best - 1e-4:
                best, pat = vl, 0; bs = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= 6: break
        if bs: model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(torch.tensor(X[te], device=dev))).cpu().numpy()
    return oof

# ---------------- 유틸 ----------------
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

def build_window(v, piv, half, mode):
    n = len(v); lo, hi = piv - half, piv + half
    if lo - 2 < 0 or hi > n: return None
    if mode == "global_robust":
        med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
        s = mad if mad > 0 else (v.std() if v.std() > 0 else 1.0); v = (v - med) / s
    d1 = np.diff(v); d2 = np.diff(d1)
    chs = [v[lo:hi], d1[lo - 1:hi - 1], d2[lo - 2:hi - 2]]
    if mode == "window_z": chs = [(c - c.mean()) / (c.std() + 1e-8) for c in chs]
    return np.stack(chs).astype(np.float32)

store = load_store()
cache = pd.read_csv(OUT / "rev4_onsets.csv")
_, _, ev_n = load_onsets(True)

def build(est, neg):
    if est == "KF_BOCPD":
        true_on, _, _ = load_onsets(False); pseudo_on, _, _ = load_onsets(True); allowed = ev_n
    else:
        ca = cache[(cache.estimator == est) & (cache.anomaly == 1)].dropna(subset=["onset"])
        cn = cache[(cache.estimator == est) & (cache.anomaly == 0)].dropna(subset=["onset"])
        true_on = {(a, int(s)): int(o) for a, s, o in zip(ca.channel, ca.segment, ca.onset)}
        pseudo_on = {(a, int(s)): int(o) for a, s, o in zip(cn.channel, cn.segment, cn.onset)}; allowed = None
    parts = []
    for ch in CHANNELS:
        T = onset_table(store, true_on, 1, channel=ch)
        if len(T) < 5: continue
        if neg == "random": P = placebo_table(store, ch, T.ratio.to_numpy(), 5, 42, allowed=allowed)
        else:
            P = onset_table(store, pseudo_on, 0, channel=ch)
            if len(P) < 5: continue
        T["label"] = 1; P["label"] = 0; parts += [T, P]
    d = pd.concat(parts, ignore_index=True)
    d["group"] = d.channel + "_" + d.segment.astype(str)
    return d.dropna(subset=STAT_COLS).reset_index(drop=True)

CH = {"level": [0], "d1": [1], "d2": [2], "deriv": [1, 2], "all": [0, 1, 2],
      "all_minus_d1": [0, 2], "all_minus_d2": [0, 1]}
CONTRASTS = [("d1-level", "d1", "level"), ("d2-level", "d2", "level"), ("deriv-level", "deriv", "level"),
             ("all-deriv (level 추가 이득)", "all", "deriv"), ("all-all_minus_d1 (d1 추가 이득)", "all", "all_minus_d1"),
             ("all-all_minus_d2 (d2 추가 이득)", "all", "all_minus_d2")]

def cluster_boot(y, groups, P, n_boot, seed):
    ug, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable"); parts = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed); d = {c: [] for c, _, _ in CONTRASTS}
    for _ in range(n_boot):
        idx = np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))]); yy = y[idx]
        a = {s: auc_y(yy, P[s][idx]) for s in CH}
        for c, s1, s0 in CONTRASTS: d[c].append(a[s1] - a[s0])
    out = {}
    for c in d:
        v = np.array(d[c]); v = v[np.isfinite(v)]
        out[c] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return out

def op_point(y, p):
    fpr, tpr, thr = roc_curve(y, p); i = int(np.argmax(tpr - fpr)); pred = p >= thr[i]
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return dict(prevalence=y.mean(), recall=tpr[i], far=fpr[i], precision=tp / max(tp + fp, 1))

# ---------------- 본 분석 ----------------
auc_rows, con_rows, op_rows = [], [], []
for est in args.estimators.split(","):
    d = build(est, args.neg)
    y_all = d.label.to_numpy(int); g_all = d.group.to_numpy()
    folds_all = assign_folds(g_all, y_all, 5, np.random.default_rng(SEED))
    for mode in args.modes.split(","):
        Xs, keep = [], []
        for i, r in d.iterrows():
            w = build_window(store[(r["channel"], int(r["segment"]))]["v"], int(r["pivot"]), args.half, mode)
            if w is not None: Xs.append(w); keep.append(i)
        Xs = np.stack(Xs); keep = np.array(keep)
        if mode == "global_robust":
            m_, s_ = Xs.mean(axis=(0, 2), keepdims=True), Xs.std(axis=(0, 2), keepdims=True) + 1e-8
            Xs = np.clip((Xs - m_) / s_, -10, 10).astype(np.float32)
        ys, gs, fs = y_all[keep], g_all[keep], folds_all[keep]
        print(f"\n=== {est} | {mode} | neg={args.neg}: 윈도우 {len(Xs)} (양성 {int(ys.sum())}, 음성 {int((1 - ys).sum())})", flush=True)
        for mname in MODELS:
            P = {}; t0 = time.time()
            for sname, ci in CH.items():
                oof = train_oof_deep(ZOO[mname], np.ascontiguousarray(Xs[:, ci, :]), ys, gs, fs, args.epochs, SEED)
                ok = np.isfinite(oof); P[sname] = np.where(ok, oof, 0.5)
                auc_rows.append(dict(estimator=est, mode=mode, model=mname, set=sname, auc=auc_y(ys[ok], oof[ok]),
                                     pr_auc=float(average_precision_score(ys[ok], oof[ok])) if ok.sum() > 10 else np.nan,
                                     n=int(ok.sum()), n_pos=int(ys[ok].sum())))
                if sname == "all": op_rows.append(dict(estimator=est, mode=mode, model=mname, **op_point(ys[ok], oof[ok])))
            ci_ = cluster_boot(ys, gs, P, args.n_boot, SEED)
            A_ = {r["set"]: r["auc"] for r in auc_rows if (r["estimator"], r["mode"], r["model"]) == (est, mode, mname)}
            for c, s1, s0 in CONTRASTS:
                con_rows.append(dict(estimator=est, mode=mode, model=mname, contrast=c, delta_auc=A_[s1] - A_[s0],
                                     ci_lo=ci_[c][0], ci_hi=ci_[c][1]))
            print(f"  {mname:16s} level={A_['level']:.3f} d1={A_['d1']:.3f} d2={A_['d2']:.3f} all={A_['all']:.3f} ({time.time() - t0:.0f}s)", flush=True)

auc = pd.DataFrame(auc_rows); con = pd.DataFrame(con_rows); op = pd.DataFrame(op_rows)
auc.to_csv(OUT / f"rev6_auc{SFX}.csv", index=False); con.to_csv(OUT / f"rev6_contrasts{SFX}.csv", index=False)
op.to_csv(OUT / f"rev6_operating_point{SFX}.csv", index=False)

print(f"\n[1] 모델 중앙값 AUC (neg={args.neg})")
print(auc.pivot_table(index=["estimator", "mode"], columns="set", values="auc", aggfunc="median")[list(CH)].round(3).to_string())
con["pos"] = con.ci_lo > 0; con["neg"] = con.ci_hi < 0
print("\n[2] ΔAUC: 모델 중앙값, CI>0 모델 비율, CI<0 모델 비율")
print(con.groupby(["estimator", "mode", "contrast"]).agg(median_delta=("delta_auc", "median"),
      frac_ci_pos=("pos", "mean"), frac_ci_neg=("neg", "mean")).round(3).to_string())
print("\n[3] 운영점 (all, 같은 OOF에서 Youden 선택 → 낙관적)")
print(op.groupby(["estimator", "mode"])[["prevalence", "recall", "far", "precision"]].median().round(3).to_string())