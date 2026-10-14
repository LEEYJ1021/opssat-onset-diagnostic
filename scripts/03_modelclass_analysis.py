#!/usr/bin/env python3
"""
03_modelclass_analysis.py -- dataset build + 16 models + attribution + agreement
(README sections 7 and 9).

Inputs : data/OPSSAT-AD/segments.csv
         outputs/stage2_mc_draws_raw.csv
         outputs/stage2_mc_segment_summary_corrected.csv
Outputs (outputs/):
  modelclass_dataset_tabular_raw.csv     all 177 + 4,980 pivots incl. NaN statistics (used by 04)
  modelclass_dataset_tabular.csv         rows with all three statistics defined (4,996 expected)
  modelclass_feature_importance.csv      model x representation attribution shares
  modelclass_performance_leaderboard.csv out-of-fold AUC, recall/FA at Youden optimum
  modelclass_agreement_summary.csv       Kendall's W, chi2, binomial
  modelclass_family_summary.csv          family-level means (Supplementary Table S2)
  modelclass_bootstrap_stability.csv     bootstrap criterion (RF, L2 logistic)
  modelclass_final_report.txt

IMPORTANT: run top-to-bottom and unmodified. One module-level Generator
(rng, seed 42) is consumed sequentially by pivot sampling, sequence-fold
assignment and the bootstrap; inserting draws changes later results.

Flags: --include-mlp (re-enable the excluded shallow net), --skip-deep,
       --skip-bootstrap.  Env DETERMINISTIC=1 enables deterministic torch kernels.
"""
import argparse
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--include-mlp", action="store_true")
ap.add_argument("--skip-deep", action="store_true")
ap.add_argument("--skip-bootstrap", action="store_true")
ARGS = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OPSSAT_DATA_DIR", ROOT / "data" / "OPSSAT-AD"))
OUT_DIR = Path(os.environ.get("OPSSAT_OUT_DIR", ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]
BURN_IN = 5
MIN_WINDOW_POINTS = 5
K_PLACEBO_DRAWS_PER_NORMAL = 5
MAX_NORMAL_PER_CHANNEL = 300
RANDOM_STATE = 42
SEQ_WINDOW_HALF = 15
N_BOOT_TABULAR = 200
EXPLAIN_SAMPLE_SIZE = 400
FEATURES = ["level", "diff", "diff2"]

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)


def section(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


# ---------------------------------------------------------------- dependencies
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("[info] xgboost missing: XGBoost skipped (the 16-model set will be incomplete)")
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False
    print("[info] lightgbm missing: LightGBM skipped (the 16-model set will be incomplete)")
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print("[info] shap missing: permutation importance only")

HAVE_TORCH = not ARGS.skip_deep
HAVE_CAPTUM = False
if HAVE_TORCH:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        torch.manual_seed(RANDOM_STATE)
        if os.environ.get("DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[info] torch device: {DEVICE}")
        try:
            from captum.attr import IntegratedGradients
            HAVE_CAPTUM = True
        except ImportError:
            print("[info] captum missing: manual Integrated Gradients (not Gauss-Legendre)")
    except ImportError:
        HAVE_TORCH = False
        print("[warn] torch missing: sequence models skipped")


# ---------------------------------------------------------------- features
def cohens_d_abs(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return float(abs(np.mean(a) - np.mean(b)) / sd)


def log_var_ratio(pre, post):
    if len(pre) < MIN_WINDOW_POINTS or len(post) < MIN_WINDOW_POINTS:
        return np.nan
    vp, vq = np.var(pre, ddof=1), np.var(post, ddof=1)
    if vp <= 0 or vq <= 0 or not np.isfinite(vp) or not np.isfinite(vq):
        return np.nan
    return float(np.log(vq / vp))


def compute_effects(values, pivot):
    """Equations (1)-(2): standardised level change and log variance ratios."""
    out = {}
    out["level"] = cohens_d_abs(values[pivot:], values[BURN_IN:pivot])
    d1 = np.diff(values)
    s1 = max(pivot - 1, 0)
    out["diff"] = log_var_ratio(d1[max(BURN_IN - 1, 0):s1], d1[s1:])
    d2 = np.diff(d1)
    s2 = max(pivot - 2, 0)
    out["diff2"] = log_var_ratio(d2[max(BURN_IN - 2, 0):s2], d2[s2:])
    return out


def build_3channel_window(values, pivot, half):
    lo, hi = pivot - half, pivot + half
    if lo - 2 < 0 or hi > len(values):
        return None

    def z(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    d1 = np.diff(values)
    d2 = np.diff(d1)
    lw, w1, w2 = values[lo:hi], d1[lo - 1:hi - 1], d2[lo - 2:hi - 2]
    if not (len(lw) == len(w1) == len(w2) == 2 * half):
        return None
    return np.stack([z(lw), z(w1), z(w2)], axis=0).astype(np.float32)


# ---------------------------------------------------------------- dataset
section("1. Dataset construction (README section 7)")
seg = pd.read_csv(DATA_DIR / "segments.csv", parse_dates=["timestamp"])
seg = seg[seg["channel"].isin(FINAL_CHANNELS)].sort_values(["channel", "segment", "timestamp"]).reset_index(drop=True)
values_by_seg = {(c, int(s)): g["value"].values for (c, s), g in seg.groupby(["channel", "segment"])}
anom_flag = {(c, int(s)): int(g["anomaly"].iloc[0]) for (c, s), g in seg.groupby(["channel", "segment"])}

draws = pd.read_csv(OUT_DIR / "stage2_mc_draws_raw.csv")
summ = pd.read_csv(OUT_DIR / "stage2_mc_segment_summary_corrected.csv")
reliable = summ.loc[summ["reliability_tier"].isin(["high", "medium"]) & summ["channel"].isin(FINAL_CHANNELS),
                    ["channel", "segment"]]
canonical = (draws.merge(reliable, on=["channel", "segment"], how="inner")
             .groupby(["channel", "segment"])["onset_idx_sampled"]
             .median().round().astype(int).rename("onset").reset_index())
print(f"Anomalous segments with high/medium reliability: {len(canonical)} (paper: 177)")

ratio_pool = {c: [] for c in FINAL_CHANNELS}
tab_rows, tab_group, seq_rows, seq_group = [], [], [], []

for _, row in canonical.iterrows():
    ch, sid, tau = row["channel"], int(row["segment"]), int(row["onset"])
    if anom_flag.get((ch, sid)) != 1:
        continue
    v = values_by_seg[(ch, sid)]
    n = len(v)
    if tau < BURN_IN + MIN_WINDOW_POINTS or tau > n - MIN_WINDOW_POINTS:
        continue
    tab_rows.append({"channel": ch, "segment": sid, "label": 1, **compute_effects(v, tau)})
    tab_group.append(f"{ch}_{sid}")
    ratio_pool[ch].append(tau / n)
    w = build_3channel_window(v, tau, SEQ_WINDOW_HALF)
    if w is not None:
        seq_rows.append({"label": 1, "window": w})
        seq_group.append(f"{ch}_{sid}")

for ch in FINAL_CHANNELS:
    pool = np.array(ratio_pool[ch])
    if len(pool) == 0:
        continue
    meta = (seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)]
            .groupby("segment").size().rename("n_points").reset_index())
    if len(meta) > MAX_NORMAL_PER_CHANNEL:
        meta = meta.sample(n=MAX_NORMAL_PER_CHANNEL, random_state=RANDOM_STATE)
    for _, nr in meta.iterrows():
        sid, n = int(nr["segment"]), int(nr["n_points"])
        v = values_by_seg[(ch, sid)]
        for _ in range(K_PLACEBO_DRAWS_PER_NORMAL):
            ratio = rng.choice(pool)
            lo = BURN_IN + MIN_WINDOW_POINTS
            tau = int(np.clip(int(round(ratio * n)), lo, max(n - MIN_WINDOW_POINTS, lo)))
            if tau < lo or tau > n - MIN_WINDOW_POINTS:
                continue
            tab_rows.append({"channel": ch, "segment": sid, "label": 0, **compute_effects(v, tau)})
            tab_group.append(f"{ch}_{sid}")
            w = build_3channel_window(v, tau, SEQ_WINDOW_HALF)
            if w is not None:
                seq_rows.append({"label": 0, "window": w})
                seq_group.append(f"{ch}_{sid}")

raw_df = pd.DataFrame(tab_rows)
raw_df["group"] = tab_group
raw_df.to_csv(OUT_DIR / "modelclass_dataset_tabular_raw.csv", index=False)   # README 7.7 (for script 04)

valid = raw_df[FEATURES].notna().all(axis=1).values
tab_df = raw_df.loc[valid].reset_index(drop=True)
tab_group = np.array(tab_group)[valid]
tab_df.drop(columns="group").to_csv(OUT_DIR / "modelclass_dataset_tabular.csv", index=False)
print(f"raw pivots: {len(raw_df)} (anomalous {int((raw_df.label==1).sum())}, placebo {int((raw_df.label==0).sum())})")
print(f"tabular set: {len(tab_df)} (pos {int((tab_df.label==1).sum())} / neg {int((tab_df.label==0).sum())})  paper: 4996 (168/4828)")
print(f"sequence set: {len(seq_rows)}  paper: 2733")


# ---------------------------------------------------------------- tabular models
section("2. Tabular models")


def build_tabular_zoo():
    z = [("LogReg_L2", "Linear", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000), "linear"),
         ("LogReg_L1", "Linear(sparse)", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000), "linear"),
         ("GaussianNB", "Probabilistic", GaussianNB(), "kernel"),
         ("kNN", "Instance-based", KNeighborsClassifier(n_neighbors=15), "kernel"),
         ("SVM_RBF", "Kernel", SVC(kernel="rbf", probability=True, class_weight="balanced"), "kernel"),
         ("RandomForest", "Tree ensemble(bagging)", RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "tree"),
         ("ExtraTrees", "Tree ensemble(bagging,extra-random)", ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), "tree"),
         ("GradBoost_sklearn", "Tree ensemble(boosting)", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE), "tree")]
    if HAVE_XGB:
        z.append(("XGBoost", "Tree ensemble(boosting)", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0), "tree"))
    if HAVE_LGBM:
        z.append(("LightGBM", "Tree ensemble(boosting)", LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1), "tree"))
    if ARGS.include_mlp:   # excluded in the paper: out-of-fold AUC ~0.403
        z.append(("MLP_shallow", "Neural net(shallow)", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE, early_stopping=True), "kernel"))
    return z


def normalize_shap(sv, n_features):
    """list / 2-D / 3-D SHAP output -> (n_samples, n_features) for the positive class."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv if (sv.ndim == 2 and sv.shape[1] == n_features) else None


def shap_importance(model, kind, X_train, X_explain):
    if not HAVE_SHAP:
        return None
    try:
        if kind == "tree":
            sv = shap.TreeExplainer(model).shap_values(X_explain)
        elif kind == "linear":
            sv = shap.LinearExplainer(model, X_train).shap_values(X_explain)
        else:
            bg = shap.kmeans(X_train, min(20, len(X_train)))
            sv = shap.KernelExplainer(model.predict_proba, bg).shap_values(X_explain, nsamples=100)
        sv = normalize_shap(sv, X_explain.shape[1])
        if sv is None:
            print("    [SHAP shape normalisation failed -> permutation importance]")
            return None
        imp = np.abs(sv).mean(axis=0).reshape(-1)
        tot = imp.sum()
        return imp / tot if tot > 0 else None
    except Exception as e:
        print(f"    [SHAP failed -> permutation importance] {e}")
        return None


def youden_row(name, family, y, p):
    fpr, tpr, _ = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return dict(model=name, family=family, auc=float(roc_auc_score(y, p)),
                recall_at_youden=float(tpr[i]), fa_rate_at_youden=float(fpr[i]))


tab_imp, tab_perf = [], []
X_all = tab_df[FEATURES].values.astype(float)
y_all = tab_df["label"].values.astype(int)
gkf = GroupKFold(n_splits=5)
for name, family, template, kind in build_tabular_zoo():
    print(f"\n[{name}] ({family})")
    oof = np.full(len(y_all), np.nan)
    fold_imps = []
    for tr, te in gkf.split(X_all, y_all, tab_group):
        sc = StandardScaler().fit(X_all[tr])
        Xtr, Xte = sc.transform(X_all[tr]), sc.transform(X_all[te])
        model = template.__class__(**template.get_params())
        try:
            model.fit(Xtr, y_all[tr])
        except Exception as e:
            print(f"    fold failed: {e}")
            continue
        oof[te] = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
        idx = np.random.choice(len(Xte), size=min(EXPLAIN_SAMPLE_SIZE, len(Xte)), replace=False)
        imp = shap_importance(model, kind, Xtr, Xte[idx])
        if imp is None:
            pi = permutation_importance(model, Xte, y_all[te], n_repeats=20, random_state=RANDOM_STATE, scoring="roc_auc")
            v = np.clip(pi.importances_mean, 0, None)
            imp = v / v.sum() if v.sum() > 0 else np.full(3, 1 / 3)
        fold_imps.append(imp)
    if not fold_imps:
        continue
    mean_imp = np.mean(fold_imps, axis=0)
    for f, v in zip(FEATURES, mean_imp):
        tab_imp.append(dict(model=name, family=family, feature=f, importance_share=float(v)))
    ok = ~np.isnan(oof)
    if ok.sum() > 10 and len(np.unique(y_all[ok])) == 2:
        tab_perf.append(youden_row(name, family, y_all[ok], oof[ok]))
        print(f"    AUC={tab_perf[-1]['auc']:.3f}  level/diff/diff2 = " + "/".join(f"{x:.3f}" for x in mean_imp))
tab_imp_df, tab_perf_df = pd.DataFrame(tab_imp), pd.DataFrame(tab_perf)


# ---------------------------------------------------------------- sequence models
section("3. Sequence models")
deep_imp, deep_perf = [], []

if HAVE_TORCH and len(seq_rows) > 50:
    class WindowDataset(Dataset):
        def __init__(self, rows):
            self.X = np.stack([r["window"] for r in rows]).astype(np.float32)
            self.y = np.array([r["label"] for r in rows], dtype=np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    class CNN1D(nn.Module):
        def __init__(self, c_in=3, h=32):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(c_in, h, 5, padding=2), nn.ReLU(),
                                     nn.Conv1d(h, h, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            return self.head(self.net(x).squeeze(-1)).squeeze(-1)

    class TCNBlock(nn.Module):
        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            self.pad = 4 * dilation
            self.conv = nn.Conv1d(c_in, c_out, 5, padding=self.pad, dilation=dilation)
            self.relu = nn.ReLU()
            self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            out = self.conv(x)[:, :, :-self.pad]
            return self.relu(out + self.res(x))

    class TCN(nn.Module):
        def __init__(self, c_in=3, h=32):
            super().__init__()
            self.blocks = nn.Sequential(TCNBlock(c_in, h, 1), TCNBlock(h, h, 2), TCNBlock(h, h, 4))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            return self.head(self.pool(self.blocks(x)).squeeze(-1)).squeeze(-1)

    class RNNNet(nn.Module):
        def __init__(self, cell, c_in=3, h=32):
            super().__init__()
            self.rnn = cell(c_in, h, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * h, 1)

        def forward(self, x):
            out, _ = self.rnn(x.permute(0, 2, 1))
            return self.head(out.mean(dim=1)).squeeze(-1)

    class BiLSTM(RNNNet):
        def __init__(self):
            super().__init__(nn.LSTM)

    class BiGRU(RNNNet):
        def __init__(self):
            super().__init__(nn.GRU)

    class TinyTransformer(nn.Module):
        def __init__(self, c_in=3, d=32, nhead=4, layers=2, max_len=200):
            super().__init__()
            self.proj = nn.Linear(c_in, d)
            pe = torch.zeros(max_len, d)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=64, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)
            h = self.encoder(self.proj(xt) + self.pe[:, :xt.size(1), :])
            return self.head(h.mean(dim=1)).squeeze(-1)

    class LightweightMamba(nn.Module):
        """Simplified selective SSM in plain PyTorch -- NOT the published Mamba."""
        def __init__(self, c_in=3, d=32, ds=16):
            super().__init__()
            self.proj_in = nn.Linear(c_in, d)
            self.A_log = nn.Parameter(torch.randn(d, ds) * 0.1)
            self.B_proj, self.C_proj = nn.Linear(d, ds), nn.Linear(d, ds)
            self.delta_proj = nn.Linear(d, d)
            self.head = nn.Linear(d, 1)
            self.ds = ds

        def forward(self, x):
            u = self.proj_in(x.permute(0, 2, 1))
            B_, T_, D = u.shape
            delta = torch.nn.functional.softplus(self.delta_proj(u))
            A = -torch.exp(self.A_log)
            Bc, Cc = self.B_proj(u), self.C_proj(u)
            h = torch.zeros(B_, D, self.ds, device=u.device)
            ys = []
            for t in range(T_):
                dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
                dBu = delta[:, t, :].unsqueeze(-1) * Bc[:, t, :].unsqueeze(1) * u[:, t, :].unsqueeze(-1)
                h = dA * h + dBu
                ys.append((h * Cc[:, t, :].unsqueeze(1)).sum(-1))
            return self.head(torch.stack(ys, dim=1).mean(dim=1)).squeeze(-1)

    deep_zoo = [("CNN1D", "Convolutional", CNN1D), ("TCN", "Convolutional(dilated causal)", TCN),
                ("BiLSTM", "Recurrent", BiLSTM), ("BiGRU", "Recurrent", BiGRU),
                ("TinyTransformer", "Attention", TinyTransformer), ("LightMamba", "State-space(SSM)", LightweightMamba)]

    def train_epoch(model, loader, opt, crit):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        return tot / len(loader.dataset)

    @torch.no_grad()
    def eval_model(model, loader, crit):
        model.eval()
        tot, probs, ys = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            lg = model(xb)
            tot += crit(lg, yb).item() * len(yb)
            probs.append(torch.sigmoid(lg).cpu().numpy())
            ys.append(yb.cpu().numpy())
        return tot / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)

    def manual_ig(model, x, steps=32):
        model.eval()
        base = torch.zeros_like(x)
        tot = torch.zeros_like(x)
        for a in np.linspace(0, 1, steps):
            xi = (base + a * (x - base)).clone().requires_grad_(True)
            g, = torch.autograd.grad(model(xi).sum(), xi)
            tot += g
        return ((tot / steps) * (x - base)).detach()

    groups = np.array(seq_group)
    uniq = np.unique(groups)
    fold_of = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    fold_row = np.array([fold_of[g] for g in groups])
    y_seq = np.array([r["label"] for r in seq_rows], dtype=np.float32)
    pos_weight = torch.tensor([(1 - y_seq).sum() / max(y_seq.sum(), 1)], dtype=torch.float32).to(DEVICE)

    for name, family, Cls in deep_zoo:
        print(f"\n[{name}] ({family})")
        oof = np.full(len(y_seq), np.nan)
        sal = []
        for k in range(5):
            te, tr = np.where(fold_row == k)[0], np.where(fold_row != k)[0]
            if len(te) < 5 or len(tr) < 20:
                continue
            tr_ds = WindowDataset([seq_rows[i] for i in tr])
            te_ds = WindowDataset([seq_rows[i] for i in te])
            trl = DataLoader(tr_ds, batch_size=32, shuffle=True)
            tel = DataLoader(te_ds, batch_size=64, shuffle=False)
            model = Cls().to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            best, ctr, state = np.inf, 0, None
            for _ in range(60):
                train_epoch(model, trl, opt, crit)
                vl, _, _ = eval_model(model, tel, crit)   # NOTE: monitors held-out fold (upper-range AUC)
                if vl < best - 1e-4:
                    best, ctr = vl, 0
                    state = {a: b.clone() for a, b in model.state_dict().items()}
                else:
                    ctr += 1
                    if ctr >= 6:
                        break
            if state is not None:
                model.load_state_dict(state)
            _, probs, _ = eval_model(model, tel, crit)
            oof[te] = probs
            idx = np.random.choice(len(te_ds), size=min(EXPLAIN_SAMPLE_SIZE, len(te_ds)), replace=False)
            Xe = torch.from_numpy(np.stack([te_ds.X[i] for i in idx])).to(DEVICE)
            model.eval()
            with torch.backends.cudnn.flags(enabled=False):   # cuDNN RNN backward fails in eval mode
                attr = (IntegratedGradients(model).attribute(Xe, n_steps=32) if HAVE_CAPTUM else manual_ig(model, Xe))
            ci = attr.abs().sum(dim=2).mean(dim=0).detach().cpu().numpy()
            sal.append(ci / ci.sum() if ci.sum() > 0 else np.full(3, 1 / 3))
        if not sal:
            continue
        m = np.mean(sal, axis=0)
        for f, v in zip(FEATURES, m):
            deep_imp.append(dict(model=name, family=family, feature=f, importance_share=float(v)))
        ok = ~np.isnan(oof)
        if ok.sum() > 10 and len(np.unique(y_seq[ok])) == 2:
            deep_perf.append(youden_row(name, family, y_seq[ok], oof[ok]))
            print(f"    AUC={deep_perf[-1]['auc']:.3f}  level/diff/diff2 = " + "/".join(f"{x:.3f}" for x in m))
else:
    print("[skipped] sequence models")
deep_imp_df, deep_perf_df = pd.DataFrame(deep_imp), pd.DataFrame(deep_perf)


# ---------------------------------------------------------------- merge + leaderboard
section("4. Leaderboard")
imp_df = pd.concat([tab_imp_df, deep_imp_df], ignore_index=True)
imp_df.to_csv(OUT_DIR / "modelclass_feature_importance.csv", index=False)
perf_df = pd.concat([tab_perf_df, deep_perf_df], ignore_index=True).sort_values("auc", ascending=False)
perf_df.to_csv(OUT_DIR / "modelclass_performance_leaderboard.csv", index=False)
print(perf_df.to_string(index=False))


# ---------------------------------------------------------------- agreement
section("5. Agreement across models")
pivot = None
if len(imp_df):
    pivot = imp_df.pivot_table(index=["model", "family"], columns="feature", values="importance_share").reset_index()
    pivot = pivot.reindex(columns=["model", "family"] + FEATURES)
    ranks = pivot[FEATURES].rank(axis=1, ascending=False, method="average").values
    m, n = ranks.shape
    Rsum = ranks.sum(axis=0)
    S = np.sum((Rsum - m * (n + 1) / 2) ** 2)          # equation (8)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W                              # equation (9)
    chi2_p = 1 - stats.chi2.cdf(chi2, df=n - 1)
    level_top = pivot["level"] >= pivot[["diff", "diff2"]].max(axis=1)
    k_not_top = int((~level_top).sum())
    bt = stats.binomtest(k_not_top, n=m, p=2 / 3, alternative="greater")
    print(f"models={m}  Kendall W={W:.4f}  chi2={chi2:.3f} (d.f.={n-1})  p={chi2_p:.2e}")
    print(f"level not ranked first: {k_not_top}/{m}   binomial p={bt.pvalue:.4f}   (2/3)^{m}={(2/3)**m:.4f}")
    print(f"level ranked last in {int((ranks[:,0]==3).sum())}/{m}; second-difference > first in {int((pivot['diff2']>pivot['diff']).sum())}/{m}")
    fam = pivot.groupby("family").agg(n_models=("model", "count"), mean_level=("level", "mean"),
                                      mean_diff=("diff", "mean"), mean_diff2=("diff2", "mean")).reset_index()
    fam["diff_plus_diff2_beats_level"] = (fam["mean_diff"] + fam["mean_diff2"]) > fam["mean_level"]
    fam.to_csv(OUT_DIR / "modelclass_family_summary.csv", index=False)
    print(fam.to_string(index=False))
    pd.DataFrame([dict(n_models=m, kendalls_W=W, chi2_stat=chi2, chi2_p_value=chi2_p,
                       n_models_level_not_top=k_not_top, binomial_p_value=bt.pvalue,
                       n_level_last=int((ranks[:, 0] == 3).sum()))]
                 ).to_csv(OUT_DIR / "modelclass_agreement_summary.csv", index=False)


# ---------------------------------------------------------------- bootstrap
section("6. Bootstrap criterion (RF, L2 logistic)")
boot_rows = []
if not ARGS.skip_bootstrap:
    pos_pool, neg_pool = np.where(y_all == 1)[0], np.where(y_all == 0)[0]
    reps = [("RandomForest_boot", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("LogReg_boot", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))]
    for name, template in reps:
        wins = np.zeros(N_BOOT_TABULAR, dtype=bool)
        for b in range(N_BOOT_TABULAR):
            idx = np.concatenate([rng.choice(pos_pool, len(pos_pool), replace=True),
                                  rng.choice(neg_pool, len(neg_pool), replace=True)])
            Xs = StandardScaler().fit_transform(X_all[idx])
            model = template.__class__(**template.get_params())
            try:
                model.fit(Xs, y_all[idx])
            except Exception:
                continue
            pi = permutation_importance(model, Xs, y_all[idx], n_repeats=10, random_state=b, scoring="roc_auc")
            imp = np.clip(pi.importances_mean, 0, None)
            wins[b] = (imp[1] + imp[2]) > imp[0]
        boot_rows.append(dict(model=name, n_boot=N_BOOT_TABULAR, frac_diff_or_diff2_beats_level=float(wins.mean())))
        print(f"[{name}] difference representations > level in {wins.mean():.1%} of {N_BOOT_TABULAR} resamples")
pd.DataFrame(boot_rows).to_csv(OUT_DIR / "modelclass_bootstrap_stability.csv", index=False)


# ---------------------------------------------------------------- report
section("7. Report")
lines = ["Model-class analysis -- summary", "=" * 60,
         f"models: {imp_df['model'].nunique() if len(imp_df) else 0} "
         f"(tabular {tab_imp_df['model'].nunique() if len(tab_imp_df) else 0}, "
         f"sequence {deep_imp_df['model'].nunique() if len(deep_imp_df) else 0})"]
if pivot is not None:
    lines += [f"Kendall's W = {W:.4f}, chi2 = {chi2:.2f}, p = {chi2_p:.2e}",
              f"models with level not ranked first: {k_not_top}/{m} (binomial p = {bt.pvalue:.4f})"]
lines += ["", "Attribution shares describe how trained models use each representation;",
          "they are not information content. AUCs are upper-range (early stopping on the",
          "held-out fold; thresholds chosen on the same predictions) and are NOT detection performance."]
(OUT_DIR / "modelclass_final_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
