import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev5_models.py — 추정기별 pivot 위에서 표현 단독/제거 재학습 (표 형태 모델)
요구: rev_common.py, revision_outputs/rev4_onsets.csv (rev4 실행 결과)"""
import argparse, time, warnings
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_curve
from rev_common import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--estimators", default="KF_BOCPD,PELT3,CUSUM")
ap.add_argument("--models", default="LogReg_L2,GaussianNB,kNN,SVM_RBF,RandomForest,LightGBM")  # all = 10개 전체
ap.add_argument("--n-boot", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--skip-mlp-diag", action="store_true")
ap.add_argument("--neg", default="random", choices=["random", "pseudo"])
ap.add_argument("--half", type=int, default=0)            # 0=세그먼트 전체 pre/post, 15/30=고정 윈도우 ±h
ap.add_argument("--common-length", action="store_true")   # 양성/음성 길이 공통 범위로 제한
args = ap.parse_args()
H = args.half if args.half > 0 else None
SFX = ("" if args.neg == "random" else "_pseudo") + (f"_h{H}" if H else "") + ("_cl" if args.common_length else "")
SEED = args.seed

# ---------------- 표현 / 통계량 집합 ----------------
LV = ["level_lv", "d1_lv", "d2_lv"]; MS = [c for c in STAT_COLS if c.endswith("_ms") or c.endswith("_msg")]
rep = lambda r: [c for c in STAT_COLS if c.startswith(r + "_")]
SETS = {
    "orig3": ["level_ms", "d1_lv", "d2_lv"],             # 원고 설정 재현
    "orig_level": ["level_ms"], "orig_d1": ["d1_lv"], "orig_d2": ["d2_lv"],
    "level_all": rep("level"), "d1_all": rep("d1"), "d2_all": rep("d2"),
    "all9": STAT_COLS,
    "all_minus_level": rep("d1") + rep("d2"),
    "all_minus_d1": rep("level") + rep("d2"),
    "all_minus_d2": rep("level") + rep("d1"),
    "var_only": LV, "mean_only": MS,
}
CONTRASTS = [("orig_d1-orig_level", "orig_d1", "orig_level"), ("orig_d2-orig_level", "orig_d2", "orig_level"),
             ("d1_all-level_all", "d1_all", "level_all"), ("d2_all-level_all", "d2_all", "level_all"),
             ("all9-all_minus_level (level 추가 이득)", "all9", "all_minus_level"),
             ("all9-all_minus_d1 (d1 추가 이득)", "all9", "all_minus_d1"),
             ("all9-all_minus_d2 (d2 추가 이득)", "all9", "all_minus_d2"),
             ("var_only-mean_only", "var_only", "mean_only")]

def zoo():
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    s = SEED
    z = {"LogReg_L2": lambda: LogisticRegression(class_weight="balanced", max_iter=3000),
         "LogReg_L1": lambda: LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000),
         "GaussianNB": lambda: GaussianNB(),
         "kNN": lambda: KNeighborsClassifier(n_neighbors=15),
         "SVM_RBF": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=s),
         "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=s, n_jobs=-1),
         "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=s, n_jobs=-1),
         "GradBoost": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=s)}
    try:
        from xgboost import XGBClassifier
        z["XGBoost"] = lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=s, verbosity=0)
    except ImportError: pass
    try:
        from lightgbm import LGBMClassifier
        z["LightGBM"] = lambda: LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=s, verbosity=-1)
    except ImportError: pass
    return z

ZOO = zoo()
MODELS = list(ZOO) if args.models == "all" else [m for m in args.models.split(",") if m in ZOO]

# ---------------- 유틸 ----------------
def auc_y(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 2 or n0 < 2: return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def assign_folds(groups, labels, k, rng):
    fold = {}
    for lab in (0, 1):
        gs = sorted({g for g, l in zip(groups, labels) if l == lab})
        for i, g in enumerate(rng.permutation(gs)): fold[g] = i % k
    return np.array([fold[g] for g in groups])

def oof_pred(X, y, folds, mk, scale=True):
    oof = np.full(len(y), np.nan)
    for f in range(5):
        tr, te = folds != f, folds == f
        if y[tr].sum() < 3: continue
        sc = StandardScaler().fit(X[tr]) if scale else None
        T = (lambda a: sc.transform(a)) if scale else (lambda a: a)
        try:
            m = mk(); m.fit(T(X[tr]), y[tr]); oof[te] = m.predict_proba(T(X[te]))[:, 1]
        except Exception as e:
            print("   fold 실패:", e)
    return oof

def cluster_boot(y, groups, P, n_boot, seed):
    """P: {set: oof}. 세그먼트(그룹) 단위 재표집으로 각 대조의 ΔAUC 95% CI."""
    ug, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable"); parts = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed); need = sorted({s for _, a, b in CONTRASTS for s in (a, b)})
    d = {c: [] for c, _, _ in CONTRASTS}
    for _ in range(n_boot):
        idx = np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))])
        yy = y[idx]; a = {s: auc_y(yy, P[s][idx]) for s in need}
        for c, s1, s0 in CONTRASTS: d[c].append(a[s1] - a[s0])
    out = {}
    for c, s1, s0 in CONTRASTS:
        v = np.array(d[c]); v = v[np.isfinite(v)]
        out[c] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return out

def op_point(y, p):
    ok = np.isfinite(p); y, p = y[ok], p[ok]
    fpr, tpr, thr = roc_curve(y, p); i = int(np.argmax(tpr - fpr)); t = thr[i]
    pred = p >= t; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return dict(prevalence=y.mean(), recall=tpr[i], far=fpr[i], precision=tp / max(tp + fp, 1), tp=tp, fp=fp)

# ---------------- 데이터셋 구성 ----------------
store = load_store()
cache = pd.read_csv(OUT / "rev4_onsets.csv")
_, _, ev_n = load_onsets(True)

def build(est, neg="random"):
    if est == "KF_BOCPD":
        true_on, _, _ = load_onsets(False); pseudo_on, _, _ = load_onsets(True); allowed = ev_n
    else:
        ca = cache[(cache.estimator == est) & (cache.anomaly == 1)].dropna(subset=["onset"])
        cn = cache[(cache.estimator == est) & (cache.anomaly == 0)].dropna(subset=["onset"])
        true_on = {(a, int(s)): int(o) for a, s, o in zip(ca.channel, ca.segment, ca.onset)}
        pseudo_on = {(a, int(s)): int(o) for a, s, o in zip(cn.channel, cn.segment, cn.onset)}; allowed = None
    parts = []
    for ch in CHANNELS:
        T = onset_table(store, true_on, 1, h=H, channel=ch)
        if len(T) < 5: continue
        if neg == "random":
            P = placebo_table(store, ch, T.ratio.to_numpy(), 5, 42, h=H, allowed=allowed)
        else:
            P = onset_table(store, pseudo_on, 0, h=H, channel=ch)
            if len(P) < 5: continue
        if args.common_length:
            lo_, hi_ = max(T.n.min(), P.n.min()), min(T.n.max(), P.n.max())
            T = T[(T.n >= lo_) & (T.n <= hi_)]; P = P[(P.n >= lo_) & (P.n <= hi_)]
            if len(T) < 5 or len(P) < 5: continue
        T["label"] = 1; P["label"] = 0; parts += [T, P]
    d = pd.concat(parts, ignore_index=True)
    d["group"] = d.channel + "_" + d.segment.astype(str)
    return d.dropna(subset=STAT_COLS).reset_index(drop=True)

# ---------------- 본 분석 ----------------
auc_rows, con_rows, op_rows = [], [], []
datasets = {}
for est in args.estimators.split(","):
    d = build(est, args.neg); datasets[est] = d
    y = d.label.to_numpy(int); g = d.group.to_numpy()
    folds = assign_folds(g, y, 5, np.random.default_rng(SEED))
    print(f"\n=== {est}: n={len(d)} (양성 {int(y.sum())}, 음성 {int((1 - y).sum())}, 그룹 {len(set(g))})", flush=True)
    for mname in MODELS:
        t0 = time.time(); P = {}
        for sname, cols in SETS.items():
            oof = oof_pred(d[cols].to_numpy(float), y, folds, ZOO[mname])
            ok = np.isfinite(oof); P[sname] = np.where(ok, oof, 0.5)
            auc_rows.append(dict(estimator=est, model=mname, set=sname, auc=auc_y(y[ok], oof[ok]),
                                 pr_auc=float(average_precision_score(y[ok], oof[ok])), n=int(ok.sum()), n_pos=int(y[ok].sum())))
            if sname in ("all9", "orig3"):
                op_rows.append(dict(estimator=est, model=mname, set=sname, **op_point(y, oof)))
        ci = cluster_boot(y, g, P, args.n_boot, SEED)
        A_ = {r["set"]: r["auc"] for r in auc_rows if r["estimator"] == est and r["model"] == mname}
        for c, s1, s0 in CONTRASTS:
            con_rows.append(dict(estimator=est, model=mname, contrast=c, delta_auc=A_[s1] - A_[s0], ci_lo=ci[c][0], ci_hi=ci[c][1]))
        print(f"  {mname:12s} orig_level={A_['orig_level']:.3f} orig_d1={A_['orig_d1']:.3f} all9={A_['all9']:.3f} "
              f"level_all={A_['level_all']:.3f} d1_all={A_['d1_all']:.3f} ({time.time() - t0:.0f}s)", flush=True)

auc = pd.DataFrame(auc_rows); con = pd.DataFrame(con_rows); op = pd.DataFrame(op_rows)
auc.to_csv(OUT / f"rev5_auc_table{SFX}.csv", index=False); con.to_csv(OUT / f"rev5_contrasts{SFX}.csv", index=False)
op.to_csv(OUT / f"rev5_operating_point{SFX}.csv", index=False)

print("\n[1] 모델 중앙값 AUC (행=추정기, 열=입력 집합)")
print(auc.pivot_table(index="estimator", columns="set", values="auc", aggfunc="median")[list(SETS)].round(3).T.to_string())
print("\n[2] ΔAUC 대조: 모델 중앙값, CI가 0 위인 모델 비율 / 0 아래인 모델 비율")
con["pos"] = con.ci_lo > 0; con["neg"] = con.ci_hi < 0
print(con.groupby(["estimator", "contrast"]).agg(median_delta=("delta_auc", "median"), frac_ci_pos=("pos", "mean"),
      frac_ci_neg=("neg", "mean")).round(3).to_string())
print("\n[3] 운영점 (Youden는 같은 OOF 예측에서 선택 → 낙관적, 보조 지표)")
print(op.groupby(["estimator", "set"])[["prevalence", "recall", "far", "precision"]].median().round(3).to_string())

# ---------------- MLP(17번째 모델) 진단 ----------------
if not args.skip_mlp_diag:
    from sklearn.neural_network import MLPClassifier
    d = datasets["KF_BOCPD"] if "KF_BOCPD" in datasets else next(iter(datasets.values()))
    y = d.label.to_numpy(int); g = d.group.to_numpy()
    folds = assign_folds(g, y, 5, np.random.default_rng(SEED)); X = d[SETS["orig3"]].to_numpy(float)
    rows = []
    for scale in (True, False):
        for es in (True, False):
            mk = lambda es=es: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=es, random_state=SEED)
            oof = oof_pred(X, y, folds, mk, scale=scale); ok = np.isfinite(oof)
            fa = [auc_y(y[(folds == f) & ok], oof[(folds == f) & ok]) for f in range(5)]
            rows.append(dict(scale=scale, early_stopping=es, pooled_auc=auc_y(y[ok], oof[ok]),
                             mean_fold_auc=np.nanmean(fa), min_fold_auc=np.nanmin(fa),
                             pred_std=float(np.nanstd(oof)), n_unique_pred=int(len(np.unique(np.round(oof[ok], 6))))))
    mlp = pd.DataFrame(rows); mlp.to_csv(OUT / f"rev5_mlp_diag{SFX}.csv", index=False)
    print("\n[4] MLP(16,8) 진단: pooled AUC vs 폴드별 AUC (pooled만 낮고 폴드별이 정상이면 폴드간 보정 차이, "
          "둘 다 <0.5이면 입력/라벨 파이프라인 점검)")
    print(mlp.round(3).to_string(index=False))