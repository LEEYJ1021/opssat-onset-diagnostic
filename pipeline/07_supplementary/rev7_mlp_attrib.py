import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "common"))
"""rev7: 17번째 모델(MLP) 포함 attribution 민감도 (tabular, permutation importance)"""
import argparse, time, warnings
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from rev_common import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--n-repeats", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args(); SEED = args.seed
FEATS = ["level_ms", "d1_lv", "d2_lv"]; NAMES = ["level", "first", "second"]

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
    z["MLP_ES_off (17th)"] = lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=False, random_state=s)
    z["MLP_ES_on (original, failed)"] = lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=True, random_state=s)
    return z

# ---- 데이터: 원고 설정 (KF-BOCPD onset vs random placebo) ----
store = load_store()
true_on, _, _ = load_onsets(False); _, _, ev_n = load_onsets(True)
parts = []
for ch in CHANNELS:
    T = onset_table(store, true_on, 1, channel=ch)
    if len(T) < 5: continue
    P = placebo_table(store, ch, T.ratio.to_numpy(), 5, 42, allowed=ev_n)
    T["label"] = 1; P["label"] = 0; parts += [T, P]
d = pd.concat(parts, ignore_index=True)
d["group"] = d.channel + "_" + d.segment.astype(str)
d = d.dropna(subset=FEATS).reset_index(drop=True)
y = d.label.to_numpy(int); g = d.group.to_numpy(); X = d[FEATS].to_numpy(float)
folds = assign_folds(g, y, 5, np.random.default_rng(SEED))
print(f"n={len(d)} (양성 {int(y.sum())}, 음성 {int((1-y).sum())}), 그룹 {len(set(g))}", flush=True)

rows = []
for mname, mk in zoo().items():
    t0 = time.time(); fold_shares, fold_auc = [], []
    for f in range(5):
        tr, te = folds != f, folds == f
        sc = StandardScaler().fit(X[tr])
        try:
            m = mk(); m.fit(sc.transform(X[tr]), y[tr])
            Xte = sc.transform(X[te]); base = auc_y(y[te], m.predict_proba(Xte)[:, 1])
        except Exception as e:
            print("  fold 실패:", mname, e); continue
        rng = np.random.default_rng(SEED + f); imp = np.zeros(3)
        for j in range(3):
            drops = []
            for _ in range(args.n_repeats):
                Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - auc_y(y[te], m.predict_proba(Xp)[:, 1]))
            imp[j] = max(np.mean(drops), 0.0)
        fold_auc.append(base)
        if imp.sum() > 0: fold_shares.append(imp / imp.sum())
    if not fold_shares:
        print(f"{mname}: 중요도 계산 불가 (모든 폴드 importance=0)"); continue
    sh = np.mean(fold_shares, axis=0); rank_level = int(1 + (sh > sh[0]).sum())
    rows.append(dict(model=mname, mean_fold_auc=np.mean(fold_auc), level=sh[0], first=sh[1], second=sh[2],
                     rank_of_level=rank_level, n_folds_used=len(fold_shares)))
    print(f"  {mname:30s} AUC={np.mean(fold_auc):.3f} level={sh[0]:.3f} first={sh[1]:.3f} second={sh[2]:.3f} "
          f"rank(level)={rank_level} ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows); res.to_csv(OUT / "rev7_attrib_sensitivity.csv", index=False)
print("\n[1] 전체 표"); print(res.round(3).to_string(index=False))

base10 = res[~res.model.str.startswith("MLP")]
for label, extra in [("10 tabular (permutation)", []),
                     ("+ MLP_ES_off (17th, 정상 학습)", ["MLP_ES_off (17th)"]),
                     ("+ MLP_ES_on (원 설정, 실패 모델)", ["MLP_ES_on (original, failed)"])]:
    sub = pd.concat([base10, res[res.model.isin(extra)]])
    print(f"\n[2] {label}: n={len(sub)} | level 1위 {int((sub.rank_of_level==1).sum())}, "
          f"2위 {int((sub.rank_of_level==2).sum())}, 3위 {int((sub.rank_of_level==3).sum())} | "
          f"level 점유율 중앙값 {sub.level.median():.3f}, 최대 {sub.level.max():.3f}")
