"""Train and validate the knee OA plugin. Subject-level nested cross-validation; feature set and C are chosen inside the
training folds only. Usage: train.py <nova77_full folder> <out folder>"""
import json, sys, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.dataset import koa_pd_nm, subject_mean
from core.plugin import discover

warnings.filterwarnings("ignore")
NEG = ("NM", "PD")
GRID = [(fs, c) for fs in ("knee", "lower_limb", "all") for c in (0.1, 0.5, 2.0)]


def lr(c):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def matrix(subj, cols):
    return np.array([[s[k] for k in cols] for s in subj], float)


def pick(subj, y, strat, feats, seed):
    """Feature set and C with the best inner-CV AUC, averaged over 3 shuffles so one split does not decide."""
    best, score = None, -1
    for fs, c in GRID:
        x, aucs = matrix(subj, feats.FEATURE_SETS[fs]), []
        for k in range(3):
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(4, shuffle=True, random_state=1000 * k + seed).split(x, strat):
                p[te] = lr(c).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
            aucs.append(roc_auc_score(y, p))
        if np.mean(aucs) > score + 1e-9:
            best, score = (fs, c), np.mean(aucs)
    return best


def nested(subj, y, strat, feats, seed):
    p, chosen = np.zeros(len(y)), []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), strat):
        s = [subj[i] for i in tr]
        fs, c = pick(s, y[tr], strat[tr], feats, seed)
        chosen.append(f"{fs}/C={c}")
        cols = feats.FEATURE_SETS[fs]
        p[te] = lr(c).fit(matrix(s, cols), y[tr]).predict_proba(matrix([subj[i] for i in te], cols))[:, 1]
    return p, chosen


def simple_oof(x, y, strat, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, strat):
        p[te] = lr(1.0).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return p


def ci(v):  # mean, 2.5th and 97.5th percentile (over CV seeds, except for the bootstrap entry)
    return [round(float(np.mean(v)), 3), round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main(data, out):
    plugins, failures = discover(HERE.parent)
    plug = next(p for p in plugins if p.manifest.id == "koa_gait_baseline")
    feats = sys.modules["gait_plugin_koa_features"] if "gait_plugin_koa_features" in sys.modules else None
    if feats is None:
        feats = sys.modules[type(plug).__module__].features
    rows, excluded = [], {}
    for v in koa_pd_nm(data):
        reasons = plug.check(v["prepared"])
        if reasons:
            excluded.setdefault(v["group"], []).append((v["file"], reasons))
            continue
        f = plug.adapt(v["prepared"])
        rows.append({**{k: v[k] for k in ("subject", "group", "severity", "age", "sex")}, "clip_frames": float(len(v["prepared"].walk)),
                     "observed": float((v["prepared"].walk.status == 1).mean()), "conf": float(np.nanmean(v["prepared"].walk.conf)), **f})
    subj = subject_mean(rows)
    y = np.array([s["group"] == "KOA" for s in subj], int)
    strat = np.array([s["group"] for s in subj])
    age = np.array([s["age"] for s in subj])
    rep = {"videos_used": len(rows), "videos_excluded": {g: len(v) for g, v in excluded.items()},
           "subjects": {g: int((strat == g).sum()) for g in ("KOA", "NM", "PD")},
           "mean_age": {g: round(float(age[strat == g].mean()), 1) for g in ("KOA", "NM", "PD")}}
    runs, seeds = {k: [] for k in ("auc", "bal_acc", "sens", "spec_NM", "fp_PD", "auc_vs_NM", "auc_vs_PD", "auc_age45_70", "auc_vs_NM_age45_70")}, range(10)
    chosen, psum = [], np.zeros(len(y))
    mid = (age >= 45) & (age <= 70)
    for sd in seeds:
        p, ch = nested(subj, y, strat, feats, sd)
        psum += p; chosen += ch
        runs["auc"].append(roc_auc_score(y, p)); runs["bal_acc"].append(balanced_accuracy_score(y, p >= .5))
        runs["sens"].append((p[y == 1] >= .5).mean()); runs["spec_NM"].append((p[strat == "NM"] < .5).mean())
        runs["fp_PD"].append((p[strat == "PD"] >= .5).mean())
        for k, m in (("auc_vs_NM", strat != "PD"), ("auc_vs_PD", strat != "NM"), ("auc_age45_70", mid), ("auc_vs_NM_age45_70", mid & (strat != "PD"))):
            runs[k].append(roc_auc_score(y[m], p[m]))
    rep["nested_cv"] = {k: ci(v) for k, v in runs.items()}
    rep["n_age45_70"] = {g: int((mid & (strat == g)).sum()) for g in ("KOA", "NM", "PD")}
    rep["bootstrap_auc_95ci"] = ci([roc_auc_score(y[i], psum[i]) for i in (np.random.default_rng(b).integers(0, len(y), len(y)) for b in range(1000)) if 0 < y[i].sum() < len(i)])
    ctl = {"age_only": ["age"], "clip_only": ["clip_frames", "observed", "conf"], "speed_only": ["speed_leg_s"]}
    rep["controls_auc"] = {k: ci([roc_auc_score(y, simple_oof(matrix(subj, c), y, strat, sd)) for sd in seeds]) for k, c in ctl.items()}
    rep["controls_auc"]["age_only_vs_NM"] = ci([roc_auc_score(y[strat != "PD"], simple_oof(matrix([s for s in subj if s["group"] != "PD"], ["age"]), y[strat != "PD"], strat[strat != "PD"], sd)) for sd in seeds])
    from collections import Counter
    rep["chosen_in_outer_folds"] = dict(Counter(chosen).most_common())
    fs, c = pick(subj, y, strat, feats, 0)
    cols = feats.FEATURE_SETS[fs]
    m = lr(c).fit(matrix(subj, cols), y)
    (HERE / "model.json").write_text(json.dumps({
        "features": cols, "median": m[0].statistics_.tolist(), "mean": m[1].mean_.tolist(), "scale": m[1].scale_.tolist(),
        "coef": m[2].coef_[0].tolist(), "intercept": float(m[2].intercept_[0]), "positive": 0.6, "negative": 0.4,
        "feature_set": fs, "C": c, "negatives": NEG, "validation": rep}, indent=1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "koa_validation.json").write_text(json.dumps({**rep, "oof": [{"subject": s["subject"], "group": s["group"], "severity": s["severity"], "age": s["age"], "p": round(float(q), 4)} for s, q in zip(subj, psum / len(seeds))],
                                                         "excluded": {g: v for g, v in excluded.items()}}, indent=1))
    print(json.dumps(rep, indent=1)); print("shipped:", fs, "C =", c)
    print("coef:", sorted([(k, round(v, 2)) for k, v in zip(cols, m[2].coef_[0])], key=lambda t: -abs(t[1]))[:8])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
