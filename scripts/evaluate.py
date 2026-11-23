"""Run every plugin on all KOA-PD-NM videos and measure what survives. Usage: evaluate.py <nova77_full> <out>"""
import csv, json, sys, warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from core.plugin import discover, run_one
from scripts.dataset import koa_pd_nm, subject_mean

warnings.filterwarnings("ignore")
data, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
plugins, failures = discover()
print("plugins:", [p.manifest.id for p in plugins], "| failed to load:", failures)
videos = koa_pd_nm(data, workers=2)
rows = []
for v in videos:
    p = v["prepared"]
    r = {k: v[k] for k in ("file", "subject", "group", "severity", "age", "sex")} | {"view": p.axes.view, "lr_ok": p.axes.left_right_ok, "walk_frames": len(p.walk),
         "filled": round(float((p.walk.status == 2).mean()), 3) if len(p.walk) else 0}
    for pl in plugins:
        res = run_one(pl, p)
        r |= {pl.manifest.id + "_status": res.status, pl.manifest.id + "_score": res.score, pl.manifest.id + "_why": "; ".join(res.reasons[:1])}
    rows.append(r)
with (out / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for pl in plugins:
    i = pl.manifest.id
    print(f"\n[{i}] " + ", ".join(f"{g}: {dict(Counter(r[i + '_status'] for r in rows if r['group'] == g))}" for g in ("NM", "PD", "KOA")))
    print("   not run because:", dict(Counter(r[i + "_why"].split(":")[0] for r in rows if r[i + "_status"] != "ok")))
print("\nleft/right inconsistent videos:", sum(not r["lr_ok"] and r["view"] == "Side" for r in rows), "| view:", dict(Counter(r["view"] for r in rows)))

# ---- pretrained parkinsonism plugin: zero-shot on Nova-77 input
pl = next(p for p in plugins if p.manifest.id == "pd_stgcn_sabo2022")
P = sys.modules[type(pl).__module__]
import torch

def score_all(fn):
    """Subject means of fn(video) over videos the plugin accepts."""
    acc = []
    for v, r in zip(videos, rows):
        if r["pd_stgcn_sabo2022_status"] == "ok":
            s = fn(v["prepared"])
            if s is not None:
                acc.append({k: v[k] for k in ("subject", "group", "severity", "age")} | {"s": s})
    return subject_mean(acc)

def net(x):
    with torch.no_grad():
        return float(pl.model(torch.from_numpy(x[None].astype(np.float32).copy())).item())

from core import jointsets
from core.view import Camera, build_world, look

def kin(p, tilt=P.TILT_DEG):
    w = p.world; n, c = jointsets.select(w.body, "kinect13", w.xyz)
    return n, look(c, Camera(azimuth=0, elevation=tilt))

def same_length(p, n_frames):
    n, k = kin(p)
    if len(k) < n_frames:
        return None
    b = (len(k) - n_frames) // 2
    return net(P.study_preprocess(k[b:b + n_frames], n))

def no_view_rotation(p):
    n, c = jointsets.select(p.walk, "kinect13")
    c = c * np.array([1, -1, 1])                       # camera axes as they are (only y flipped up)
    return net(P.study_preprocess(c, n))

def mirrored(p):
    n, k = kin(p)
    k = k.copy(); k[..., 0] *= -1
    order = [0] + [i + 1 if i % 2 else i - 1 for i in range(1, 13)]
    return net(P.study_preprocess(k[:, order], n))

variants = {"as designed": lambda p: net(pl.adapt(p)), "mean with mirror image": lambda p: (net(pl.adapt(p)) + mirrored(p)) / 2,
            "equal length, 60 frames each": lambda p: same_length(p, 60), "equal length, 90 frames each": lambda p: same_length(p, 90),
            "no camera tilt": lambda p: (lambda t: net(P.study_preprocess(t[1], t[0])))(kin(p, 0.0)), "no view rotation (raw camera axes)": no_view_rotation,
            "CONTROL clip length": lambda p: -float(len(p.walk)), "CONTROL walking speed": lambda p: -p.axes.distance_m / len(p.walk)}
SEV = {"": 0, "ML": 1, "MD": 2, "SV": 3}
summary = {}
print(f"\n{'variant':36s} n(NM/PD/KOA)  AUC PD-NM [95% boot]   PD-KOA  KOA-NM  rho(severity in PD)  mean NM/PD/KOA")
for name, fn in variants.items():
    S = score_all(fn); g = np.array([s["group"] for s in S]); x = np.array([s["s"] for s in S])
    if name == "CONTROL clip length": x = -x                       # slower walk = longer clip = more parkinsonian
    def auc(a, b):
        m = (g == a) | (g == b)
        return roc_auc_score(g[m] == a, x[m])
    m = np.flatnonzero((g == "PD") | (g == "NM")); rng = np.random.default_rng(0); bs = []
    for _ in range(1000):
        i = rng.choice(m, len(m))
        if len(set(g[i])) == 2: bs.append(roc_auc_score(g[i] == "PD", x[i]))
    pd_ = [s for s in S if s["group"] == "PD"]
    rho = spearmanr([SEV[s["severity"]] for s in pd_], x[g == "PD"]).statistic if len(pd_) > 4 else float("nan")
    summary[name] = {"n": {k: int((g == k).sum()) for k in ("NM", "PD", "KOA")}, "auc_pd_nm": auc("PD", "NM"), "ci": list(np.percentile(bs, [2.5, 97.5])),
                     "auc_pd_koa": auc("PD", "KOA"), "auc_koa_nm": auc("KOA", "NM"), "rho": float(rho), "mean": {k: float(x[g == k].mean()) for k in ("NM", "PD", "KOA")}}
    s = summary[name]
    print(f"{name:36s} {s['n']['NM']:3d}/{s['n']['PD']:2d}/{s['n']['KOA']:2d}     {s['auc_pd_nm']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]      {s['auc_pd_koa']:.2f}    {s['auc_koa_nm']:.2f}        {s['rho']:+.2f}           "
          + "/".join(f"{s['mean'][k]:.2f}" for k in ("NM", "PD", "KOA")))
S = score_all(variants["as designed"])
old = [s for s in S if 45 <= s["age"] <= 70 and s["group"] in ("NM", "PD")]
print("age 45-70 only, PD-NM AUC:", round(roc_auc_score([s["group"] == "PD" for s in old], [s["s"] for s in old]), 2), dict(Counter(s["group"] for s in old)))
print("score range:", round(min(s["s"] for s in S), 2), "to", round(max(s["s"] for s in S), 2), "| predicted class counts:",
      {g: dict(sorted(Counter(int(np.clip(round(s["s"]), 0, 2)) for s in S if s["group"] == g).items())) for g in ("NM", "PD", "KOA")})
(out / "pd_stgcn_zero_shot.json").write_text(json.dumps(summary, indent=1))

# ---- does the pretrained score say anything beyond walking speed? (subject level, repeated 5-fold CV)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

speed = {s["subject"]: s["s"] for s in score_all(lambda p: p.axes.distance_m / len(p.walk))}
S = score_all(variants["as designed"])
print("\nbeyond speed:")
for a, b in (("PD", "KOA"), ("KOA", "NM")):
    T = [s for s in S if s["group"] in (a, b)]
    y = np.array([s["group"] == a for s in T], int)
    X = np.array([[speed[s["subject"]], s["s"]] for s in T])
    res = {}
    for name, cols in (("speed", [0]), ("score", [1]), ("speed+score", [0, 1])):
        aucs = []
        for seed in range(20):
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                p[te] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
            aucs.append(roc_auc_score(y, p))
        res[name] = round(float(np.mean(aucs)), 3)
    print(f"  {a} vs {b}: {res} | correlation of score with speed: {spearmanr(X[:, 0], X[:, 1]).statistic:+.2f}")
