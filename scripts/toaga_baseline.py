"""How much does the study's own model change when only the pose extractor changes? This is the yardstick for Nova-77.
Usage: toaga_baseline.py <toaga/x/PoseTracking> <stgcn repo> <out>"""
import json, sys, warnings
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scipy.stats import spearmanr
from scripts.study_code import StudyModel
from scripts.toaga import STUDY_RES, align_pair, at_30fps, centre, fill, index, read, smooth
from scripts.toaga_criteria import summarise
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
root, repo, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
idx = index(root)
print("walks", len(idx), "| engines per walk", dict(Counter(tuple(sorted(v)) for v in idx.values())), "| camera/direction", dict(Counter(k[1:3] for k in idx)))
lens, rng, res = [], [], {}
models = {e: StudyModel(repo, e) for e in ("alphapose", "detectron")}
full, common = {}, {}
for k, files in idx.items():
    tr = {}
    for eng, p in files.items():
        t, xy, conf = read(p); lens.append(len(t)); rng.append((np.nanmin(xy[..., 0]), np.nanmax(xy[..., 0]), np.nanmin(xy[..., 1]), np.nanmax(xy[..., 1])))
        z = at_30fps(t, fill(xy))[1]
        if len(z) >= 90 and np.isfinite(z).all(): tr[eng] = z
    score = lambda z: {m: sm.scores(centre(smooth(z)), np.ones(z.shape[:2]), STUDY_RES).tolist() for m, sm in models.items()}
    for eng, z in tr.items():
        for m, v in score(z).items(): res.setdefault(k, {})[f"{m}|{eng}"] = v              # whole track (walk-to-walk and camera comparisons)
    for e1 in ("alphapose", "detectron", "openpose"):                                       # same frames for a pair of extractors
        for e2 in ("alphapose", "detectron", "openpose"):
            if e1 < e2 and e1 in tr and e2 in tr:
                pos = align_pair(tr[e1], tr[e2])
                if pos:
                    i, j, n = pos; s1, s2 = score(tr[e1][i:i + n]), score(tr[e2][j:j + n])
                    for m in models: res[k][f"{m}|{e1}~{e2}"], res[k][f"{m}|{e2}~{e1}"] = s1[m], s2[m]
                    res[k][f"frames {e1}~{e2}"] = [n]
seen = defaultdict(list)
r = np.array(rng)
print(f"frames per walk: median {int(np.median(lens))} (min {min(lens)}, max {max(lens)}); pixel range x {r[:,0].min():.0f}..{r[:,1].max():.0f}, y {r[:,2].min():.0f}..{r[:,3].max():.0f}")
out.mkdir(parents=True, exist_ok=True)
(out / "toaga_scores.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in res.items()}))
S = lambda k, name: float(np.mean(res[k][name])) if name in res.get(k, {}) else np.nan
cl = lambda v: np.clip(np.round(v), 0, 2)

def pair(a, b, keys):
    """a, b: names of the two series; compares the same frames when the names carry '~'."""
    v = [(S(k, a), S(k, b), k[0]) for k in keys]; v = [x for x in v if np.isfinite(x[:2]).all()]
    return summarise([x[0] for x in v], [x[1] for x in v], [x[2] for x in v])
show = lambda n, d: print(f"  {n:58s} n={d['n']:3d}  mean |diff| {d['diff']:.2f}  bias {d['bias']:+.2f}  class changes {d['cls']:.0%}  rank corr {d['rho']:.2f}  | 95% bound: diff {d['diff_hi']:.2f}, class {d['cls_hi']:.0%}, rank {d['rho_lo']:.2f}")
front = [k for k in res if k[2] == "front"]; yard = {}
n_f = [res[k][f] [0] for k in res if k[2] == "front" for f in res[k] if f == "frames alphapose~detectron"]
print(f"common frames of AlphaPose and Detectron: median {np.median(n_f):.0f}, found for {len(n_f)} of {sum(1 for k in res if k[2]=='front')} front walks")
print("\nA. same model, different extractor on the SAME walk (front view)")
for m in ("alphapose", "detectron"):
    for e in ("alphapose", "detectron", "openpose"):
        if e != m:
            a_, b_ = f"{m}|{m}~{e}", f"{m}|{e}~{m}"
            d = pair(a_, b_, front); yard[f"{m} model: {m} -> {e}"] = d; show(f"{m} model: {m} track -> {e} track", d)
print("B. the study's two pipelines on the same walk"); show("alphapose pipeline vs detectron pipeline", pair("alphapose|alphapose~detectron", "detectron|detectron~alphapose", front))
print("C. same pipeline, other things that change")
for m in ("alphapose", "detectron"):
    n = f"{m}|{m}"
    fs = np.array([np.std(res[k][n]) for k in front if n in res[k]]); print(f"  {m}: spread between the study's 10 folds, SD {fs.mean():.2f}")
    top, bot = [k for k in front if k[1] == "top"], None
    v = np.array([(S(k, n), S((k[0], "bottom", k[2], k[3]), n)) for k in top]); v = v[np.isfinite(v).all(1)]
    print(f"  {m}: top camera vs bottom camera, same walk   n={len(v)}  mean |diff| {np.abs(v[:,0]-v[:,1]).mean():.2f}  bias {(v[:,1]-v[:,0]).mean():+.2f}  class changes {np.mean(cl(v[:,0])!=cl(v[:,1])):.0%}  rank corr {spearmanr(v[:,0],v[:,1]).statistic:.2f}")
    g = defaultdict(list)
    for k in front:
        if np.isfinite(S(k, n)): g[k[:3]].append(S(k, n))
    print(f"  {m}: same person, same camera, another walk   mean |diff| {np.mean([abs(a-b) for v_ in g.values() for i,a in enumerate(v_) for b in v_[i+1:]]):.2f}")
(out / "toaga_yardstick.json").write_text(json.dumps(yard, indent=1))
