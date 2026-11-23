"""How the added plugins behave on our own recordings. Usage: evaluate_new.py <nova77_full> <out>"""
import csv, json, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from core.plugin import discover, run_one
from scripts.dataset import _load, koa_pd_nm
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
data, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
BY = {p.manifest.id: p for p in discover()[0]}
ids = ["pd_stgcn_sabo2022", "pd_stgcn_sabo2022_ap2d", "pd_stgcn_sabo2022_dt2d", "gait_events_zeni2008", "gait_gaitlab2020"]
rows = []
for v in koa_pd_nm(data, workers=2):
    r = {k: v[k] for k in ("subject", "group", "severity", "direction")}
    for i in ids:
        x = run_one(BY[i], v["prepared"]); r[i] = x.score if x.status == "ok" else np.nan
        if i == "gait_gaitlab2020" and x.status == "ok": r["cadence_gaitlab"] = x.detail["outputs"]["cadence"] * 120      # strides/s -> steps/min
    rows.append(r)
A = lambda i, rs=rows: np.array([r[i] for r in rs], float)
g = np.array([r["group"] for r in rows])
print(f"KOA-PD-NM videos {len(rows)} | scored:", {i: int(np.isfinite(A(i)).sum()) for i in ids})
def subj(i):
    by = defaultdict(list)
    for r in rows:
        if np.isfinite(r[i]): by[(r["subject"], r["group"])].append(r[i])
    return {k: float(np.mean(v)) for k, v in by.items()}
S = {i: subj(i) for i in ids}
def rank(a, b):
    k = [x for x in S[a] if x in S[b]]
    return spearmanr([S[a][x] for x in k], [S[b][x] for x in k]).statistic, len(k)
def auc(i, p, n):
    v = [(g_, s) for (u, g_), s in S[i].items() if g_ in (p, n)]
    return roc_auc_score([a == p for a, _ in v], [b for _, b in v])
print("subject-level agreement of the same study's models (Spearman):", {f"{a[-4:]} vs {b[-4:]}": tuple(round(x, 2) if i == 0 else x for i, x in enumerate(rank(a, b))) for a, b in (("pd_stgcn_sabo2022", "pd_stgcn_sabo2022_ap2d"), ("pd_stgcn_sabo2022", "pd_stgcn_sabo2022_dt2d"), ("pd_stgcn_sabo2022_ap2d", "pd_stgcn_sabo2022_dt2d"))})
for i in ids[:3]: print(f"   {i:26s} PD vs NM AUC {auc(i,'PD','NM'):.2f}  PD vs KOA {auc(i,'PD','KOA'):.2f}  mean NM/PD/KOA " + "/".join(f"{np.mean([s for (u, gg), s in S[i].items() if gg == x]):.2f}" for x in ("NM", "PD", "KOA")))
ok = np.isfinite(A("gait_events_zeni2008")) & np.isfinite(np.array([r.get("cadence_gaitlab", np.nan) for r in rows], float))
z, gl = A("gait_events_zeni2008")[ok], np.array([r.get("cadence_gaitlab", np.nan) for r in rows], float)[ok]
print(f"cadence, Zeni events vs mobile-gaitlab network (videos where both run, n={ok.sum()}): Spearman {spearmanr(z, gl).statistic:.2f}, mean Zeni {z.mean():.0f}, network {gl.mean():.0f} steps/min")
print("Zeni by group (steps/min):", {x: round(float(np.nanmean([s for (u, gg), s in S["gait_events_zeni2008"].items() if gg == x])), 0) for x in ("NM", "PD", "KOA")})
# Hip_OA front views: score and forget each recording (571 prepared recordings at once need more memory than this shared server has).
def _score(args):
    path, ids_ = args
    p = _load(path)
    return (p.axes.direction, [(lambda x: x.score if x.status == "ok" else np.nan)(run_one(BY[i], p)) for i in ids_]) if p.axes.direction == "AwayFromCamera" else (p.axes.direction, None)
meta = [m for m in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if m["dataset"] == "Hip_OA" and (data / "frame" / m["output"]).is_file()]
with ProcessPoolExecutor(2) as ex: res = list(ex.map(_score, [(data / "frame" / m["output"], ids[:3]) for m in meta], chunksize=4))
for j, i in enumerate(ids[:3]):
    h = defaultdict(list)
    for m, (d_, sc) in zip(meta, res):
        if sc is not None and np.isfinite(sc[j]): h[m["subject_id"], m["session"]].append(sc[j])
    v = [x for x in h.values() if len(x) >= 2]; d = np.mean([np.mean([abs(a - b) for q, a in enumerate(x) for b in x[q + 1:]]) for x in v])
    vv = [(np.mean(h[s, "M0"]), np.mean(h[s, "M6"])) for s in {k[0] for k in h} if h.get((s, "M0")) and h.get((s, "M6"))]
    print(f"   {i:26s} front view (Hip_OA, away): repeat walks in one session differ by {d:.2f} points (n={len(v)} sessions); before/after surgery {np.mean([x[0] for x in vv]):.2f} -> {np.mean([x[1] for x in vv]):.2f} (n={len(vv)})")
for i in ids[:3]:
    by = defaultdict(dict)
    for r in rows:
        if np.isfinite(r[i]): by[r["subject"]][r["direction"]] = r[i]
    d = [abs(x["left to right"] - x["right to left"]) for x in by.values() if len(x) == 2]
    print(f"   {i:26s} side view: left-to-right vs right-to-left differ by {np.mean(d):.2f} (n={len(d)})")
(out / "new_plugins.json").write_text(json.dumps(rows, default=float))
