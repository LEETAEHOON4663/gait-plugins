"""Does the published gait model give the same numbers as measuring the skeleton directly? Usage: evaluate_gaitlab.py <nova77_full> <out>
Cadence is the fair test: it can be counted from the skeleton by a separate algorithm (foot events), so the two must agree if
nothing is lost between the study's 2D input and our 3D skeleton."""
import json, sys, warnings
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.stats import pearsonr, spearmanr
from core import jointsets
from core.common import in_walk_frame
from core.plugin import discover
from scripts.dataset import koa_pd_nm
warnings.filterwarnings("ignore")
data, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
plugins, failures = discover()
pl = next(p for p in plugins if p.manifest.id == "gait_gaitlab2020"); from shared import gait_features as koa
from collections import Counter
rows, why = [], Counter()
for v in koa_pd_nm(data, workers=2):
    p = v["prepared"]; reasons = pl.check(p)
    if reasons:
        why[reasons[0].split(":")[0]] += 1; continue
    res = pl.predict(pl.adapt(p))["detail"]["outputs"]
    n, cam = jointsets.select(p.walk, "lower_limb_arms"); f = koa.compute(n, in_walk_frame(cam, p.axes), p.raw.fps)
    rows.append({**{k: v[k] for k in ("subject", "group", "severity")}, **res, "m_stride_time": f["stride_time_s"], "m_cadence_stride_s": 1 / f["stride_time_s"] if f["stride_time_s"] else np.nan,
                 "m_cadence_steps_s": f["cadence"] / 60 if f["cadence"] else np.nan, "m_speed": p.axes.distance_m / len(p.walk) * p.raw.fps, "clip_s": len(p.walk) / p.raw.fps})
print("videos with a result:", len(rows), "| not run:", dict(why), "| by group:", dict(Counter(r["group"] for r in rows)))
a = lambda k: np.array([r[k] for r in rows], float)
def line(name, pred, meas):
    ok = np.isfinite(pred) & np.isfinite(meas); pr, me = pred[ok], meas[ok]
    return f"{name:34s} n={ok.sum():3d}  Pearson {pearsonr(pr, me)[0]:+.2f}  Spearman {spearmanr(pr, me).statistic:+.2f}  mean |diff| {np.abs(pr - me).mean():.3f}  (constant guess would be off by {np.abs(me - me.mean()).mean():.3f})  bias {np.mean(pr - me):+.3f}"
print("\ncadence: predicted by the published model vs counted from the 3D skeleton (video level)")
print("  if the model outputs strides/s:", line("", a("cadence"), a("m_cadence_stride_s")).strip())
print("  if the model outputs steps/s:  ", line("", a("cadence"), a("m_cadence_steps_s")).strip())
by = defaultdict(list)
for r in rows: by[r["subject"]].append(r)
S = [{k: (np.nanmean([x[k] for x in v]) if not isinstance(v[0][k], str) else v[0][k]) for k in v[0]} for v in by.values()]
sa = lambda k: np.array([s[k] for s in S], float)
print("\nsubject level:"); print("  cadence (strides/s):", line("", sa("cadence"), sa("m_cadence_stride_s")).strip())
print("  walking speed (m/s, model-estimated path vs network):", line("", sa("speed"), sa("m_speed")).strip())
for g in ("NM", "PD", "KOA"):
    x = [s for s in S if s["group"] == g]
    if len(x) >= 3: print(f"  {g:4s} n={len(x):2d} cadence predicted {np.mean([s['cadence'] for s in x]):.2f} counted {np.mean([s['m_cadence_stride_s'] for s in x]):.2f} strides/s | GDI {np.mean([s['GDI'] for s in x]):.1f} (sd {np.std([s['GDI'] for s in x]):.1f}) | speed predicted {np.mean([s['speed'] for s in x]):.2f} measured {np.mean([s['m_speed'] for s in x]):.2f}")
rep = defaultdict(dict)
for r in rows: rep[r["subject"]].setdefault("v", []).append((r["cadence"], r["GDI"]))
pairs = [v["v"] for v in rep.values() if len(v["v"]) == 2]
print(f"\nrepeatability, two recordings of the same person (both directions): cadence mean |diff| {np.mean([abs(p[0][0] - p[1][0]) for p in pairs]):.3f}, GDI mean |diff| {np.mean([abs(p[0][1] - p[1][1]) for p in pairs]):.1f} points, n={len(pairs)}")
(out / "gaitlab.json").write_text(json.dumps({"rows": rows}, default=float))
