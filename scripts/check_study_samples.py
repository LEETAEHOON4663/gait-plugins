"""Do the study's own 2D sample walks score like their labels when run through study_code.py? Usage: check_study_samples.py <repo>"""
import csv, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scripts.study_code import JOINTS, StudyModel
repo = sys.argv[1]
for eng in ("alphapose", "detectron"):
    sm = StudyModel(repo, eng); print(eng, "folds", len(sm.models))
    for f in sorted(glob.glob(f"{repo}/sample_data/data/tri_{eng}/*/*.csv")):
        rows = list(csv.DictReader(open(f)))
        num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan
        a = np.array([[[num(r, j + s) for s in ("_x", "_y", "_conf")] for j in JOINTS] for r in rows])
        m = np.nanmean(a, 0)                                       # missing_joint_val: mean
        a = np.where(np.isnan(a), m[None], a)
        hip = (a[:, 7, :2] + a[:, 8, :2]) / 2
        s = sm.scores(a[..., :2], a[..., 2])
        print(f"  {Path(f).name[:40]:40s} T={len(rows):3d} label={rows[0]['UPDRS_gait']:>4s} score mean {s.mean():.2f} (folds {s.min():.2f}..{s.max():.2f}) | hip centre at {hip.mean(0).round(1).tolist()}")
