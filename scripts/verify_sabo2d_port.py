"""Port of the 2D models vs the study's ORIGINAL code, on the study's own sample walks, for every fold. Usage: verify_sabo2d_port.py <repo>"""
import csv, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scripts.study_code import JOINTS, StudyModel
from shared import stgcn
repo = Path(sys.argv[1]); worst = 0.0
for eng, folder in (("alphapose", "alphapose_models_to_share/v0/13/0.2"), ("detectron", "detectron_models_to_share/v2/13/0.2")):
    orig = StudyModel(repo, eng)
    port = [stgcn.load(f) for f in sorted((repo / "sample_data/model_zoo" / folder).glob("*/checkpoint_final.pt"), key=lambda p: int(p.parent.name))]
    n = 0
    for f in sorted(glob.glob(str(repo / f"sample_data/data/tri_{eng}/*/*.csv"))):
        rows = list(csv.DictReader(open(f))); num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan
        a = np.array([[[num(r, j + s) for s in ("_x", "_y", "_conf")] for j in JOINTS] for r in rows]); a = np.where(np.isnan(a), np.nanmean(a, 0)[None], a)
        so = orig.scores(a[..., :2], a[..., 2])
        x = torch.from_numpy(stgcn.study_preprocess_2d(a[..., :2], conf=a[..., 2])[None].copy())
        with torch.no_grad(): sp = np.array([float(m(x).item()) for m in port])
        d = np.abs(so - sp).max(); worst = max(worst, d); n += 1
    print(f"{eng}: {len(port)} folds x {n} sample walks, largest difference {worst:.1e}")
sys.exit(worst > 1e-5)
