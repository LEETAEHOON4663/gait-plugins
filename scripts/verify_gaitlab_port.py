"""The authors published the predictions for their own demo video in demo.ipynb. Does the rebuilt model give the same numbers?
Usage: verify_gaitlab_port.py <mobile-gaitlab/demo> <npz to write>"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("gl", Path(__file__).resolve().parent.parent / "plugins/gait_gaitlab2020/model.py"); gl = importlib.util.module_from_spec(spec); spec.loader.exec_module(gl)
demo, npz = Path(sys.argv[1]), Path(sys.argv[2])
gl.convert(demo / "models", npz); net = gl.GaitLab(npz)
want = {"GDI": 84.80461127527384, "gmfcs": 1.3795744707902589, "speed": 1.039787208230336, "cadence": 1.073140446145435,
        "SEMLS_dev_residual": -0.4406224325202497, "KneeFlex_maxExtension_L": 12.886017305048433, "KneeFlex_maxExtension_R": 7.5525704326495005}
rows = np.full((300, 75), np.nan)                      # convert_json2csv of the demo: frames 1..299, first person
for fr in range(1, 300):
    f = demo / "out/keypoints" / f"input_{fr:012d}_keypoints.json"
    if not f.is_file(): break
    for p in json.load(open(f))["people"]:
        rows[fr - 1] = p["pose_keypoints_2d"]; break
last = max(i for i in range(300) if not np.isnan(rows[i]).all()); rows = rows[:last + 1]
xy = np.stack([rows[:, 0::3], rows[:, 1::3]], -1)
got = gl.predict_all(net, xy)
worst = 0.0
for k, w in want.items():
    d = abs(got[k] - w); worst = max(worst, d / max(abs(w), 1e-9)); print(f"{k:26s} authors {w:10.5f}  port {got[k]:10.5f}  diff {d:.1e}")
print(f"frames used {len(xy)}, largest relative difference {worst:.1e}"); sys.exit(worst > 1e-3)
