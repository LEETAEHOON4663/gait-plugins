"""Does our step-timing rule give the cadence that TOAGA's authors measured? Their Xsens motion-capture data (11 people) is used with the
same rule as plugins/gait_events_zeni2008. Published for comparison (TOAGA paper, Table 2, Xsens): cadence front 105.2 (9.1) / back
105.0 (10.3) steps/min, step time 0.55 (0.05) / 0.58 (0.06) s. Usage: validate_zeni_on_toaga.py <toaga/x/Xsens/Xsens>"""
import csv, glob, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.signal import savgol_filter
from shared import gait_features as gf
warnings.filterwarnings("ignore")
FPS = 100.0
MAP = {"Pelvis": "Pelvis", "LHip": "LULeg", "LKnee": "LLLeg", "LAnkle": "LFoot", "LToe": "LToe", "RHip": "RULeg", "RKnee": "RLLeg", "RAnkle": "RFoot", "RToe": "RToe"}


def segments(x, min_s=4.0):
    """Straight walking along the X axis: (start, end, sign) with a steady speed above 0.4 m/s."""
    v = np.gradient(savgol_filter(x, 101, 2)) * FPS
    ok = np.abs(v) > 0.4
    s = np.sign(v); out = []; i = 0
    while i < len(v):
        if ok[i]:
            j = i
            while j + 1 < len(v) and ok[j + 1] and s[j + 1] == s[i]: j += 1
            if j - i >= min_s * FPS: out.append((i, j + 1, int(s[i])))
            i = j + 1
        else: i += 1
    return out


rows = []
for f in sorted(glob.glob(sys.argv[1] + "/OAW*.csv")):
    r = list(csv.reader(open(f))); hdr = r[0]; a = {h: i for i, h in enumerate(hdr)}
    P = lambda n: np.array([[float(x[a[n + c]] or 0) for c in "XYZ"] for x in r[3:]])
    J = {k: P(v) for k, v in MAP.items()}
    pelvis = J["Pelvis"]; seg = segments(pelvis[:, 0])
    cad, st, stance, asym, n = [], [], [], [], 0
    for s, e, sign in seg:
        fwd, left, up = np.array([sign, 0, 0.0]), np.array([0, sign, 0.0]), np.array([0, 0, 1.0])
        names = ["Pelvis", "LHip", "LKnee", "LAnkle", "LToe", "RHip", "RKnee", "RAnkle", "RToe", "Neck", "LShoulder", "RShoulder", "LWrist", "RWrist"]
        rot = np.stack([fwd, left, up], 1)
        w = np.stack([(J[k][s:e] if k in J else J["Pelvis"][s:e]) @ rot for k in names], 1)          # (T, K, 3) in (forward, left, up)
        g = gf.compute(names, w, FPS)
        if g["strides_L"] >= 3 and g["strides_R"] >= 3:
            cad.append(g["cadence"]); st.append(g["stride_time_s"] / 2); stance.append(g["stance_pct_mean"]); n += 1
    if cad:
        rows.append((Path(f).stem, n, np.mean(cad), np.mean(st), np.nanmean(stance)))
print("participants with usable straight walks:", len(rows), "| segments:", sum(x[1] for x in rows))
c, s = np.array([x[2] for x in rows]), np.array([x[3] for x in rows])
print(f"cadence from our rule on their sensor data: mean {c.mean():.1f} (SD {c.std(ddof=1):.1f}) steps/min | published Xsens 105.2 (9.1) front, 105.0 (10.3) back")
print(f"step time: mean {s.mean():.3f} (SD {s.std(ddof=1):.3f}) s | published Xsens 0.55 (0.05) front, 0.58 (0.06) back")
for x in rows: print(f"   {x[0]}: {x[1]} segments, cadence {x[2]:.1f}, step time {x[3]:.2f} s, stance {x[4]:.0f}%")
