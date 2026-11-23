"""Gait features from 15 joints in walking axes (forward, left, up). Used only by this plugin.

Events: Zeni et al. 2008 (foot furthest ahead of / behind the pelvis). Nova-77 has no heel, so the ankle is used for
heel strike and the toe base for toe off. Lengths are divided by leg length; left/right pairs become mean + asymmetry.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

GROUPS = {
    "knee": ["knee_rom_mean", "knee_rom_asym", "knee_peak_flex_mean", "knee_peak_flex_asym", "stance_pct_mean", "stance_pct_asym"],
    "lower": ["hip_rom_mean", "hip_rom_asym", "ankle_rom_mean", "ankle_rom_asym", "step_time_asym"],
    "pace": ["stride_time_s", "stride_time_cv", "cadence", "speed_leg_s", "stride_len_leg"],
    "arm": ["arm_swing_mean", "arm_swing_asym"],
}
FEATURE_SETS = {"knee": GROUPS["knee"], "lower_limb": GROUPS["knee"] + GROUPS["lower"],
                "all": GROUPS["knee"] + GROUPS["lower"] + GROUPS["pace"] + GROUPS["arm"]}


def _angle(a, b):
    c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _asym(l, r):
    m = (l + r) / 2
    return float(abs(l - r) / m * 100) if np.isfinite(m) and m > 0 else np.nan


def _pair(out, name, l, r):
    v = [x for x in (l, r) if np.isfinite(x)]
    out[name + "_mean"] = float(np.mean(v)) if v else np.nan
    out[name + "_asym"] = _asym(l, r)


def compute(names: list, walk: np.ndarray, fps: float) -> dict:
    """walk: (T, K, 3) in (forward, left, up)."""
    win = max(5, int(round(0.25 * fps)) | 1)
    x = savgol_filter(walk, win, 2, axis=0) if len(walk) > win else walk
    j = {n: x[:, i] for i, n in enumerate(names)}
    leg = np.median([np.linalg.norm(j[s + "Hip"] - j[s + "Knee"], axis=1) + np.linalg.norm(j[s + "Knee"] - j[s + "Ankle"], axis=1) for s in "LR"])
    f, ev = {}, {}
    for s in "LR":
        ahead = (j[s + "Ankle"] - j["Pelvis"])[:, 0]
        behind = -(j[s + "Toe"] - j["Pelvis"])[:, 0]
        prom = 0.25 * (np.percentile(ahead, 95) - np.percentile(ahead, 5))
        ev[s] = (find_peaks(ahead, distance=int(0.6 * fps), prominence=prom)[0], find_peaks(behind, distance=int(0.6 * fps), prominence=prom)[0])
    f["strides_L"], f["strides_R"] = (max(len(ev[s][0]) - 1, 0) for s in "LR")

    def per_stride(sig, hs, fn=np.ptp):
        v = [fn(sig[a:b + 1]) for a, b in zip(hs[:-1], hs[1:])]
        return float(np.mean(v)) if v else np.nan

    strides, stance, rom = [], {}, {}
    for s in "LR":
        hs, to = ev[s]
        strides += [(b - a) / fps for a, b in zip(hs[:-1], hs[1:])]
        st = [(t[0] - a) / (b - a) * 100 for a, b in zip(hs[:-1], hs[1:]) if len(t := to[(to > a) & (to < b)])]
        stance[s] = float(np.mean(st)) if st else np.nan
        thigh, shank, foot = j[s + "Knee"] - j[s + "Hip"], j[s + "Ankle"] - j[s + "Knee"], j[s + "Toe"] - j[s + "Ankle"]
        knee = _angle(thigh, shank)
        rom[s] = {"knee_rom": per_stride(knee, hs), "knee_peak_flex": per_stride(knee, hs, np.max),
                  "hip_rom": per_stride(np.degrees(np.arctan2(thigh[:, 0], -thigh[:, 2])), hs),
                  "ankle_rom": per_stride(_angle(-shank, foot), hs)}
    _pair(f, "stance_pct", stance["L"], stance["R"])
    for k in ("knee_rom", "knee_peak_flex", "hip_rom", "ankle_rom"):
        _pair(f, k, rom["L"][k], rom["R"][k])
    f["stride_time_s"] = float(np.mean(strides)) if strides else np.nan
    f["stride_time_cv"] = float(np.std(strides) / np.mean(strides) * 100) if len(strides) >= 3 else np.nan
    hits = sorted([(i, s) for s in "LR" for i in ev[s][0]])
    t = np.array([i for i, _ in hits]) / fps
    f["cadence"] = float((len(t) - 1) / (t[-1] - t[0]) * 60) if len(t) >= 3 else np.nan
    to_side = {s: [b - a for (a, sa), (b, sb) in zip(zip(t, [h[1] for h in hits]), list(zip(t, [h[1] for h in hits]))[1:]) if sa != sb and sb == s] for s in "LR"}
    f["step_time_asym"] = _asym(np.mean(to_side["L"]), np.mean(to_side["R"])) if to_side["L"] and to_side["R"] else np.nan
    dur = (len(x) - 1) / fps
    f["speed_leg_s"] = float((j["Pelvis"][-1, 0] - j["Pelvis"][0, 0]) / dur / leg)
    f["stride_len_leg"] = f["speed_leg_s"] * f["stride_time_s"]
    ref = max((ev[s][0] for s in "LR"), key=len)
    _pair(f, "arm_swing", *[per_stride((j[s + "Wrist"] - j[s + "Shoulder"])[:, 0], ref) / leg for s in "LR"])
    return f
