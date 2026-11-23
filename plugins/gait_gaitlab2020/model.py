"""mobile-gaitlab (Kidzinski et al., Nature Communications 2020) rebuilt in torch + numpy from the published demo code.

`convert(repo_models, out)` reads the Keras 2.3 HDF5 weights and the OLS correction coefficients once (no Keras, no statsmodels, and no
code from the .pb pickle files is executed) and writes a single .npz. Everything after that needs only numpy and torch.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# mean/scale used by the authors' demo to undo the target scaling
MAPS = {"KneeFlex_maxExtension": (-29.4408212510502, 114.8431545843835), "GDI": (36.314492983907, 77.03271217530302), "gmfcs": (1, 3),
        "speed": (0.0718863507111867, 1.5259117583433834), "cadence": (0.222, 1.71556665023985), "SEMLS_dev_residual": (-0.8205001909638112, 3.309054961371647)}
WINDOW, STEP = 124, 31


class _Safe(pickle.Unpickler):
    """Reads the OLS results without importing or running statsmodels: only numpy arrays are rebuilt, other objects become plain holders."""

    def find_class(self, module, name):
        if (module, name) in {("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar")}:
            return getattr(__import__(module, fromlist=[name]), name)

        class Holder:
            def __setstate__(self, state):
                self.state = state

            def __reduce__(self):  # pragma: no cover
                raise pickle.PicklingError("holder")
        return Holder


def _params(path):
    obj = _Safe(io.BytesIO(Path(path).read_bytes())).load()
    stack = [obj]
    while stack:
        o = stack.pop()
        st = getattr(o, "state", o)
        if isinstance(st, dict):
            if isinstance(st.get("params"), np.ndarray):
                return np.asarray(st["params"], float)
            stack += list(st.values())
    raise ValueError(f"no OLS params in {path}")


def convert(folder, out):
    import h5py
    data = {}
    for name in MAPS:
        f = h5py.File(Path(folder) / f"{name}_best.pb", "r")["model_weights"]
        for k in f:
            for w in f[k].attrs.get("weight_names", []):
                w = w.decode() if isinstance(w, bytes) else w
                data[f"{name}/{w}"] = np.asarray(f[k][w])
        data[f"{name}/ols"] = _params(Path(folder) / f"{name}_correction.pb")
    np.savez_compressed(out, **data)


class CNN(nn.Module):
    def __init__(self, w, prefix):
        super().__init__()
        t = lambda n: torch.tensor(w[f"{prefix}/{n}"], dtype=torch.float32)
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, 7):
            k, b = t(f"conv1d_{i}/kernel:0"), t(f"conv1d_{i}/bias:0")                         # (8, in, out)
            c = nn.Conv1d(k.shape[1], k.shape[2], k.shape[0])
            c.weight.data, c.bias.data = k.permute(2, 1, 0).contiguous(), b
            self.conv.append(c)
            n = nn.BatchNorm1d(32, eps=1e-3, momentum=0.01)
            n.weight.data, n.bias.data = t(f"batch_normalization_{i}/gamma:0"), t(f"batch_normalization_{i}/beta:0")
            n.running_mean, n.running_var = t(f"batch_normalization_{i}/moving_mean:0"), t(f"batch_normalization_{i}/moving_variance:0")
            self.bn.append(n)
        self.dense = [(t(f"dense_{i}/kernel:0"), t(f"dense_{i}/bias:0")) for i in (1, 2)]
        self.eval()

    def forward(self, x):                                                # (N, T, C)
        x = x.permute(0, 2, 1)
        for i, (c, n) in enumerate(zip(self.conv, self.bn)):
            x = F.pad(x, (3, 4))                                         # Keras "same" for kernel 8: 3 left, 4 right
            x = n(F.relu(c(x)))
            if i in (1, 3):
                x = F.max_pool1d(x, 2)
            if i == 5:
                x = F.max_pool1d(x, 3)
        x = x.permute(0, 2, 1).reshape(len(x), -1)                       # Keras flattens time first, channels fastest
        for j, (k, b) in enumerate(self.dense):
            x = x @ k + b
            x = F.relu(x) if j == 0 else x
        return x[:, 0]


class GaitLab:
    def __init__(self, npz):
        w = np.load(npz)
        self.nets = {n: CNN(w, n) for n in MAPS}
        self.ols = {n: w[f"{n}/ols"] for n in MAPS}

    @torch.no_grad()
    def predict(self, name, filtered, cols):
        starts = range(0, filtered.shape[0] - WINDOW, STEP)
        x = np.stack([filtered[s:s + WINDOW, cols] for s in starts])
        p = self.nets[name](torch.from_numpy(x.astype(np.float32))).numpy()
        lo, span = MAPS[name]
        p = p * span + lo
        a, b = self.ols[name][:2]                                       # OLS on [prediction, 1]
        return float(np.mean(a * p + b))


# ---------------------------------------------------------------- the authors' preprocessing (video_process_utils.py), unchanged logic

from scipy import interpolate                                            # noqa: E402
from scipy.ndimage import gaussian_filter1d                              # noqa: E402

MHIP, RHIP, RKNE, RANK, LHIP, LKNE, LANK, LBTO, RBTO = 8, 9, 10, 11, 12, 13, 14, 19, 22
LEFT = [2 * LANK, 2 * LANK + 1, 2 * LKNE, 2 * LKNE + 1, 2 * LHIP, 2 * LHIP + 1, 2 * LBTO, 2 * LBTO + 1, 50, 52, 54, 56]
RIGHT = [2 * RANK, 2 * RANK + 1, 2 * RKNE, 2 * RKNE + 1, 2 * RHIP, 2 * RHIP + 1, 2 * RBTO, 2 * RBTO + 1, 50, 52, 54, 56]
BOTH = LEFT[:8] + RIGHT[:8] + [50, 51, 52, 53, 54, 55, 56]


def _fill(a):
    inds, good = np.arange(len(a)), np.where(np.isfinite(a))[0]
    if len(good) <= 1:
        return a
    f = interpolate.interp1d(inds[good], a[good], kind="linear", bounds_error=False)
    b = np.where(np.isfinite(a), a, f(inds))
    return np.where(np.isfinite(b), b, np.nanmean(b))


def preprocess(xy, swap=True):
    """xy (T, 25, 2) pixels, BODY_25 order; joints that are not used may be NaN. Returns (T, 50 + 8) exactly as the authors' demo builds it."""
    res = xy.reshape(len(xy), -1).astype(float).copy()
    res[res == 0] = np.nan
    res = np.apply_along_axis(_fill, 0, res)
    res = np.apply_along_axis(lambda c: gaussian_filter1d(c, 1), 0, res)
    mx, my = ((res[:, 2 * RHIP] + res[:, 2 * LHIP]) / 2)[:, None], ((res[:, 2 * RHIP + 1] + res[:, 2 * LHIP + 1]) / 2)[:, None]
    hip = np.hstack([mx, my] * 25)
    d = lambda i, j: np.linalg.norm(res[:, 2 * i:2 * i + 2] - res[:, 2 * j:2 * j + 2], axis=1)
    scale = ((d(RHIP, RKNE) + d(LHIP, LKNE)) / 2)[:, None]
    res = (res - hip) / scale
    orient = np.where((res[:, 2 * LANK] - res[:, 2 * LBTO]) + (res[:, 2 * RANK] - res[:, 2 * RBTO]) >= 0, 1, -1)[:, None]
    if swap:
        res[:, ::2] = res[:, ::2] / orient
    p = lambda i: res[:, 2 * i:2 * i + 2]
    def angle(a, b, c):
        u, v = p(a) - p(b), p(c) - p(b)
        # clipped: a perfectly straight limb gives 1 + 1e-16, which is NaN in the authors' code; no real value changes
        return np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1))
    cols = [angle(LANK, LKNE, LHIP), angle(RANK, RKNE, RHIP), angle(LBTO, LANK, LKNE), angle(RBTO, RANK, RKNE), np.linalg.norm(p(LBTO) - p(LANK), axis=1),
            np.linalg.norm(p(RBTO) - p(RANK), axis=1), res[:, 2 * RANK] - res[:, 2 * LANK], res[:, 2 * LANK] - res[:, 2 * RANK]]
    return np.hstack([res, np.stack(cols, 1)])


def predict_all(net: GaitLab, xy) -> dict:
    sw, ns = preprocess(xy, True), preprocess(xy, False)
    out = {n: net.predict(n, sw, BOTH) for n in ("GDI", "gmfcs", "speed", "cadence", "SEMLS_dev_residual")}
    out["KneeFlex_maxExtension_L"] = net.predict("KneeFlex_maxExtension", ns, LEFT)
    out["KneeFlex_maxExtension_R"] = net.predict("KneeFlex_maxExtension", ns, RIGHT)
    return out
