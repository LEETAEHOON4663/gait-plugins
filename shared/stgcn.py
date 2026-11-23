"""ST-GCN of Sabo et al. 2022 in every published size and input dimension, rebuilt so a checkpoint loads without mmcv.
Channel counts, strides and residual connections are read from the weights themselves."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S_KERNEL = 3


class _GCN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cout, self.conv = cout, nn.Conv2d(cin, cout * S_KERNEL, 1)

    def forward(self, x, A):
        x = self.conv(x)
        n, _, t, v = x.shape
        return torch.einsum("nkctv,kvw->nctw", x.view(n, S_KERNEL, self.cout, t, v), A).contiguous()


class _Block(nn.Module):
    def __init__(self, cin, cout, stride, residual, tk):
        super().__init__()
        self.gcn = _GCN(cin, cout)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(), nn.Conv2d(cout, cout, (tk, 1), (stride, 1), ((tk - 1) // 2, 0)), nn.BatchNorm2d(cout), nn.Dropout(0.0))
        self.residual = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)) if residual == "project" else None
        self.kind = residual

    def forward(self, x, A):
        res = 0 if self.kind == "none" else x if self.kind == "identity" else self.residual(x)
        return F.relu(self.tcn(self.gcn(x, A)) + res)


class STGCN(nn.Module):
    def __init__(self, spec, in_channels, joints, out):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.register_buffer("A", torch.zeros(S_KERNEL, joints, joints))
        self.encoder.data_bn = nn.BatchNorm1d(in_channels * joints)
        self.encoder.st_gcn_networks = nn.ModuleList([_Block(*s) for s in spec])
        self.encoder.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(S_KERNEL, joints, joints)) for _ in spec])
        self.in_channels = in_channels
        self.pretrain_head = nn.Conv2d(spec[-1][1], 4 * in_channels, 1)          # unused at inference; kept for a strict load
        self.classification_head = nn.Conv2d(spec[-1][1], 1, 1)
        self.head = nn.Conv2d(spec[-1][1], out, 1)

    def forward(self, x):                                                          # (N, C, T, V)
        n, c, t, v = x.shape
        x = self.encoder.data_bn(x.permute(0, 3, 1, 2).reshape(n, v * c, t)).view(n, v, c, t).permute(0, 2, 3, 1)
        for block, imp in zip(self.encoder.st_gcn_networks, self.encoder.edge_importance):
            x = block(x, self.encoder.A * imp)
        return self.head(F.avg_pool2d(x, x.shape[2:])).flatten(1)


def load(path) -> STGCN:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    n = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.st_gcn_networks."))
    tk = sd["encoder.st_gcn_networks.0.tcn.2.weight"].shape[2]
    spec = []
    for i in range(n):
        p = f"encoder.st_gcn_networks.{i}."
        cin, cout = sd[p + "gcn.conv.weight"].shape[1], sd[p + "gcn.conv.weight"].shape[0] // S_KERNEL
        proj = p + "residual.0.weight" in sd
        spec.append((cin, cout, 2 if proj else 1, "none" if i == 0 else "project" if proj else "identity", tk))
    joints = sd["encoder.A"].shape[1]
    m = STGCN(spec, sd["encoder.data_bn.weight"].shape[0] // joints, joints, sd["head.weight"].shape[0])
    m.load_state_dict(sd, strict=True)
    return m.eval()


def study_preprocess_2d(xy: np.ndarray, window: int = 120, conf: np.ndarray | None = None) -> np.ndarray:
    """(T, 13, 2) pixels with the hip centre at (100, 100) -> (2, window, 13): the study's normalize_by_resolution (1920 x 1080),
    mask_by_visibility (joints with confidence 0 become 0), pad_zero_beginning and crop_middle, in that order."""
    x = xy.astype(np.float32).copy()
    x[..., 0], x[..., 1] = x[..., 0] / 1920 - 0.5, x[..., 1] / 1080 - 0.5
    if conf is not None:
        x[conf == 0] = 0.0
    if len(x) < window:
        x = np.concatenate([np.zeros((window - len(x), *x.shape[1:]), np.float32), x])
    elif len(x) > window:
        b = int((len(x) - window) / 2)
        x = x[b:b + window]
    return x.transpose(2, 0, 1)
