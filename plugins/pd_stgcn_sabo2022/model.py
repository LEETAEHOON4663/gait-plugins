"""ST-GCN of Sabo et al. 2022, rebuilt so the published checkpoint loads without mmcv/mmskeleton.

Parameter names match the checkpoint (strict load). scripts/verify_stgcn_port.py checks that outputs equal
the original implementation's.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_KERNEL, S_KERNEL, V = 13, 3, 13
_EDGES = [(13, 11), (12, 10), (11, 9), (10, 8), (9, 8), (8, 2), (9, 3), (3, 2), (2, 4), (3, 5), (5, 7), (1, 3), (1, 2)]


def adjacency() -> np.ndarray:
    """Spatial partitioning of the 13-joint Kinect graph (root / closer / further). Kept for reference and tests;
    inference uses the `A` buffer stored in the checkpoint."""
    a = np.zeros((V, V))
    for i, j in [(k, k) for k in range(V)] + [(i - 1, j - 1) for i, j in _EDGES]:
        a[i, j] = a[j, i] = 1
    hop = np.full((V, V), np.inf)
    reach = [np.linalg.matrix_power(a, d) > 0 for d in range(2)]
    for d in (1, 0):
        hop[reach[d]] = d
    adj = (hop <= 1).astype(float)
    norm = adj @ np.diag(1 / adj.sum(0))
    out = []
    for h in (0, 1):
        root, close, further = np.zeros((V, V)), np.zeros((V, V)), np.zeros((V, V))
        for i in range(V):
            for j in range(V):
                if hop[j, i] == h:
                    (root if hop[j, 0] == hop[i, 0] else close if hop[j, 0] > hop[i, 0] else further)[j, i] = norm[j, i]
        out += [root] if h == 0 else [root + close, further]
    return np.stack(out)


class _GCN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cout = cout
        self.conv = nn.Conv2d(cin, cout * S_KERNEL, 1)

    def forward(self, x, A):
        x = self.conv(x)
        n, _, t, v = x.shape
        return torch.einsum("nkctv,kvw->nctw", x.view(n, S_KERNEL, self.cout, t, v), A).contiguous()


class _Block(nn.Module):
    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.gcn = _GCN(cin, cout)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, (T_KERNEL, 1), (stride, 1), ((T_KERNEL - 1) // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(0.0))
        self.residual = None
        if residual == "project":
            self.residual = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        self.kind = residual

    def forward(self, x, A):
        res = 0 if self.kind == "none" else x if self.kind == "identity" else self.residual(x)
        return F.relu(self.tcn(self.gcn(x, A)) + res)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("A", torch.zeros(S_KERNEL, V, V))
        self.data_bn = nn.BatchNorm1d(3 * V)
        self.st_gcn_networks = nn.ModuleList([_Block(3, 32, 1, "none"), _Block(32, 32, 1, "identity"),
                                              _Block(32, 64, 2, "project"), _Block(64, 64, 1, "identity")])
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(S_KERNEL, V, V)) for _ in range(4)])

    def forward(self, x):                                   # (N, 3, T, V)
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(n, v * c, t)).view(n, v, c, t).permute(0, 2, 3, 1)
        for block, imp in zip(self.st_gcn_networks, self.edge_importance):
            x = block(x, self.A * imp)
        return F.avg_pool2d(x, x.shape[2:])


class STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.pretrain_head = nn.Conv2d(64, 12, 1)            # unused at inference; kept for a strict checkpoint load
        self.classification_head = nn.Conv2d(64, 1, 1)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).flatten(1)


def load(path) -> STGCN:
    m = STGCN()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.eval()
