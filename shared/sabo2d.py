"""Plugin for the 2D ST-GCN models of Sabo et al. (one per pose extractor), shared by the two plugin folders."""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import numpy as np

from core import jointsets
from core.common import Prepared
from core.plugin import Manifest, Plugin, Requires
from core.view import Camera, look
from shared import stgcn

TRAIN_BODY_PX = 291.0        # body height (nose to ankles) in pixels of the study's 2D training data (from the checkpoints' input statistics)


def to_study_pixels(world: np.ndarray) -> np.ndarray:
    """(T, 13, 3) in (forward, left, up) -> (T, 13, 2): seen from in front, scaled so that the body is as tall as in the study's
    training images, hip centre at (100, 100). The study never normalises size, so the scale has to come from somewhere."""
    v = look(world, Camera(azimuth=0, elevation=0, output="3d"))[..., :2]           # level camera: the tilt of the study's RGB cameras is unknown
    x, y = v[..., 0], v[..., 1]
    height = np.median(y[:, 0] - (y[:, 11] + y[:, 12]) / 2)
    k = TRAIN_BODY_PX / max(height, 0.3)
    px = np.stack([x * k, -y * k], -1)
    hip = (px[:, 7] + px[:, 8]) / 2
    return px - hip[:, None] + 100.0


class Sabo2D(Plugin):
    extractor = "?"
    folder = Path(".")

    @cached_property
    def models(self):
        files = sorted((Path(os.environ.get("STGCN2D_WEIGHTS", "")) / self.extractor if os.environ.get("STGCN2D_WEIGHTS") else self.folder / "weights").glob("*.pt"), key=lambda p: int(p.stem))
        if not files:
            raise FileNotFoundError(f"no weights for {self.extractor}")
        return [stgcn.load(f) for f in files]

    def adapt(self, p: Prepared) -> np.ndarray:
        names, w = jointsets.select(p.world.body, "coco13_nose", p.world.xyz)
        return stgcn.study_preprocess_2d(to_study_pixels(w))

    def predict(self, x: np.ndarray) -> dict:
        import torch
        t = torch.from_numpy(x[None].astype(np.float32).copy())
        with torch.no_grad():
            s = np.array([float(m(t).item()) for m in self.models])
        padded = int((x == 0).all(axis=(0, 2)).sum())
        score = float(s.mean())
        return {"label": f"UPDRS-gait {int(np.clip(round(score), 0, 2))}", "score": score, "detail": {"folds": len(s), "fold_spread": round(float(s.std()), 3), "padded_frames": padded},
                "warnings": [f"clip_shorter_than_120_frames_zero_padded"] if padded else []}


def make(extractor: str, title: str, folder: Path, variant: str):
    class P(Sabo2D):
        manifest = Manifest(
            id=f"pd_stgcn_sabo2022_{variant}", version="0.1.0", condition=f"parkinsonism severity ({title} 2D model)",
            output="predicted MDS-UPDRS gait item, 0 (none) to 2, mean of the study's 10 cross-validation models; a severity estimate, not a diagnosis",
            study="Sabo et al., JBHI 2022 (2D model trained on " + title + " joints)", source="https://github.com/TaatiTeam/stgcn_parkinsonism_prediction", license="UHN, academic non-commercial use only",
            requires=Requires(joint_set="coco13_nose", views=("Side", "Front"), min_frames=60, min_observed=0.5), kind="pretrained",
            fidelity="port equals the original code on the study's sample walks; nose is estimated and the body is scaled to the study's size; accuracy on Nova-77 input not established")
    P.extractor, P.folder = extractor, folder
    return P
