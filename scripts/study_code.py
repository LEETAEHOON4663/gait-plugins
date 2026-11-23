"""Run the study's ORIGINAL ST-GCN code (Sabo et al.) on 2D tracks, unmodified, with light stubs instead of mmskeleton/mmcv."""
import builtins, importlib, sys, types
from pathlib import Path

import numpy as np
import torch

JOINTS = ["Nose", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]
ENGINES = {"alphapose": ("alphapose_models_to_share/v0/13/0.2", "orig"), "detectron": ("detectron_models_to_share/v2/13/0.2", "smaller_2")}


class StudyModel:
    """The study's ensemble for one pose extractor (10 cross-validation folds)."""

    def __init__(self, repo, engine):
        self.repo, (folder, arch) = Path(repo), ENGINES[engine]
        self._stub()
        with _quiet():
            mod = importlib.import_module(f"mmskeleton.models.backbones.ST_GCN.st_gcn_aaai18_ordinal_{arch}_position_pretrain")
            cls = getattr(mod, f"ST_GCN_18_ordinal_{arch}_position_pretrain")
            self.proc = importlib.import_module("mmskeleton.datasets.skeleton.skeleton_process")
            self.models = []
            for f in sorted((self.repo / "sample_data/model_zoo" / folder).glob("*/checkpoint_final.pt"), key=lambda p: int(p.parent.name)):
                m = cls(in_channels=2, num_class=3, graph_cfg={"layout": "coco_simplified_head", "strategy": "spatial"}, edge_importance_weighting=True,
                        dropout=0.2, num_ts_predicting=1, num_joints_predicting=4, temporal_kernel_size=13, use_gait_features=False)
                m.set_stage_2()
                m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
                self.models.append(m.eval())

    def _stub(self):
        pkg = self.repo / "mmskeleton" / "mmskeleton"
        def add(name, path=None, **kw):
            m = types.ModuleType(name)
            if path: m.__path__ = [str(path)]
            for k, v in kw.items(): setattr(m, k, v)
            sys.modules.setdefault(name, m)
        add("icecream", ic=lambda *a, **k: None); add("mmskeleton", pkg); add("mmskeleton.ops", pkg / "ops"); add("mmskeleton.models", pkg / "models")
        add("mmskeleton.models.backbones", pkg / "models/backbones"); add("mmskeleton.models.backbones.ST_GCN", pkg / "models/backbones/ST_GCN")
        add("mmskeleton.deprecated"); add("mmskeleton.deprecated.datasets"); add("mmskeleton.deprecated.datasets.utils", skeleton=None)
        add("mmskeleton.datasets", pkg / "datasets"); add("mmskeleton.datasets.skeleton", pkg / "datasets/skeleton")

    def preprocess(self, xy, conf, resolution=(1920, 1080)):
        """xy (T, 13, 2) already centred at the hips on (100, 100) in pixels; conf (T, 13). Uses the study's own functions."""
        d = np.concatenate([xy, conf[..., None]], -1).astype(np.float32).transpose(2, 1, 0)[..., None]
        s = {"info": {"keypoint_channels": ["x", "y", "score"], "resolution": list(resolution)}, "keypoint_order": JOINTS, "data": d.copy(), "data_flipped": d.copy()}
        for f, kw in ((self.proc.normalize_by_resolution, {}), (self.proc.mask_by_visibility, {}), (self.proc.pad_zero_beginning, {"size": 120}),
                      (self.proc.crop_middle, {"size": 120}), (self.proc.transpose, {"order": [0, 2, 1, 3]})):
            s = f(s, **kw)
        return s["data"]

    @torch.no_grad()
    def scores(self, xy, conf, resolution=(1920, 1080)):
        """One score per fold. `resolution` is (width, height), as in the study's loader (portrait video: 1080, 1920)."""
        x = torch.from_numpy(self.preprocess(xy, conf, resolution)[None].astype(np.float32))
        return np.array([m(x, torch.zeros(1, 0)).item() for m in self.models])


class _quiet:
    def __enter__(self): self.p, builtins.print = builtins.print, lambda *a, **k: None
    def __exit__(self, *a): builtins.print = self.p


def centre(xy, names=JOINTS):
    """The study's 2D input: hip centre at (100, 100), pixels, no scaling."""
    hip = (xy[:, names.index("LHip")] + xy[:, names.index("RHip")]) / 2
    return xy - hip[:, None] + 100.0
