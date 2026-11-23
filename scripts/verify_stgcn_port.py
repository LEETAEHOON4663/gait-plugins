"""Does the ported plugin equal the study's original code? Runs the original model classes and the original
preprocessing functions on the study's own Kinect3D sample walks, and compares with the port."""
import copy, csv, glob, importlib, json, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO = Path(sys.argv[1])
import numpy as np, torch

# --- original code, imported with light stubs for mmskeleton's package __init__ files
pkg = REPO / "mmskeleton" / "mmskeleton"
def stub(name, path=None, **attrs):
    m = types.ModuleType(name)
    if path: m.__path__ = [str(path)]
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m
stub("icecream", ic=lambda *a, **k: None)
stub("mmskeleton", pkg); stub("mmskeleton.ops", pkg / "ops")
stub("mmskeleton.models", pkg / "models"); stub("mmskeleton.models.backbones", pkg / "models/backbones")
stub("mmskeleton.models.backbones.ST_GCN", pkg / "models/backbones/ST_GCN")
stub("mmskeleton.deprecated"); stub("mmskeleton.deprecated.datasets"); stub("mmskeleton.deprecated.datasets.utils", skeleton=None)
stub("mmskeleton.datasets", pkg / "datasets"); stub("mmskeleton.datasets.skeleton", pkg / "datasets/skeleton")
import builtins
_print, builtins.print = builtins.print, lambda *a, **k: None
Orig = importlib.import_module("mmskeleton.models.backbones.ST_GCN.st_gcn_aaai18_ordinal_smaller_2_position_pretrain").ST_GCN_18_ordinal_smaller_2_position_pretrain
proc = importlib.import_module("mmskeleton.datasets.skeleton.skeleton_process")
ck = REPO / "sample_data/model_zoo/kinect3d_models_to_share/v2/13/0.1/1/all/(0,0,0)/gait_feats_False/checkpoint_final.pt"
om = Orig(in_channels=3, num_class=3, graph_cfg={"layout": "kinect_coco_simplified_head", "strategy": "spatial"},
          edge_importance_weighting=True, dropout=0.1, num_ts_predicting=1, num_joints_predicting=4,
          temporal_kernel_size=13, use_gait_features=False)
om.set_stage_2(); sd = torch.load(ck, map_location="cpu", weights_only=True)
missing = om.load_state_dict(sd, strict=False); om.eval()
builtins.print = _print
print("original load_state_dict:", missing)

# --- port
from core.plugin import discover
plugins, failures = discover()
assert "pd_stgcn_sabo2022" not in failures, failures
P = sys.modules["gait_plugin_pd_stgcn_sabo2022"]
plug = next(p for p in plugins if p.manifest.id == "pd_stgcn_sabo2022")
spec = importlib.util.spec_from_file_location("m", ROOT / "plugins/pd_stgcn_sabo2022/model.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
print("graph equals checkpoint buffer:", np.allclose(M.adjacency(), sd["encoder.A"].numpy(), atol=1e-6))

NAMES = ["Head", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]
out, tilts, worst = [], [], 0.0
for f in sorted(glob.glob(str(REPO / "sample_data/data/kinect_skel_data_3d/kinect_all/*.csv"))):
    rows = list(csv.DictReader(open(f)))
    k = np.array([[[float(r[n + a]) for a in ("_x", "_y", "_z")] for n in NAMES] for r in rows], np.float32)      # T,13,3
    # original preprocessing on its own data layout (C, V, T, M)
    d = {"info": {"keypoint_channels": ["x", "y", "z"], "resolution": [1, 1, 1]}, "keypoint_order": NAMES,
         "data": k.transpose(2, 1, 0)[..., None].copy(), "data_flipped": k.transpose(2, 1, 0)[..., None].copy()}
    d = proc.kinect3D_centre_hip(d, [0, 0, 0]); d = proc.normalize_by_resolution(d); d = proc.mask_by_visibility(d)
    d = proc.pad_zero_beginning(d, 120); d = proc.crop_middle(d, 120); d = proc.transpose(d, [0, 2, 1, 3])
    xo = d["data"][None]
    xp = P.study_preprocess(k, NAMES)[None]
    with torch.no_grad():
        so = om(torch.from_numpy(xo.astype(np.float32)), torch.zeros(1, 0)).item()
        sp = plug.model(torch.from_numpy(xp)).item()
    di, ds = float(np.abs(xo[..., 0] - xp).max()), abs(so - sp); worst = max(worst, di, ds)
    up = k[:, 0] - (k[:, 11] + k[:, 12]) / 2
    tilts.append(float(np.degrees(np.arctan2(-up[:, 2], up[:, 1])).mean()))
    toward = float((k[-1, 7:9] - k[0, 7:9]).mean(0)[2]); left = float((k[:, 7, 0] - k[:, 8, 0]).mean())
    out.append((Path(f).name, len(rows), rows[0]["UPDRS_gait"], so, sp))
    print(f"{Path(f).name[:36]:36s} T={len(rows):3d} label={rows[0]['UPDRS_gait']} original={so:.6f} port={sp:.6f} input_diff={di:.1e} | dz={toward:+.2f} m (walks toward camera) LHip-RHip x={left:+.2f} tilt={tilts[-1]:.1f} deg")
print(f"worst difference {worst:.2e} | mean camera tilt {np.mean(tilts):.1f} deg (plugin uses {P.TILT_DEG})")
sys.exit(0 if worst < 1e-5 else 1)
