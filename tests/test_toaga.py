import csv, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import pytest
from core import jointsets, nova77
from core.common import prepare
from core.view import project
from dataclasses import replace
from test_core import synth

REPO = Path(os.environ.get("STGCN_REPO") or next((p for p in (ROOT / "third_party", ROOT.parent / "third_party") if p.is_dir()), ROOT / "third_party")
            / "stgcn_parkinsonism_prediction")
pytestmark = pytest.mark.skipif(not (REPO / "sample_data/model_zoo/detectron_models_to_share").exists(), reason="study repository not available")
F, CX, CY = 1600.0, 540.0, 960.0                                    # a 1080 x 1920 portrait camera
COCO = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]


def camera_walk(tmp, t, pitch, direction=(0, -1)):
    """Synthetic walk seen by a camera at 5 m that looks down by `pitch` degrees. 3D in that camera's axes, 2D by a true pinhole."""
    s = nova77.load(synth(tmp / "w.jsonl", t=t, direction=direction, planted=True))
    a = np.radians(pitch); R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    xyz = s.xyz @ R.T
    xyz = xyz + np.array([0, 0, 5.0])
    return replace(s, xyz=xyz, xy=project(xyz, (F, CX, CY)))


def dump_nova(s, path, scale=1.0, lead=40, tail=30):
    """Write a Skeleton as the export writes it, inside a longer video: `lead` empty seconds of standing before it, 2D resized by `scale`."""
    rows = []
    n = len(s)
    still = [0] * lead + list(range(n)) + [n - 1] * tail
    rng = np.random.default_rng(0)
    for i, f in enumerate(still):
        xy = s.xy[f] + (rng.normal(0, 25, s.xy[f].shape) + np.array([300, 0]) if i < lead or i >= lead + n else 0)
        d = {"tracking_id": 1, "keypoints_2d": (xy * scale).tolist(), "keypoints_3d": s.xyz[f].tolist(), "keypoints_confidence": [0.9] * 77, "rest_pose": [], "joint_rotations": [], "root_pose": {}}
        rows.append({"frame_id": i, "time_ms": round(i * 1000 / 30), "detections": [d]})
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))


def dump_track(s, path, noise=0.0, stretch=1.0, seed=1):
    xy = jointsets.select_xy(s, "coco13_nose")[1]
    if noise: xy = xy + np.random.default_rng(seed).normal(0, noise, xy.shape)
    if stretch != 1: xy = xy[:, [0]] + (xy - xy[:, [0]]) * np.array([1.0, stretch])
    from scripts.study_code import JOINTS
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["time"] + [c + s_ for c in COCO for s_ in ("_x", "_y", "_conf")])
        for i in range(len(xy)):
            v = []
            for c in COCO:
                v += [*xy[i, JOINTS.index(c)], 0.9] if c in JOINTS else [0, 0, 0]
            w.writerow([round(i / 30, 3)] + v)


def make(tmp, pitch_top=12, n_people=3, stretch=1.0, scale=0.375):
    nim, tr = tmp / "nim", tmp / "tracks"; nim.mkdir(); tr.mkdir()
    for p in range(1, n_people + 1):
        for cam, pitch in (("bottom", 0), ("top", pitch_top)):
            s = camera_walk(tmp, 150 + 10 * p, pitch)
            dump_nova(s, nim / f"OAW{p:02d}-{cam}.jsonl", scale)
            for eng, noise in (("AlphaPose", 0.0), ("Detectron", 4.0)):
                dump_track(s, tr / f"OAW{p:02d}-{eng}-{cam}-front-1.csv", noise, stretch if eng == "AlphaPose" else 1.0)
    return nim, tr


def test_same_joints_give_the_same_scores_and_the_walk_is_found(tmp_path, capsys):
    from scripts.toaga_compare import main
    nim, tr = make(tmp_path)
    r = main(nim, tr, REPO, tmp_path / "out")
    assert r["matched"] == 6
    for m in ("alphapose", "detectron"):
        assert r[f"{m}_2d"]["new"]["diff"] < 0.02
    assert "image scale between the files: 2.667" in capsys.readouterr().out


def test_a_distorted_skeleton_does_not_align(tmp_path, capsys):
    """The real-data negative control is scripts/toaga_selftest.py (OpenPose joints must fail the verdict)."""
    from scripts.toaga_compare import align
    s = camera_walk(tmp_path, 150, 0); xy = jointsets.select_xy(s, "coco13_nose")[1]
    tall = xy[:, [0]] + (xy - xy[:, [0]]) * np.array([1.0, 1.6])
    from scripts.toaga_compare import USE
    good, bad = align(np.concatenate([xy, xy]), xy), align(np.concatenate([xy, xy]), tall)
    assert good[0] == 0 and good[2] < 0.5
    assert bad[2] > 20                                             # the error is reported per axis, so a one-axis stretch shows


def test_virtual_top_camera_matches_the_real_one(tmp_path, capsys):
    from scripts.toaga_compare import main
    nim, tr = make(tmp_path)
    r = main(nim, tr, REPO, tmp_path / "out")
    for m in ("alphapose", "detectron"):
        assert r[f"{m}_view"]["new"]["n"] == 3
        assert r[f"{m}_view"]["new"]["diff"] < 0.15, r[f"{m}_view"]


def test_view_change_is_not_a_no_op(tmp_path, capsys):
    """The camera difference must matter for the score, otherwise the test above proves nothing."""
    from scripts.toaga_compare import main
    nim, tr = make(tmp_path, pitch_top=25)
    r = main(nim, tr, REPO, tmp_path / "out")
    from scripts.toaga import index, read, at_30fps, centre, fill, smooth, STUDY_RES
    from scripts.study_code import StudyModel
    sm, idx = StudyModel(REPO, "alphapose"), index(tr)
    f = lambda k: float(np.mean(sm.scores(centre(smooth(fill(at_30fps(*read(idx[k]["alphapose"])[:2])[1]))), np.ones((len(at_30fps(*read(idx[k]["alphapose"])[:2])[0]), 13)), STUDY_RES)))
    assert abs(f((1, "top", "front", 1)) - f((1, "bottom", "front", 1))) > 0.1


def test_criteria():
    from scripts.toaga_criteria import summarise, verdict
    rng = np.random.default_rng(0); people = np.repeat(np.arange(14), 5); ref = rng.normal(.8, .5, 70)
    yard = summarise(ref, ref + rng.normal(0, .25, 70), people)
    assert verdict(summarise(ref, ref + rng.normal(0, .25, 70), people), yard)[0] is True        # the same kind of swap passes
    assert verdict(summarise(ref, ref + rng.normal(0, .8, 70), people), yard)[0] is False        # much noisier
    assert verdict(summarise(ref, ref + .9, people), yard)[0] is False                           # large bias
    assert verdict(summarise(ref, rng.normal(.8, .5, 70), people), yard)[0] is False             # unrelated scores
    assert verdict(summarise(ref, ref, people), yard, coverage=0.5)[0] is False                  # half the walks were lost
    assert verdict(summarise(ref[:10], ref[:10], people[:10]), yard)[0] is None
