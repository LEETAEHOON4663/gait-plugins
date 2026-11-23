"""Does the verdict work? Real TOAGA tracks are written as if they were Nova-77 files, so the right answer is known beforehand:
AlphaPose joints must give no difference and pass; Detectron joints must match the yardstick and pass; OpenPose joints must fail.
Usage: toaga_selftest.py <PoseTracking> <repo> <tmp folder>"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import jointsets, nova77
from scripts.study_code import JOINTS
from scripts.toaga import at_30fps, fill, index, read
from scripts.toaga_compare import main
tracks, repo, tmp = sys.argv[1], sys.argv[2], Path(sys.argv[3])
idx = index(tracks)
MAP = {n: JOINTS.index(c) for c, n in (("LShoulder", "LeftArm"), ("RShoulder", "RightArm"), ("LElbow", "LeftForeArm"), ("RElbow", "RightForeArm"), ("LWrist", "LeftHand"),
       ("RWrist", "RightHand"), ("LHip", "LeftLeg"), ("RHip", "RightLeg"), ("LKnee", "LeftShin"), ("RKnee", "RightShin"), ("LAnkle", "LeftFoot"), ("RAnkle", "RightFoot"))}
MAP.update({n: 0 for n in ("LeftEye", "RightEye", "Jaw")})

def write(engine, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for person, cam in sorted({k[:2] for k in idx}):
        rows, frame = [], 0
        for k in sorted(k for k in idx if k[:2] == (person, cam) and k[2] == "front" and engine in idx[k]):
            t, xy, _ = read(idx[k][engine]); xy = at_30fps(t, fill(xy))[1]
            if not np.isfinite(xy).all(): continue
            for f in range(len(xy) + 45):                      # a walk, then 1.5 s without a person
                if f < len(xy):
                    p2 = np.repeat(((xy[f, 7] + xy[f, 8]) / 2)[None], 77, 0)
                    for n, j in MAP.items(): p2[nova77.JOINTS.index(n)] = xy[f, j]
                    p3 = np.concatenate([(p2 - [540, 960]) / 1600 * 4, np.full((77, 1), 4.0)], 1)
                    det = [{"tracking_id": 1, "keypoints_2d": (p2 * 0.375).tolist(), "keypoints_3d": p3.tolist(), "keypoints_confidence": [0.9] * 77, "rest_pose": [], "joint_rotations": [], "root_pose": {}}]
                else: det = []
                rows.append(json.dumps({"frame_id": frame, "time_ms": round(frame * 1000 / 30), "detections": det})); frame += 1
        (folder / f"OAW{person:02d}-{cam}.jsonl").write_text("\n".join(rows))

want = {"alphapose": True, "detectron": True, "openpose": False}
bad = 0
for eng, expect in want.items():
    d = tmp / f"mock_{eng}"; write(eng, d)
    print(f"\n######## joints from {eng}, written as Nova-77 files (3D is a placeholder; only the 2D test runs)")
    r = main(d, tracks, repo, tmp / f"out_{eng}", view_change=False)
    got = r["alphapose_2d"]["passed"]; ok = got is expect; bad += not ok
    print(f"-> verdict for the AlphaPose model: {'PASS' if got else 'FAIL' if got is False else 'NO VERDICT'}; expected {'PASS' if expect else 'FAIL'}: {'as expected' if ok else 'WRONG'}")
sys.exit(bool(bad))
