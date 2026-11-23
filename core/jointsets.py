"""Joint sets that studies use, expressed in canonical joints.

A string is a direct joint, a tuple is the midpoint of joints (marked "derived"), and None means the
canonical skeleton cannot supply it. A plugin that needs a None joint becomes ineligible instead of guessing.
"""
from __future__ import annotations

import numpy as np

from .skeleton import Skeleton

_LR = {"Shoulder": "Arm", "Elbow": "ForeArm", "Wrist": "Hand", "Hip": "Leg", "Knee": "Shin", "Ankle": "Foot"}


def _pairs(parts, prefix=("L", "R")):
    return {f"{p}{n}": f"{side}{_LR[n]}" for n in parts for p, side in zip(prefix, ("Left", "Right"))}


JOINT_SETS = {
    # Sabo et al. ST-GCN, Kinect layout "kinect_coco_simplified_head"; Kinect's Head is the head centre
    "kinect13": {"Head": ("Head", "HeadEnd"), **_pairs(["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"])},
    "h36m17": {"Pelvis": "Hips", "RHip": "RightLeg", "RKnee": "RightShin", "RAnkle": "RightFoot", "LHip": "LeftLeg",
               "LKnee": "LeftShin", "LAnkle": "LeftFoot", "Spine": "Spine2", "Thorax": "Chest", "Neck": "Neck1",
               "Head": "HeadEnd", "LShoulder": "LeftArm", "LElbow": "LeftForeArm", "LWrist": "LeftHand",
               "RShoulder": "RightArm", "RElbow": "RightForeArm", "RWrist": "RightHand"},
    "coco17": {"Nose": None, "LEye": "LeftEye", "REye": "RightEye", "LEar": None, "REar": None,
               **_pairs(["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"])},
    # 2D models trained on COCO joints without eyes and ears. Nova-77 has no nose, so it is the mean of the eyes and the jaw.
    "coco13_nose": {"Nose": ("LeftEye", "RightEye", "Jaw"), **_pairs(["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"])},
    "gaitlab_lower": {**_pairs(["Hip", "Knee", "Ankle"]), "LBigToe": "LeftToeEnd", "RBigToe": "RightToeEnd"},
    "lower_limb_arms": {"Pelvis": "Hips", "Neck": "Neck1", **_pairs(["Shoulder", "Wrist", "Hip", "Knee", "Ankle"]),
                        "LToe": "LeftToeBase", "RToe": "RightToeBase"},
}


def resolve(name: str, joints) -> dict:
    """Which joints of a set come directly, which are derived, and which are unavailable."""
    out = {"direct": [], "derived": [], "unavailable": []}
    for target, src in JOINT_SETS[name].items():
        parts = () if src is None else (src,) if isinstance(src, str) else src
        kind = "unavailable" if not parts or any(p not in joints for p in parts) else "direct" if len(parts) == 1 else "derived"
        out[kind].append(target)
    return out


def canonical_joints(name: str) -> list:
    out = []
    for src in JOINT_SETS[name].values():
        out += [] if src is None else [src] if isinstance(src, str) else list(src)
    return out


def select_xy(s: Skeleton, name: str) -> tuple[list, np.ndarray]:
    """Image positions (T, K, 2) in the order the study expects."""
    bad = resolve(name, s.joints)["unavailable"]
    if bad:
        raise KeyError(f"{name}: joints not available: {bad}")
    return list(JOINT_SETS[name]), np.stack([np.mean([s.xy[:, s.index(p)] for p in ((src,) if isinstance(src, str) else src)], axis=0)
                                             for src in JOINT_SETS[name].values()], axis=1)


def select(s: Skeleton, name: str, xyz: np.ndarray | None = None) -> tuple[list, np.ndarray]:
    """Return (joint names, (T, K, 3)) in the order the study expects, from camera coordinates or from `xyz` (e.g. world.xyz)."""
    xyz = s.xyz if xyz is None else xyz
    bad = resolve(name, s.joints)["unavailable"]
    if bad:
        raise KeyError(f"{name}: joints not available: {bad}")
    cols = []
    for src in JOINT_SETS[name].values():
        parts = (src,) if isinstance(src, str) else src
        cols.append(np.mean([xyz[:, s.index(p)] for p in parts], axis=0))
    return list(JOINT_SETS[name]), np.stack(cols, axis=1)
