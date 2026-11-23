import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from dataclasses import replace
import numpy as np
import pytest
from core import nova77
from core.common import prepare
from core.view import Camera, build_world, depth_scale, fit_pinhole, look, project, refine_with_2d, world_axes
from core.common import walking_axes
from test_core import synth

F, CX, CY = 1450.0, 640.0, 360.0


def capture(tmp_path, direction, pitch=0.0, dist=4.0, t=180, stretch=None, squeeze=None):
    """A synthetic walk as a real camera would deliver it: rotated by the camera's pitch, with a true pinhole 2D projection.
    `stretch` scales the legs of the reported 3D only, like the bias seen in the real export."""
    s = nova77.load(synth(tmp_path / "a.jsonl", t=t, direction=direction, planted=True))
    a = np.radians(pitch)                                        # camera looks down: rotate the scene about the camera's x axis
    Rm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    xyz = s.xyz @ Rm.T
    xyz = xyz + np.array([0, 0, dist - 5.0 + 5.0])
    xy = project(xyz, (F, CX, CY))
    if stretch:
        x3 = xyz.copy()
        for side in ("Left", "Right"):
            hip = xyz[:, s.index(side + "Leg")]
            for n in ("Shin", "Foot", "ToeBase", "ToeEnd"):
                i = s.index(side + n); x3[:, i] = hip + stretch * (xyz[:, i] - hip)
        xyz = x3
    if squeeze:                                                  # the pelvis path is too short in depth, the pose is right
        h = xyz[:, :1, 2]; xyz = xyz.copy(); xyz[..., 2] += (h - h[:1]) * (1 / squeeze - 1)
    return replace(s, xyz=xyz, xy=xy)


def rel(w):                                                      # hip-centred, so translation does not matter
    return w - (w[:, [nova77.JOINTS.index("LeftLeg")]] + w[:, [nova77.JOINTS.index("RightLeg")]]) / 2


def test_pinhole_is_recovered(tmp_path):
    f, cx, cy = fit_pinhole(capture(tmp_path, (1, 0)))
    assert (f, cx, cy) == pytest.approx((F, CX, CY), rel=1e-6)


@pytest.mark.parametrize("d", [(1, 0), (-1, 0), (0, -1), (0, 1), (1, 0.4)])
def test_same_walk_from_any_direction_gives_the_same_world(tmp_path, d):
    a = build_world(prepare(capture(tmp_path, (1, 0))).walk, refine=False).xyz
    b = build_world(prepare(capture(tmp_path, d)).walk, refine=False).xyz
    assert np.abs(rel(a) - rel(b)).max() < 1e-6


@pytest.mark.parametrize("pitch", [-12, 5, 10, 20])
def test_camera_tilt_is_removed_in_front_view(tmp_path, pitch):
    level = build_world(prepare(capture(tmp_path, (0, -1))).walk, refine=False)
    tilted = build_world(prepare(capture(tmp_path, (0, -1), pitch=pitch)).walk, refine=False)
    assert abs(np.tan(np.radians(tilted.floor_slope))) == pytest.approx(np.tan(np.radians(abs(pitch))) * 0.93, abs=0.012)
    assert np.abs(rel(level.xyz) - rel(tilted.xyz)).max() < 0.05


def test_short_depth_is_measured_and_does_not_steepen_the_floor(tmp_path):
    # the planted synthetic walker has ratio 1.0, so SIDE_SKATE (0.93, real side clips) shows up as a factor here
    for sq in (1.0, 1.5, 2.0):
        w = prepare(capture(tmp_path, (0, -1), pitch=6, squeeze=sq)).walk
        assert depth_scale(w, walking_axes(w)) == pytest.approx(sq / 0.93, rel=0.08)
    plain = build_world(prepare(capture(tmp_path, (0, -1), pitch=6)).walk).floor_slope
    short = build_world(prepare(capture(tmp_path, (0, -1), pitch=6, squeeze=1.5)).walk).floor_slope
    assert abs(short) == pytest.approx(abs(plain), abs=0.8)


def test_side_view_has_no_depth_correction(tmp_path):
    w = prepare(capture(tmp_path, (1, 0))).walk
    assert depth_scale(w, walking_axes(w)) == 1.0


def test_known_camera_tilt_is_used(tmp_path):
    w = prepare(capture(tmp_path, (0, -1), pitch=10)).walk
    assert build_world(w, tilt_deg=3.0).floor_slope == 3.0


def test_side_view_pitch_needs_calibration(tmp_path):
    level = build_world(prepare(capture(tmp_path, (1, 0))).walk, refine=False)
    p = prepare(capture(tmp_path, (1, 0), pitch=8)).walk
    assert np.abs(rel(level.xyz) - rel(build_world(p, refine=False).xyz)).max() > 0.08
    best = min((np.abs(rel(level.xyz) - rel(build_world(p, refine=False, lateral_deg=k).xyz)).max(), k) for k in (-8, 8))
    assert best[0] < 0.02


def test_virtual_camera_equals_a_real_camera(tmp_path):
    """Filmed from the side, converted to the view of a front camera tilted 30 degrees == what that camera sees."""
    side = build_world(prepare(capture(tmp_path, (1, 0))).walk, refine=False)
    front = look(side.xyz, Camera(azimuth=0, elevation=30))
    w = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])              # forward = -z of the camera's level frame, left = +x, up = +y
    real = capture(tmp_path, (0, -1)).xyz                         # x right, y down, z away (level)
    e = np.radians(30)
    x, y, z = real[..., 0], -real[..., 1], real[..., 2]           # y up
    tilt = np.stack([x, y * np.cos(e) + z * np.sin(e), -y * np.sin(e) + z * np.cos(e)], -1)
    assert np.abs(rel(front) - rel(tilt)).max() < 1e-6


def test_mirror_is_not_the_same_as_turning(tmp_path):
    w = build_world(prepare(capture(tmp_path, (1, 0))).walk, refine=False).xyz
    lf, rt = nova77.JOINTS.index("LeftFoot"), nova77.JOINTS.index("RightFoot")
    left_cam, right_cam = look(w, Camera(azimuth=90)), look(w, Camera(azimuth=-90))
    flipped = right_cam.copy(); flipped[..., 0] *= -1
    assert np.abs(rel(left_cam)[:, [lf, rt], 2] - rel(flipped)[:, [lf, rt], 2]).max() > 0.1       # which leg is near differs
    assert np.abs(rel(left_cam)[..., :2] - rel(flipped)[..., :2]).max() < 1e-9                     # but the outline is the same


def test_two_d_view_is_a_pinhole_image(tmp_path):
    w = build_world(prepare(capture(tmp_path, (1, 0))).walk, refine=False).xyz
    cam = Camera(azimuth=90, output="2d", distance_m=4, focal_px=1000)
    near = look(w, cam); far = look(w, replace(cam, distance_m=8))
    h = lambda q: np.median(np.linalg.norm(q[:, nova77.JOINTS.index("Head")] - q[:, nova77.JOINTS.index("LeftFoot")], axis=1))
    assert h(near) == pytest.approx(2 * h(far), rel=0.1)


def test_refine_removes_a_leg_length_bias(tmp_path):
    true = capture(tmp_path, (1, 0)); bad = capture(tmp_path, (1, 0), stretch=1.10)
    out, info = refine_with_2d(bad, cam=(F, CX, CY))
    i = [nova77.JOINTS.index(n) for n in ("LeftLeg", "LeftFoot")]
    leg = lambda s: np.median(np.linalg.norm(s.xyz[:, i[0]] - s.xyz[:, i[1]], axis=1))
    trunk = lambda s: np.median(np.linalg.norm(s.xyz[:, nova77.JOINTS.index("Hips")] - s.xyz[:, nova77.JOINTS.index("Neck1")], axis=1))
    assert leg(bad) / trunk(bad) > 1.08 * leg(true) / trunk(true)
    assert leg(out) / trunk(out) == pytest.approx(leg(true) / trunk(true), rel=0.02)
    foot = lambda s: np.median(np.linalg.norm(project(s.xyz[:, i[1]], (F, CX, CY)) - bad.xy[:, i[1]], axis=1))
    assert foot(bad) > 10 and foot(out) < 0.15 * foot(bad)


def test_refine_leaves_consistent_input_alone(tmp_path):
    s = capture(tmp_path, (1, 0))
    out, info = refine_with_2d(s)
    assert all(abs(v - 1) < 0.01 for v in info["scale"].values())
    assert np.nanmax(np.abs(out.xyz - s.xyz)) < 0.02


def test_refine_does_not_touch_raw(tmp_path):
    s = capture(tmp_path, (1, 0), stretch=1.1); before = s.xyz.copy()
    refine_with_2d(s)
    assert np.array_equal(s.xyz, before)


def test_refine_keeps_body_size_even_with_noisy_2d(tmp_path):
    """Regression: without the size constraint noisy 2D made every bone shrink to the lower limit."""
    s = capture(tmp_path, (1, 0), stretch=1.05)
    s = replace(s, xy=s.xy + np.random.default_rng(0).normal(0, 12, s.xy.shape))
    out, info = refine_with_2d(s, cam=(F, CX, CY))
    i = nova77.JOINTS.index
    size = lambda q: sum(np.median(np.linalg.norm(q.xyz[:, i(a)] - q.xyz[:, i(b)], axis=1)) for a, b in (("Hips", "Head"), ("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot")))
    assert size(out) == pytest.approx(size(s), rel=0.02)
