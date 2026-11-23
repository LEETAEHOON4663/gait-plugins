import json, sys, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pytest
from core import jointsets, nova77
from core.common import fill_short_gaps, in_walk_frame, main_segment, prepare, walking_axes
from core.plugin import discover, run
from core.skeleton import INTERPOLATED, MISSING, OBSERVED, InputError


def synth(path, t=150, gap=(), direction=(1.0, 0.0), bbox=False, fps=30.0, planted=False):
    """Write a synthetic walk in the team's export format (no bbox, time_ms rounded to whole ms)."""
    f = np.array([direction[0], 0, direction[1]]) / np.hypot(*direction)
    up = np.array([0, -1.0, 0]); left = np.cross(up, f)
    body = {"Hips": (0, 0, 0), "Spine2": (0, 0, .25), "Chest": (0, 0, .35), "Neck1": (0, 0, .5), "Head": (0, 0, .58), "HeadEnd": (0, 0, .75)}
    for s, k in (("Left", 1), ("Right", -1)):
        body.update({s + "Arm": (0, .18 * k, .45), s + "ForeArm": (0, .2 * k, .2), s + "Hand": (0, .2 * k, -.05), s + "Leg": (0, .1 * k, 0),
                     s + "Shin": (0, .1 * k, -.42), s + "Foot": (0, .1 * k, -.84), s + "ToeBase": (.14, .1 * k, -.88), s + "ToeEnd": (.2, .1 * k, -.88)})
    with open(path, "w") as fh:
        for i in range(t):
            ph = 2 * np.pi * i / fps / 1.1
            pts = np.zeros((77, 3))
            for j, n in enumerate(nova77.JOINTS):
                fw, lf, u = body.get(n, (0, 0, .3))
                sgn = 1 if n.startswith("Left") else -1 if n.startswith("Right") else 0
                amp = {"Shin": .15, "Foot": .32, "ToeBase": .32, "ToeEnd": .32}.get(n.replace("Left", "").replace("Right", ""), 0)
                amp = -.2 if n.endswith("Hand") else amp
                if planted and n.replace("Left", "").replace("Right", "") in ("Foot", "ToeBase", "ToeEnd"):
                    c = ((i / fps / 1.1) + (0 if sgn > 0 else .5)) % 1          # 60% stance: the foot stays where it is on the floor
                    fw += .33 - 1.1 * c if c < .6 else -.33 + .66 * (c - .6) / .4
                    u += 0 if c < .6 else .06 * np.sin(np.pi * (c - .6) / .4)
                    amp = 0
                pts[j] = np.array([-2 * f[0], 0, 5 - 2 * f[2]]) + f * (i / fps * 1.0 + fw + sgn * amp * np.sin(ph)) + left * lf + up * u
            d = {"tracking_id": 1, "keypoints_2d": (pts[:, :2] * 100 + 500).tolist(), "keypoints_3d": pts.tolist(), "keypoints_confidence": [0.9] * 77,
                 "rest_pose": [], "joint_rotations": [], "root_pose": {}}
            if bbox:
                d["bbox"] = [0, 0, 1, 1]
            fh.write(json.dumps({"frame_id": i, "time_ms": round(i * 1000 / fps), "detections": [] if i in gap else [d]}) + "\n")
    return path


def test_joint_order_matches_official_names():
    assert [nova77.JOINTS.index(n) for n in ("Hips", "Head", "LeftArm", "LeftHand", "LeftHandIndex1", "RightShoulder", "RightHand", "LeftLeg", "LeftToeEnd", "RightLeg", "RightToeEnd")] \
        == [0, 6, 12, 14, 19, 39, 42, 67, 71, 72, 76]
    assert len(set(nova77.JOINTS)) == 77


def test_loads_export_with_and_without_bbox(tmp_path):
    a = nova77.load(synth(tmp_path / "a.jsonl")); b = nova77.load(synth(tmp_path / "b.jsonl", bbox=True))
    assert len(a) == len(b) == 150 and np.array_equal(a.xyz, b.xyz)


def test_gzipped_file_loads_the_same(tmp_path):
    import gzip
    p = synth(tmp_path / "a.jsonl"); g = tmp_path / "a.jsonl.gz"; g.write_bytes(gzip.compress(p.read_bytes()))
    a, b = nova77.load(p), nova77.load(g)
    assert np.array_equal(a.xyz, b.xyz, equal_nan=True) and a.source["sha256"] == b.source["sha256"]


def test_fps_is_30_despite_rounded_milliseconds(tmp_path):
    assert nova77.load(synth(tmp_path / "a.jsonl")).fps == 30.0


@pytest.mark.parametrize("edit", [lambda r: r["detections"][0].pop("keypoints_3d"), lambda r: r["detections"][0]["keypoints_3d"].pop(),
                                  lambda r: r["detections"][0]["keypoints_3d"][0].__setitem__(0, float("nan")), lambda r: r.__setitem__("frame_id", 0)])
def test_malformed_input_is_rejected(tmp_path, edit):
    rows = [json.loads(l) for l in open(synth(tmp_path / "a.jsonl", t=20))]
    edit(rows[5])
    (tmp_path / "bad.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(InputError):
        nova77.load(tmp_path / "bad.jsonl")


def test_gap_filling_flags_and_limits(tmp_path):
    s = nova77.load(synth(tmp_path / "a.jsonl", gap=set(range(40, 45)) | set(range(80, 90))))
    c = fill_short_gaps(s)
    assert (c.status[40:45] == INTERPOLATED).all() and np.isfinite(c.xyz[40:45]).all()
    assert (c.status[80:90] == MISSING).all() and np.isnan(c.xyz[80:90]).all()          # long gap is not invented
    obs = s.status == OBSERVED
    assert np.array_equal(c.xyz[obs], s.xyz[obs]) and np.isnan(s.xyz[40:45]).all()      # observed values and raw are untouched
    assert len(main_segment(c)) == 80


@pytest.mark.parametrize("d,view,name", [((1, 0), "Side", "LeftToRight"), ((-1, 0), "Side", "RightToLeft"), ((0, -1), "Front", "TowardCamera"),
                                         ((0, 1), "Front", "AwayFromCamera"), ((1, 1), "Oblique", "Unknown")])
def test_view_direction_and_left_right(tmp_path, d, view, name):
    p = prepare(nova77.load(synth(tmp_path / "a.jsonl", direction=d)))
    assert (p.axes.view, p.axes.direction, p.axes.left_right_ok) == (view, name, True)
    w = in_walk_frame(p.walk.xyz, p.axes)
    assert w[-1, 0, 0] - w[0, 0, 0] == pytest.approx(149 / 30, abs=0.01)                # walks along +forward whatever the camera view
    assert np.allclose(np.linalg.norm(w, axis=-1), np.linalg.norm(p.walk.xyz, axis=-1)) # rotation only


def test_swapped_left_right_is_detected(tmp_path):
    rows = [json.loads(l) for l in open(synth(tmp_path / "a.jsonl"))]
    perm = [nova77.JOINTS.index(n.replace("Left", "@").replace("Right", "Left").replace("@", "Right")) for n in nova77.JOINTS]
    for r in rows:
        for d in r["detections"]:
            d["keypoints_3d"] = [d["keypoints_3d"][k] for k in perm]
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    s = nova77.load(tmp_path / "b.jsonl")
    assert not prepare(s).axes.left_right_ok
    add(tmp_path / "plugins", "any")
    assert run(s, discover(tmp_path / "plugins")[0])[1][0].reasons == ["left_right_contradicts_walking_direction"]


def test_plugins_cannot_modify_shared_input(tmp_path):
    s = nova77.load(synth(tmp_path / "a.jsonl"))
    with pytest.raises(ValueError):
        s.xyz[0, 0, 0] = 1.0


def test_joint_sets():
    assert jointsets.resolve("h36m17", nova77.JOINTS)["unavailable"] == []
    assert jointsets.resolve("kinect13", nova77.JOINTS) | {"direct": None} == {"direct": None, "derived": ["Head"], "unavailable": []}
    assert jointsets.resolve("coco17", nova77.JOINTS)["unavailable"] == ["Nose", "LEar", "REar"]


PLUGIN_SRC = textwrap.dedent('''
    from core.plugin import Manifest, Plugin, Requires
    class P(Plugin):
        manifest = Manifest(id="{id}", version="1", condition="{id}", output="", study="", source="", license="", kind="retrained", fidelity="none",
                            requires=Requires(joint_set="{js}", views=("Side",), min_frames={mf}))
        def adapt(self, p): return p
        def predict(self, x): {body}
    PLUGIN = P
''')


def add(folder, id, body='return {"label": "x", "score": 0.5}', js="kinect13", mf=30):
    (folder / id).mkdir(parents=True, exist_ok=True)
    (folder / id / "plugin.py").write_text(PLUGIN_SRC.format(id=id, body=body, js=js, mf=mf))


def test_body_outside_the_image_is_not_analysed(tmp_path):
    """With a known size the walk is cut to visible frames; if too little is left the plugin refuses with a reason."""
    path, folder = synth(tmp_path / "a.jsonl"), tmp_path / "plugins"
    add(folder, "any")
    plugins, _ = discover(folder)
    s = nova77.load(path)
    assert run(s, plugins)[1][0].status == "ok"                                                  # size unknown: not checked
    lo, hi = float(np.nanmin(s.xy[..., 0])), float(np.nanmax(s.xy[..., 0]))
    assert run(nova77.load(path, size=(int(hi) + 50, 2000)), plugins)[1][0].status == "ok"
    none = run(nova77.load(path, size=(5, 2000)), plugins)[1][0]                                 # nothing is inside a 5 px wide image
    assert none.status == "ineligible" and none.reasons[0].startswith(("view:Unknown", "too_short"))


def test_the_walk_is_cut_to_where_the_whole_body_is_visible(tmp_path):
    path = synth(tmp_path / "a.jsonl", t=200)
    s = nova77.load(path); x = s.xy[..., 0]
    cut = float(np.quantile(x[:, nova77.JOINTS.index("Hips")], 0.5))                     # the image ends in the middle of the walk
    p = prepare(nova77.load(path, size=(int(cut), 5000)))
    q = prepare(nova77.load(path))
    assert 10 < len(p.walk) < len(q.walk)
    assert np.isfinite(p.walk.xy).all() and float(p.walk.xy[:, nova77.JOINTS.index("Hips"), 0].max()) <= cut * 1.01 + 1e-6


def test_size_does_not_change_a_walk_that_is_fully_visible(tmp_path):
    path = synth(tmp_path / "a.jsonl", t=200)
    a, b = prepare(nova77.load(path)), prepare(nova77.load(path, size=(100000, 100000)))
    assert np.array_equal(a.walk.xyz, b.walk.xyz)


def test_plug_in_and_out_without_touching_core(tmp_path):
    s, folder = nova77.load(synth(tmp_path / "a.jsonl")), tmp_path / "plugins"
    add(folder, "disease_a")
    assert [r.plugin for r in run(s, discover(folder)[0])[1]] == ["disease_a"]
    add(folder, "disease_b")
    res = run(s, discover(folder)[0])[1]
    assert [(r.plugin, r.status, r.score) for r in res] == [("disease_a", "ok", 0.5), ("disease_b", "ok", 0.5)]
    (folder / "disease_a" / "plugin.py").unlink()
    assert [r.plugin for r in run(s, discover(folder)[0])[1]] == ["disease_b"]


def test_one_bad_plugin_never_stops_the_others(tmp_path):
    s, folder = nova77.load(synth(tmp_path / "a.jsonl")), tmp_path / "plugins"
    add(folder, "good"); add(folder, "crashes", body='raise RuntimeError("boom")'); add(folder, "nan", body='return {"label": "x", "score": float("nan")}')
    (folder / "broken").mkdir(); (folder / "broken" / "plugin.py").write_text("import does_not_exist")
    plugins, failures = discover(folder)
    assert list(failures) == ["broken"]
    assert {r.plugin: r.status for r in run(s, plugins)[1]} == {"good": "ok", "crashes": "error", "nan": "error"}


def test_unmet_requirements_give_reasons_not_guesses(tmp_path):
    folder = tmp_path / "plugins"
    add(folder, "needs_nose", js="coco17"); add(folder, "needs_long", mf=1000); add(folder, "ok")
    plugins, _ = discover(folder)
    side = {r.plugin: r for r in run(nova77.load(synth(tmp_path / "a.jsonl")), plugins)[1]}
    assert side["needs_nose"].status == "ineligible" and "Nose" in side["needs_nose"].reasons[0]
    assert side["needs_long"].reasons == ["too_short:150<1000_frames"] and side["ok"].status == "ok"
    front = {r.plugin: r for r in run(nova77.load(synth(tmp_path / "b.jsonl", direction=(0, -1))), plugins)[1]}
    assert front["ok"].status == "ineligible" and front["ok"].reasons[0].startswith("view:Front")
