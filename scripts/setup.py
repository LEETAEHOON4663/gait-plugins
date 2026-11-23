"""Get the two study repositories at the tested commits and put their published weights into the plugins.

Usage: python scripts/setup.py [third_party folder]      (default: ./third_party)

Downloads about 330 MB with git. The weights stay under the studies' own licences and are never committed here
(see .gitignore). Run it again at any time: finished steps are skipped.
"""
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPOS = {   # folder: (url, commit the plugins were tested with)
    "stgcn_parkinsonism_prediction": ("https://github.com/TaatiTeam/stgcn_parkinsonism_prediction.git", "8474ae57655aebe65fadfd5c63c0261a193f852f"),
    "mobile-gaitlab": ("https://github.com/stanfordnmbl/mobile-gaitlab.git", "4167ee6213ac81fb9168f4e20c6538d42a41a5a6"),
}
ZOO = "sample_data/model_zoo"
# Sabo et al. 2022: the 3D model and the ten cross-validation folds of each 2D model, as named in the study's model zoo
STGCN = [(f"{ZOO}/kinect3d_models_to_share/v2/13/0.1/1/all/(0,0,0)/gait_feats_False/checkpoint_final.pt", "plugins/pd_stgcn_sabo2022/weights/checkpoint_final.pt")]
STGCN += [(f"{ZOO}/alphapose_models_to_share/v0/13/0.2/{k}/checkpoint_final.pt", f"plugins/pd_stgcn_sabo2022_ap2d/weights/{k}.pt") for k in range(1, 11)]
STGCN += [(f"{ZOO}/detectron_models_to_share/v2/13/0.2/{k}/checkpoint_final.pt", f"plugins/pd_stgcn_sabo2022_dt2d/weights/{k}.pt") for k in range(1, 11)]
GAITLAB_NPZ = ROOT / "plugins/gait_gaitlab2020/weights/gaitlab.npz"


def git(*args):
    subprocess.run(["git", *args], check=True)


def clone(folder, url, commit):
    if not (folder / ".git").is_dir():
        print(f"받는 중: {url}")
        git("clone", "--quiet", url, str(folder))
    if subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() != commit:
        git("-C", str(folder), "fetch", "--quiet", "origin", commit)
        git("-C", str(folder), "checkout", "--quiet", commit)
    print(f"  {folder.name} @ {commit[:10]}")


def main():
    third = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "third_party"
    third.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in REPOS.items():
        clone(third / name, url, commit)

    stgcn = third / "stgcn_parkinsonism_prediction"
    for src, dst in STGCN:
        src, dst = stgcn / src, ROOT / dst
        if not src.is_file():
            raise SystemExit(f"가중치가 없습니다: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or hashlib.sha256(dst.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
            shutil.copyfile(src, dst)
    print(f"  ST-GCN 가중치 {len(STGCN)}개 설치 (3D 1개, AlphaPose 2D 10개, Detectron 2D 10개)")

    spec = importlib.util.spec_from_file_location("gaitlab_model", ROOT / "plugins/gait_gaitlab2020/model.py")
    gaitlab = importlib.util.module_from_spec(spec); spec.loader.exec_module(gaitlab)     # reads the authors' Keras weights once, writes one .npz
    GAITLAB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    gaitlab.convert(third / "mobile-gaitlab/demo/models", GAITLAB_NPZ)
    print(f"  mobile-gaitlab 가중치 변환: {GAITLAB_NPZ.relative_to(ROOT)}")
    print("\n끝. 다음: python -m pytest -q")


if __name__ == "__main__":
    main()
