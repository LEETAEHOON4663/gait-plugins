"""Which normalisation makes TOAGA look like the study's training input? Judged only from the input statistics stored in the
checkpoints (BatchNorm of the first layer), never from scores. Usage: toaga_norm_check.py <PoseTracking> <repo>"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scripts.study_code import JOINTS, StudyModel
from scripts.toaga import SIZE, index, read, study_input
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
idx = index(sys.argv[1]); L, R = JOINTS.index("LHip"), JOINTS.index("RHip")
SW = [0] + [k + 1 if k % 2 else k - 1 for k in range(1, 13)]
for eng in ("alphapose", "detectron"):
    sm = StudyModel(sys.argv[2], eng)
    mu = np.mean([m.encoder.data_bn.running_mean.numpy().reshape(13, 2) for m in sm.models], 0); var = np.mean([m.encoder.data_bn.running_var.numpy().reshape(13, 2) for m in sm.models], 0)
    walks = {k: v[eng] for k, v in idx.items() if eng in v and k[2] == "front"}
    data = []
    for p in walks.values():
        t, xy, c = read(p); x, _ = study_input(xy, time=t)
        if np.isfinite(x).all(): data.append(x)
    print(f"\n{eng}: study training input, mean position (normalised): hips ({(mu[L,0]+mu[R,0])/2:+.3f}, {(mu[L,1]+mu[R,1])/2:+.3f}) ankles y {(mu[11,1]+mu[12,1])/2:+.3f} nose y {mu[0,1]:+.3f}; body height {(mu[11,1]+mu[12,1])/2-mu[0,1]:.3f}")
    for name, res, scale in (("portrait rule 1080x1920", (1080, 1920), 1.0), ("landscape 1920x1080, pixels as they are", (1920, 1080), 1.0)):
        # training included zero padding (value 0) and the left/right flip; add both, then compare real and padded statistics
        X = [np.stack([x[..., 0] / res[0] - .5, x[..., 1] / res[1] - .5], -1) for x in data]
        a = np.concatenate(X); b = a.copy(); b[..., 0] = a[:, SW, 0]; a = np.concatenate([a, b])
        # the hips of every real frame sit at (100/res - .5); padded frames are 0: q from the stored hip mean in the training convention
        tr = (100 / 1920 - .5, 100 / 1080 - .5); q = 1 - ((mu[L] + mu[R]) / 2 / np.array(tr)).mean()
        m_r = mu / (1 - q); sd = np.sqrt(np.maximum((var + mu ** 2) / (1 - q) - m_r ** 2, 1e-6))
        s = (a.mean(0) - m_r) / sd; r = a.std(0) / sd
        sc = [np.mean(sm.scores(x, np.ones(x.shape[:2]), res)) for x in data]
        print(f"  {name:40s} |shift| x {np.median(np.abs(s[:,0])):5.2f} SD  y {np.median(np.abs(s[:,1])):5.2f} SD | spread ratio x {np.median(r[:,0]):.2f} y {np.median(r[:,1]):.2f} | body height {(a[:,[11,12],1].mean()-a[:,0,1].mean()):.3f} | SCORES mean {np.mean(sc):.2f} SD between walks {np.std(sc):.2f}  (padded share q={q:.2f})")
    h = [np.median((x[:, 11, 1] + x[:, 12, 1]) / 2 - x[:, 0, 1]) for x in data]
    print(f"  body height in pixels on TOAGA: median {np.median(h):.0f} (study training, 1080 px image: {((mu[11,1]+mu[12,1])/2-mu[0,1])/(1-q)*1080:.0f})")
