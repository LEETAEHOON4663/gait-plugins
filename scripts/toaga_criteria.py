"""What counts as "the same as the study" - fixed on 2026-11-22, before any Nova-77 result on TOAGA existed.

The study's own model gives different scores when only the pose extractor changes (toaga_baseline.py). Nova-77 is judged
against that: it passes if it changes the study's scores no more than swapping AlphaPose for Detectron does, allowing for
sampling error (the upper 95% bound of that swap, resampling people, not walks). At least 85% of the usable walks must be found.

Revised once, still before any Nova-77 data: comparing with the point value made the extractor that DEFINES the yardstick fail
half the time (toaga_selftest.py), so the sampling bound replaces it.
"""
import numpy as np

MIN_WALKS, MIN_COVERAGE = 30, 0.85


def stats(ref, new):
    from scipy.stats import spearmanr
    c = lambda x: np.clip(np.round(x), 0, 2)
    return dict(diff=float(np.abs(ref - new).mean()), bias=float((new - ref).mean()), cls=float(np.mean(c(ref) != c(new))),
                rho=float(spearmanr(ref, new).statistic) if len(ref) > 2 else np.nan)


def summarise(ref, new, people):
    """Numbers of one comparison plus the bounds used by the verdict (bootstrap over people)."""
    ref, new, people = map(np.asarray, (ref, new, people)); ok = np.isfinite(ref) & np.isfinite(new)
    ref, new, people = ref[ok], new[ok], people[ok]
    if len(ref) < 3: return dict(n=len(ref), diff=np.nan, bias=np.nan, cls=np.nan, rho=np.nan, diff_hi=np.nan, cls_hi=np.nan, rho_lo=np.nan)
    out = dict(n=len(ref), **stats(ref, new)); rng, who, bs = np.random.default_rng(0), np.unique(people), []
    for _ in range(1000):
        i = np.concatenate([np.flatnonzero(people == p) for p in rng.choice(who, len(who))]); bs.append(stats(ref[i], new[i]))
    out["diff_hi"], out["cls_hi"] = (float(np.nanpercentile([b[k] for b in bs], 95)) for k in ("diff", "cls"))
    out["rho_lo"] = float(np.nanpercentile([b["rho"] for b in bs], 5))
    return out


def verdict(new: dict, yard: dict, coverage: float = 1.0) -> tuple:
    if new["n"] < MIN_WALKS:
        return None, [f"only {new['n']} walks (needs {MIN_WALKS})"]
    checks = [(coverage >= MIN_COVERAGE, f"walks found {coverage:.0%} >= {MIN_COVERAGE:.0%}"),
              (new["diff"] <= yard["diff_hi"], f"mean |diff| {new['diff']:.2f} <= {yard['diff_hi']:.2f} (swap {yard['diff']:.2f}, 95% bound)"),
              (new["cls"] <= yard["cls_hi"], f"class changes {new['cls']:.0%} <= {yard['cls_hi']:.0%} (swap {yard['cls']:.0%})"),
              (new["rho"] >= yard["rho_lo"], f"rank correlation {new['rho']:.2f} >= {yard['rho_lo']:.2f} (swap {yard['rho']:.2f})")]
    return all(ok for ok, _ in checks), [("ok   " if ok else "FAIL ") + s for ok, s in checks]
