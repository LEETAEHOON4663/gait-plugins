"""One skeleton file in, one report out. Usage: run.py <file.jsonl> [...] [--plugins DIR] [--json]"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import nova77
from core.plugin import PLUGIN_DIR, discover, run

KO = {"ok": "", "ineligible": "판단 불가", "error": "오류"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--plugins", default=str(PLUGIN_DIR)); ap.add_argument("--json", action="store_true")
    ap.add_argument("--size", help="frame size the 2D joints refer to, e.g. 1280x720; without it the whole-body check is skipped")
    a = ap.parse_args()
    plugins, failures = discover(a.plugins)
    reports = []
    for f in a.files:
        p, results = run(nova77.load(f, size=tuple(map(int, a.size.split("x"))) if a.size else None), plugins)
        reports.append({"file": Path(f).name, "sha256": p.raw.source["sha256"], "view": p.axes.view, "direction": p.axes.direction, "frames": len(p.raw),
                        "walk_frames": len(p.walk), "filled_ratio": round(float((p.walk.status == 2).mean()), 3) if len(p.walk) else None,
                        "results": [r.to_dict() for r in results], "plugins_not_loaded": failures})
    if a.json:
        print(json.dumps(reports, ensure_ascii=False, indent=1)); return
    for r in reports:
        print(f"\n=== {r['file']}  방향={r['view']}/{r['direction']}  프레임={r['frames']} 분석구간={r['walk_frames']} 보간={r['filled_ratio']}")
        for x in r["results"]:
            if x["status"] == "ok":
                print(f"  [{x['plugin']}] {x['condition']}: {x['label']} (점수 {x['score']:.2f})" + (f"  주의: {', '.join(x['warnings'])}" if x["warnings"] else ""))
                for e in x["detail"].get("evidence", []):
                    print(f"       근거 {e['feature']} = {e['value']} ({e['push']:+.2f})")
            else:
                print(f"  [{x['plugin']}] {x['condition']}: {KO[x['status']]} - {x['reasons'][0]}")
        for k, v in r["plugins_not_loaded"].items():
            print(f"  [{k}] 불러오기 실패: {v}")
    print("\n연구용 시연입니다. 진단이 아니며, 서로 다른 플러그인의 점수는 비교할 수 없습니다.")


if __name__ == "__main__":
    main()
