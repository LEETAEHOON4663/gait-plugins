"""Download TOAGA (Figshare, CC0), check size and MD5, and unpack it.
Usage: fetch_toaga.py <folder> [--videos]

Zips go to <folder>/download and are unpacked to <folder>/x/<name> (e.g. x/PoseTracking, x/Xsens). The videos (2.35 GB) are
only needed to run the NVIDIA extractor on TOAGA, so they are skipped unless --videos is given. Finished files are skipped.
"""
import hashlib, json, pathlib, sys, urllib.request, zipfile

root = pathlib.Path(sys.argv[1]); out = root / "download"; out.mkdir(parents=True, exist_ok=True)
articles = [14997816, 19929884, 19929890, 16641025]              # pose tracks, Xsens, participants, gait-variable code
if "--videos" in sys.argv:
    articles.insert(0, 19929893)                                   # videos
bad = 0
for i in articles:
    d = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{i}", timeout=60))
    for f in d["files"]:
        p = out / f["name"].replace("/", "_").replace(" ", "_")
        if not (p.exists() and p.stat().st_size == f["size"]):
            part = p.with_name(p.name + ".part")
            with urllib.request.urlopen(f["download_url"], timeout=120) as r, open(part, "wb") as w:
                while b := r.read(1 << 20):
                    w.write(b)
            part.rename(p)
        h = hashlib.md5()
        with open(p, "rb") as r:
            while b := r.read(1 << 20):
                h.update(b)
        ok = p.stat().st_size == f["size"] and h.hexdigest() == (f.get("computed_md5") or f.get("supplied_md5"))
        bad += not ok
        print(f'{p.name:22s} {p.stat().st_size / 1e6:9.1f} MB  checksum {"OK" if ok else "MISMATCH"}  license {d["license"]["name"]}', flush=True)
        if ok and p.suffix == ".zip" and not (root / "x" / p.stem).is_dir():
            dest = root / "x" / p.stem
            with zipfile.ZipFile(p) as z:
                for m in z.infolist():
                    parts = pathlib.PurePosixPath(m.filename).parts
                    if m.filename.startswith("/") or ".." in parts:
                        raise SystemExit(f"unsafe path in {p.name}: {m.filename}")
                    if not m.is_dir() and "__MACOSX" not in parts:
                        z.extract(m, dest)
            print(f"{'':22s} unpacked to {dest}", flush=True)
print("DONE" if not bad else f"{bad} FILES BAD", flush=True)
sys.exit(bool(bad))
