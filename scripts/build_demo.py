"""Put results/demo.json into demo/template.html. Usage: build_demo.py  (writes demo/index.html)"""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
data = (root / "results/demo.json").read_text().replace("</", "<\\/")
page = (root / "demo/template.html").read_text().replace("/*DATA*/", data)
(root / "demo/index.html").write_text('<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n' + page.replace('<div class="wrap">', '</head><body>\n<div class="wrap">', 1) + "\n</body></html>\n")
print((root / "demo/index.html").stat().st_size)
