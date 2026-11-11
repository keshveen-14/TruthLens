"""Build the web front end's data file and a single-file copy of the site.

    python scripts/build_web.py

web/data.js     the sample corpus and questions, so the browser engine works with no server
dist/truthlens.html   the whole site in one file: open it, host it anywhere (GitHub Pages, Netlify, a USB stick)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def build_data() -> None:
    docs = [{"name": p.name, "text": p.read_text(encoding="utf-8").strip()} for p in sorted((ROOT / "data" / "corpus").glob("*.md"))]
    qs = [json.loads(l) for l in (ROOT / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    (WEB / "data.js").write_text("window.TL_DATA = " + json.dumps({"docs": docs, "questions": qs}, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")


def build_single() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="/styles.css">', "<style>\n" + css + "\n</style>")
    for name in ("data.js", "engine.js", "app.js"):
        js = (WEB / name).read_text(encoding="utf-8").replace("</script", "<\\/script")
        html = html.replace(f'<script src="/{name}" defer></script>', "<script>\n" + js + "\n</script>")
    (ROOT / "dist").mkdir(exist_ok=True)
    # inline scripts must come after the markup they query, so move them to the end of <body>
    import re
    scripts = "".join(re.findall(r"<script>.*?</script>\n?", html, flags=re.S))
    html = re.sub(r"<script>.*?</script>\n?", "", html, flags=re.S).replace("</body>", scripts + "</body>")
    html = html.replace('<link rel="icon" href="/favicon.svg" type="image/svg+xml">', "")
    (ROOT / "dist" / "truthlens.html").write_text(html, encoding="utf-8")
    # hosted-page variant: content only (title, styles, body markup, scripts)
    title = re.search(r"<title>.*?</title>", html).group(0)
    meta = '<meta name="claude:viewport-fit-sides" content="cover">'
    fonts = re.search(r'<link rel="stylesheet" href="https://fonts[^>]+>', html).group(0)
    style = re.search(r"<style>.*?</style>", html, flags=re.S).group(0)
    body = re.search(r"<body>(.*)</body>", html, flags=re.S).group(1)
    (ROOT / "dist" / "truthlens-page.html").write_text("\n".join([title, meta, fonts, style, body]), encoding="utf-8")


if __name__ == "__main__":
    build_data()
    build_single()
    print("wrote web/data.js and dist/truthlens.html")
