"""Build ``site/``: a fully static Physicam demo for any static host.

Copies ``web/`` plus each finished example job (result.json, events.jsonl,
video.mp4, strobe.jpg, thumb.jpg) under ``site/examples/<slug>/``, rewrites
the example result/event URLs to relative paths, and writes
``site/static-examples.json``. Works under a sub-path such as
``https://user.github.io/physicam/`` because every URL is relative.

Run: ``.venv/bin/python scripts/build_static.py``
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
ASSETS = ("result.json", "events.jsonl", "video.mp4", "strobe.jpg", "thumb.jpg")


def _rel(slug: str, name: str) -> str:
    return f"examples/{slug}/{name}"


def main() -> None:
    """Build the static site into site/."""
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(ROOT / "web", SITE, ignore=shutil.ignore_patterns("mock", "README.md"))
    index = json.loads((ROOT / "examples" / "examples.json").read_text())
    out = []
    for ex in index:
        slug = ex["slug"]
        src, dst = ROOT / "examples" / "jobs" / slug, SITE / "examples" / slug
        dst.mkdir(parents=True, exist_ok=True)
        for name in ASSETS:
            p = src / name
            if p.exists():
                shutil.copy2(p, dst / name)
        result = json.loads((dst / "result.json").read_text())
        if result.get("video"):
            result["video"]["url"] = _rel(slug, "video.mp4")
        if result.get("strobe_url"):
            result["strobe_url"] = _rel(slug, "strobe.jpg")
        result["thumb_url"] = _rel(slug, "thumb.jpg")
        (dst / "result.json").write_text(json.dumps(result, indent=1))
        ev = dst / "events.jsonl"
        if ev.exists():
            ev.write_text(ev.read_text().replace(f"/media/{slug}/", f"examples/{slug}/")
                          .replace(f"/api/jobs/{slug}/thumb.jpg", _rel(slug, "thumb.jpg")))
        out.append({"id": slug, "slug": slug, "title": ex["title"], "subtitle": ex["subtitle"],
                    "question": ex["question"], "headline": ex.get("headline", ""),
                    "thumb_url": _rel(slug, "thumb.jpg"), "result_url": _rel(slug, "result.json"),
                    "events_url": _rel(slug, "events.jsonl")})
    (SITE / "static-examples.json").write_text(json.dumps(out, indent=1))
    print(f"built {SITE} with {len(out)} examples")


if __name__ == "__main__":
    main()
