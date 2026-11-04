"""Download the CC-licensed demo clips from Wikimedia Commons.

Uses the Commons API (User-Agent ``physicam/0.1 (hackathon demo)``) to resolve
each file's original URL, downloads it, and transcodes to H.264 MP4 in
``examples/media/`` via :func:`physicam.video.normalize`. Also (re)writes
``examples/ATTRIBUTION.md`` from the API's license metadata.

Run: ``.venv/bin/python examples/fetch_examples.py``
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from physicam.video import normalize  # noqa: E402

UA = "physicam/0.1 (hackathon demo)"
MEDIA = Path(__file__).resolve().parent / "media"
# title, output stem, segment (t_start, t_end) or None for full clip
CLIPS = [
    ("File:Bouncy ball 240fps.webm", "bouncy-ball-240fps", (6.0, 10.0)),
    ("File:ElasticPendulum.webm", "elastic-pendulum", None),
]


def _api_imageinfo(title: str) -> dict:
    """Return the Commons imageinfo record for *title*."""
    import urllib.parse

    q = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "format": "json"})
    req = urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{q}",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as fh:
        data = json.load(fh)
    pages = data["query"]["pages"]
    return next(iter(pages.values()))["imageinfo"][0]


def _meta(info: dict, key: str) -> str:
    try:
        return str(info["extmetadata"][key]["value"]).strip()
    except Exception:
        return ""


def main() -> None:
    """Download, transcode and attribute every clip in CLIPS."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    rows = []
    for title, stem, seg in CLIPS:
        info = _api_imageinfo(title)
        url = info["url"]
        raw = MEDIA / f"{stem}.source.webm"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as fh, open(raw, "wb") as out:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        dst = MEDIA / f"{stem}.mp4"
        if seg:
            normalize(raw, dst, t_start=seg[0], t_end=seg[1])
        else:
            normalize(raw, dst)
        raw.unlink(missing_ok=True)
        # Strip HTML tags Commons wraps around Artist values.
        import re
        author = re.sub(r"<[^>]+>", "", _meta(info, "Artist")) or "Wikimedia Commons contributor"
        rows.append((title, author, _meta(info, "LicenseShortName") or "CC BY-SA 4.0",
                     info.get("descriptionurl", url), stem))
        print(f"{stem}: {dst} ({dst.stat().st_size/1e6:.1f} MB)")
    lines = ["# Example media attribution", "",
             "All clips below are from Wikimedia Commons and remain under their "
             "original licences. Physicam's code is MIT (see `../LICENSE`).", ""]
    for title, author, lic, url, stem in rows:
        lines += [f"## {title}", "", f"- File: `media/{stem}.mp4`",
                  f"- Author: {author}", f"- Licence: {lic}", f"- Source: {url}", ""]
    (MEDIA.parent / "ATTRIBUTION.md").write_text("\n".join(lines))
    print("wrote examples/ATTRIBUTION.md")


if __name__ == "__main__":
    main()
