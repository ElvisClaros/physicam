"""Run the full pipeline on every clip in ``examples/media/``.

Built-in clips use the questions below. Any extra ``*.mp4`` in
``examples/media/`` with a sidecar ``<name>.json`` of
``{title, subtitle, question, facts}`` is built the same way (slug = file stem).

Each run lands in ``examples/jobs/<slug>/`` (result.json, events.jsonl,
video.mp4, strobe.jpg, thumb.jpg), the video is re-encoded to <= 6 MB
(CRF 28, dimensions kept so track coordinates stay valid), and
``examples/examples.json`` is written for ``GET /api/examples``.

Run: ``.venv/bin/python examples/build_examples.py`` (set
``PHYSICAM_OFFLINE=1`` to replay from the SAM/plan disk cache).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from physicam.pipeline import run_job  # noqa: E402
from physicam.video import extract_frame, ffmpeg_exe, probe  # noqa: E402

ROOT = Path(__file__).resolve().parent
MEDIA, JOBS = ROOT / "media", ROOT / "jobs"
BUILTIN = {
    "bouncy-ball-240fps": {
        "title": "Bouncing ball · 240 fps",
        "subtitle": "Coefficient of restitution + g (pixel units, no ruler in frame)",
        "question": "What is the coefficient of restitution of this ball, and what does the throw tell us about g?",
        "facts": {},
    },
    "elastic-pendulum": {
        "title": "Spring pendulum · 30 fps",
        "subtitle": "Oscillation period of a mass on a spring (portrait phone video)",
        "question": "What is the oscillation period of this spring pendulum?",
        "facts": {},
    },
}


def _shrink(video: Path) -> None:
    """Re-encode *video* in place at CRF 28 until it is <= 6 MB."""
    if video.stat().st_size <= 6_000_000:
        # Still normalise encoding for the committed copy.
        tmp = video.with_suffix(".tmp.mp4")
        subprocess.run([ffmpeg_exe(), "-y", "-i", str(video), "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart", str(tmp)],
                       check=True, capture_output=True, timeout=1200)
        tmp.replace(video)
        return
    for crf in (28, 32, 36):
        tmp = video.with_suffix(".tmp.mp4")
        subprocess.run([ffmpeg_exe(), "-y", "-i", str(video), "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(tmp)],
                       check=True, capture_output=True, timeout=1200)
        tmp.replace(video)
        if video.stat().st_size <= 6_000_000:
            return


def _specs() -> list[dict]:
    specs = []
    for mp4 in sorted(MEDIA.glob("*.mp4")):
        stem = mp4.stem
        if stem in BUILTIN:
            specs.append({"slug": stem, "video": mp4, **BUILTIN[stem]})
            continue
        side = mp4.with_suffix(".json")
        if side.exists():
            d = json.loads(side.read_text())
            facts = dict(d.get("facts") or {})
            if d.get("reference_csv"):
                # Resolve relative to the repo root (examples/..).
                ref = ROOT.parent / str(d["reference_csv"])
                facts.setdefault("reference_csv", str(ref))
            specs.append({"slug": stem, "video": mp4, "title": d.get("title", stem),
                          "subtitle": d.get("subtitle", ""), "question": d["question"],
                          "facts": facts})
    return specs


def main() -> None:
    """Build every example and write examples.json."""
    JOBS.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in _specs():
        slug, job_dir = spec["slug"], JOBS / spec["slug"]
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True)
        events: list[dict] = []
        result = run_job(job_dir, spec["video"], spec["question"], spec["facts"],
                         emit=events.append)
        if result.get("status") != "done":
            print(f"{slug}: FAILED {result.get('error')}")
            continue
        video = job_dir / "video.mp4"
        _shrink(video)
        try:
            extract_frame(video, 0.0).save(job_dir / "thumb.jpg", format="JPEG", quality=85)
        except Exception as exc:
            print(f"{slug}: thumb failed: {exc}")
        ans = result.get("answer") or {}
        print(f"{slug}: {ans.get('headline')}  video={video.stat().st_size/1e6:.2f} MB "
              f"probe={probe(video)}")
        index.append({"id": slug, "slug": slug, "job_id": slug, "title": spec["title"],
                      "subtitle": spec["subtitle"], "question": spec["question"],
                      "thumb_url": f"/api/jobs/{slug}/thumb.jpg",
                      "headline": ans.get("headline", "")})
    (ROOT / "examples.json").write_text(json.dumps(index, indent=1))
    print(f"wrote examples/examples.json ({len(index)} examples)")


if __name__ == "__main__":
    main()
