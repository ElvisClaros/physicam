"""Video probing and transcoding via the bundled ffmpeg (imageio-ffmpeg)."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image


def ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class VideoInfo:
    """Container/stream facts about a video file."""

    fps: float
    n_frames: int
    width: int
    height: int
    duration_s: float
    rotation: int = 0
    codec: str = ""

    def to_dict(self) -> dict:
        return {"fps": self.fps, "n_frames": self.n_frames, "width": self.width,
                "height": self.height, "duration_s": self.duration_s,
                "rotation": self.rotation, "codec": self.codec}


def _parse_rate(value: str) -> float:
    value = value.strip()
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_ffmpeg_stderr(text: str, path: str | Path) -> VideoInfo:
    duration_s = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
    if m:
        duration_s = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    width = height = 0
    codec = ""
    fps = 0.0
    rotation = 0
    # Rotation: displaymatrix side data or rotate metadata tag.
    mr = re.search(r"displaymatrix:\s*rotation of\s*(-?[\d.]+)\s*degrees", text)
    if mr:
        try:
            rotation = int(round(float(mr.group(1)))) % 360
        except ValueError:
            rotation = 0
    mr2 = re.search(r"rotate\s*:\s*(-?\d+)", text)
    if mr2:
        try:
            rotation = int(mr2.group(1)) % 360
        except ValueError:
            pass
    for line in text.splitlines():
        if "Video:" not in line:
            continue
        cm = re.search(r"Video:\s*(\S+)", line)
        if cm:
            codec = cm.group(1)
        dm = re.search(r"(\d{2,5})x(\d{2,5})", line)
        if dm:
            width, height = int(dm.group(1)), int(dm.group(2))
        # Prefer avg_frame_rate / r_frame_rate over tbr. ffmpeg -i prints e.g.
        # "240 fps, 240 tbr" — the token labelled fps is the avg rate in practice;
        # when explicit "avg_frame_rate" appears (ffprobe-style) it wins.
        rates: list[tuple[str, float]] = []
        for rm in re.finditer(r"([\d.]+(?:/\d+)?)\s*(fps|tbr)", line):
            rates.append((rm.group(2), _parse_rate(rm.group(1))))
        for rm in re.finditer(r"(avg_frame_rate|r_frame_rate)\s*[:=]\s*([\d./]+)", line):
            rates.append((rm.group(1), _parse_rate(rm.group(2))))
        for label, val in rates:
            if label in ("avg_frame_rate", "r_frame_rate") and val > 0:
                fps = val
        if fps == 0.0:
            for label, val in rates:
                if label == "fps" and val > 0:
                    fps = val
                    break
        if fps == 0.0:
            for label, val in rates:
                if val > 0:
                    fps = val
                    break
        break
    if rotation in (90, 270):
        width, height = height, width
    n_frames = int(round(fps * duration_s)) if fps > 0 and duration_s > 0 else 0
    return VideoInfo(fps=fps, n_frames=n_frames, width=width, height=height,
                     duration_s=duration_s, rotation=rotation, codec=codec)


def probe(path: str | Path) -> VideoInfo:
    """Probe *path* by parsing ``ffmpeg -i`` stderr. Cheap; no decode."""
    p = Path(path)
    proc = subprocess.run([ffmpeg_exe(), "-i", str(p)], capture_output=True, text=True, timeout=60)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    info = _parse_ffmpeg_stderr(text, p)
    if info.fps <= 0 or info.width <= 0:
        raise RuntimeError(f"could not probe video {p}: {text[-500:]}")
    return info


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {(proc.stderr or '')[-2000:]}")


def _rotation_filter(rotation: int) -> list[str]:
    # Input streams are auto-rotated by ffmpeg on decode in most builds; be
    # explicit so stored rotation never leaks into measurements.
    return {90: ["transpose=1"], 180: ["hflip", "vflip"], 270: ["transpose=2"]}.get(rotation % 360, [])


def normalize(src: str | Path, dst: str | Path, max_side: int = 1280,
              t_start: float | None = None, t_end: float | None = None) -> VideoInfo:
    """Transcode to H.264 yuv420p CFR at source fps, max *max_side* long side, +faststart."""
    info = probe(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    vf: list[str] = _rotation_filter(info.rotation)
    vf.append(f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease")
    vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    vf.append(f"fps={info.fps}")
    args = [ffmpeg_exe(), "-y"]
    if t_start is not None:
        args += ["-ss", f"{t_start}"]
    args += ["-i", str(src)]
    if t_end is not None:
        dur = t_end - (t_start or 0.0)
        args += ["-t", f"{dur}"]
    args += ["-vf", ",".join(vf), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-r", f"{info.fps}", "-movflags", "+faststart", str(dst)]
    _run_ffmpeg(args)
    return probe(dst)


def make_proxy(src: str | Path, dst: str | Path) -> VideoInfo:
    """Downscaled proxy for Spark: 30 fps, 640 px wide, keeps audio if present."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    args = [ffmpeg_exe(), "-y", "-i", str(src),
            "-vf", "scale=640:-2,fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    _run_ffmpeg(args)
    return probe(dst)


def extract_frame(path: str | Path, t: float) -> Image.Image:
    """Extract a single frame at time *t* seconds as a PIL image."""
    import io

    proc = subprocess.run(
        [ffmpeg_exe(), "-y", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"frame extraction failed at t={t}: {(proc.stderr or b'')[-500:]!r}")
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def extract_frames(path: str | Path, times: list[float]) -> list[Image.Image]:
    """Extract frames at each time in *times*."""
    return [extract_frame(path, t) for t in times]


def contact_sheet(path: str | Path, n: int = 6) -> Image.Image:
    """Tile *n* evenly spaced frames into one contact-sheet image."""
    info = probe(path)
    times = [info.duration_s * (i + 0.5) / n for i in range(n)] if info.duration_s > 0 else [0.0] * n
    frames = extract_frames(path, times)
    w, h = frames[0].size
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, fr in enumerate(frames):
        sheet.paste(fr, ((i % cols) * w, (i // cols) * h))
    return sheet
