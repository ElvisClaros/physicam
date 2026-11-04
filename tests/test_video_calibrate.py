"""Video probe parsing + calibration helpers (no network)."""

import numpy as np

from physicam.calibrate import from_gravity, from_track_size, none
from physicam.sam import Track
from physicam.video import _parse_ffmpeg_stderr

SAMPLE_STDERR = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Duration: 00:00:04.00, start: 0.003971, bitrate: 10372 kb/s
  Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1280x720 [SAR 1:1 DAR 16:9], 10347 kb/s, 240 fps, 240 tbr, 15360 tbn (default)
"""


def test_probe_parse_prefers_fps_and_computes_frames():
    info = _parse_ffmpeg_stderr(SAMPLE_STDERR, "clip.mp4")
    assert info.width == 1280 and info.height == 720
    assert info.fps == 240.0
    assert info.duration_s == 4.0
    assert info.n_frames == 960
    assert info.codec == "h264"


def test_calibrate_from_track_size_and_gravity():
    n = 50
    track = Track(id="t0", label="ball", frames=np.arange(n), t=np.arange(n) / 240.0,
                  x=np.zeros(n), y=np.zeros(n), area=np.full(n, np.pi * 25.0 ** 2),
                  bbox=np.zeros((n, 4)), conf=np.ones(n),
                  eq_diameter=np.full(n, 50.0), contours={}, fps=240.0)
    cal = from_track_size(track, 0.05)
    assert abs(cal.m_per_px - 0.001) < 1e-9
    assert cal.rel_uncertainty >= 0.03
    gcal = from_gravity(8000.0)
    assert abs(gcal.m_per_px - 9.81 / 8000.0) < 1e-12
    assert none().method == "none"
