"""Parser on the real SAM fixture + stitching on synthetic tracks."""

import gzip
from pathlib import Path

import numpy as np

from physicam.sam import Track, parse_sam_text, stitch_tracks

FIXTURE = Path(__file__).parent / "fixtures" / "bounce240_red_ball_f0-199.sam.txt.gz"


def test_parse_real_fixture_centroids_inside_boxes():
    text = gzip.open(FIXTURE, "rt").read()
    tracks = parse_sam_text(text, fps=240.0, label="red ball")
    assert len(tracks) >= 1
    for tr in tracks:
        assert tr.n > 50
        for i in range(tr.n):
            x1, y1, x2, y2 = tr.bbox[i]
            assert x1 - 1 <= tr.x[i] <= x2 + 1, (i, tr.x[i], x1, x2)
            assert y1 - 1 <= tr.y[i] <= y2 + 1, (i, tr.y[i], y1, y2)
            assert tr.area[i] > 0
            assert 0 < tr.eq_diameter[i] < 200


def test_contour_transform_lies_within_bbox():
    """Frame-coordinate contours must fall inside their frame's bbox."""
    import re

    text = gzip.open(FIXTURE, "rt").read()
    tracks = parse_sam_text(text, fps=240.0, label="red ball")
    assert tracks and tracks[0].contours
    bbox_by_frame = {int(f): tracks[0].bbox[i] for i, f in enumerate(tracks[0].frames)}
    checked = 0
    for frame, path in tracks[0].contours.items():
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path)]
        xs, ys = nums[0::2], nums[1::2]
        x1, y1, x2, y2 = bbox_by_frame[frame]
        assert min(xs) >= x1 - 2 and max(xs) <= x2 + 2, (frame, min(xs), max(xs), x1, x2)
        assert min(ys) >= y1 - 2 and max(ys) <= y2 + 2, (frame, min(ys), max(ys), y1, y2)
        checked += 1
    assert checked >= 10


def _synth_track(track_id, frames, xs, ys, fps=30.0, diam=40.0):
    frames = np.asarray(frames, int)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = len(frames)
    area = np.full(n, np.pi * (diam / 2) ** 2)
    bbox = np.column_stack([xs - diam / 2, ys - diam / 2, xs + diam / 2, ys + diam / 2])
    return Track(id=track_id, label="obj", frames=frames, t=frames / fps,
                 x=xs, y=ys, area=area, bbox=bbox, conf=np.full(n, 0.9),
                 eq_diameter=np.full(n, diam), contours={}, fps=fps)


def test_stitch_merges_non_overlapping_consistent():
    fps = 30.0
    # Object moves right at 60 px/s, brief gap (frames 10..12 missing => ids split).
    a = _synth_track("t0", list(range(0, 10)), [100 + 2 * i for i in range(10)], [200] * 10, fps)
    b = _synth_track("t1", list(range(13, 23)), [100 + 2 * i for i in range(13, 23)], [200] * 10, fps)
    out = stitch_tracks([a, b], fps)
    assert len(out) == 1
    assert out[0].n == 20


def test_stitch_keeps_distant_tracks_separate():
    fps = 30.0
    a = _synth_track("t0", list(range(0, 10)), [100] * 10, [200] * 10, fps)
    b = _synth_track("t1", list(range(13, 23)), [900] * 10, [600] * 10, fps)
    out = stitch_tracks([a, b], fps)
    assert len(out) == 2


def test_segment_video_with_fake_client(tmp_path):
    """Regression: segment_video must not depend on a type-checking-only import."""
    from types import SimpleNamespace

    from physicam.sam import segment_video

    text = gzip.open(FIXTURE, "rt").read()

    class FakeClient:
        settings = SimpleNamespace(sam="sam-3.1", data_dir=str(tmp_path))

        def upload_file(self, path):
            return "file-fake"

        def responses(self, body):
            return {"output": [{"content": [{"type": "output_text", "text": text}]}]}

        @staticmethod
        def output_text(resp):
            return resp["output"][0]["content"][0]["text"]

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    tracks = segment_video(FakeClient(), video, "red ball", 240.0, cache=False)
    assert len(tracks) >= 1
    assert tracks[0].n > 50


def test_stitch_border_exit_reentry_top():
    """Ball exits the top (y~0) and re-enters nearby: must stitch despite
    the gravity coordinate being unextrapolatable while out of frame."""
    fps = 240.0
    # Rising to the top border, gone 0.2 s, falling back in ~140 px along.
    a = _synth_track("t0", list(range(0, 60)), [300 + 3 * i for i in range(60)],
                     [180 - 3 * i for i in range(60)], fps, diam=44.0)
    b = _synth_track("t1", list(range(108, 168)), [620 + 3 * i for i in range(60)],
                     [10 + 2 * i for i in range(60)], fps, diam=44.0)
    out = stitch_tracks([a, b], fps)
    assert len(out) == 1, [t.summary() for t in out]


def _labelled(track_id, label, frames, xs, ys, conf=0.9, fps=30.0, diam=40.0):
    tr = _synth_track(track_id, frames, xs, ys, fps, diam=diam)
    tr.label = label
    tr.conf[:] = conf
    return tr


def test_cross_prompt_duplicates_merged_with_aliases_and_gap_fill():
    from physicam.sam import deduplicate_tracks

    fps = 30.0
    frames = list(range(20))
    xs = [100 + 2 * i for i in frames]
    kept = _labelled("t0", "red ball", [f for f in frames if f != 10], [x for f, x in zip(frames, xs) if f != 10], [200] * 19, conf=0.9, fps=fps)
    orange = _labelled("t1", "orange ball", [f for f in frames if f != 5], [x + 1 for f, x in zip(frames, xs) if f != 5], [201] * 19, conf=0.8, fps=fps)
    small = _labelled("t2", "small ball", frames[:15], xs[:15], [200] * 15, conf=0.95, fps=fps)
    out = deduplicate_tracks([kept, orange, small])
    assert len(out) == 1
    # tie on frames (19): higher median confidence wins (red, 0.9 > 0.8)
    assert out[0].label == "red ball"
    assert set(out[0].aliases) == {"orange ball", "small ball"}
    assert out[0].n == 20  # frame 10 filled from the dropped orange track
    assert int(out[0].frames[10]) == 10


def test_cross_prompt_gap_fill_rejects_inconsistent_frame():
    from physicam.sam import deduplicate_tracks

    fps = 30.0
    a = _labelled("t0", "red ball", list(range(20)), [100 + i for i in range(20)], [200] * 20, conf=0.9, fps=fps)
    b_frames = list(range(19)) + [25]
    b = _labelled("t1", "orange ball", b_frames, [100 + i for i in range(19)] + [900], [200] * 19 + [600], conf=0.8, fps=fps)
    out = deduplicate_tracks([a, b])
    assert len(out) == 1
    assert 25 not in [int(f) for f in out[0].frames]


def test_cross_prompt_keeps_genuinely_different_close_objects():
    from physicam.sam import deduplicate_tracks

    fps = 30.0
    a = _labelled("t0", "red ball", list(range(20)), [100] * 20, [200] * 20, fps=fps)
    b = _labelled("t1", "blue ball", list(range(20)), [145] * 20, [200] * 20, fps=fps)
    out = deduplicate_tracks([a, b])
    assert len(out) == 2
    assert all(not t.aliases for t in out)


def test_stitch_drops_duplicates():
    fps = 30.0
    a = _synth_track("t0", list(range(0, 20)), [100 + i for i in range(20)], [200] * 20, fps)
    b = _synth_track("t1", list(range(0, 20)), [100 + i for i in range(20)], [200] * 20, fps)
    b.conf[:] = 0.5
    out = stitch_tracks([a, b], fps)
    assert len(out) == 1
    assert out[0].n == 20
