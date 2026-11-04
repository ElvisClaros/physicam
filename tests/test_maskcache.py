"""Mask cache: packbits roundtrip vs direct decode, parallel == serial."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from physicam.maskcache import box_records, decoded_masks, mask_cache_path

FIXTURE = Path(__file__).parent / "fixtures" / "bounce240_red_ball_f0-199.sam.txt.gz"


def _text() -> str:
    return gzip.open(FIXTURE, "rt").read()


def _direct_rasters(text: str) -> dict:
    """Reference rasters via the official parser + codec, decoded directly."""
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, video_segmentation_format

    parser = video_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(CompletedOutcome()).result
    out = {}
    for rec in result.records:
        if rec.kind != "mask":
            continue
        m = rec.mask
        out[(str(rec.object_id), int(rec.frame.frame_index))] = np.frombuffer(
            decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
    return out


def test_packbits_roundtrip_equals_direct_decode(tmp_path):
    text = _text()
    direct = _direct_rasters(text)
    assert direct
    ms = decoded_masks(text, "fixture_roundtrip", data_dir=str(tmp_path), max_workers=2)
    # The npz was written; a second call loads it and must agree exactly.
    assert mask_cache_path(str(tmp_path), "fixture_roundtrip").exists()
    warm = decoded_masks(text, "fixture_roundtrip", data_dir=str(tmp_path), max_workers=2)
    assert len(warm.records) == len(ms.records)
    for rec in ms.records:
        if not rec.has_mask:
            continue
        key = (rec.object_id, int(rec.frame))
        assert np.array_equal(ms.raster(rec.index), direct[key]), key
        assert np.array_equal(warm.raster(rec.index), direct[key]), key


def test_parallel_equals_serial(tmp_path):
    text = _text()
    serial = decoded_masks(text, "fixture_serial", data_dir=str(tmp_path),
                           max_workers=1, use_cache=False)
    parallel = decoded_masks(text, "fixture_parallel", data_dir=str(tmp_path),
                             max_workers=4, use_cache=False)
    assert len(serial.records) == len(parallel.records) > 0
    for a, b in zip(serial.records, parallel.records):
        assert (a.object_id, a.frame) == (b.object_id, b.frame)
        assert a.area == b.area
        assert a.centroid_x == b.centroid_x and a.centroid_y == b.centroid_y
        assert a.eq_diameter == b.eq_diameter
        assert np.array_equal(serial.raster(a.index), parallel.raster(b.index))


def test_contours_only_for_selected_records(tmp_path):
    text = _text()
    ms = decoded_masks(text, "fixture_contours", data_dir=str(tmp_path), max_workers=2)
    picked = [ms.records[0].index, ms.records[5].index]
    contours = ms.ensure_contours(picked, max_workers=2)
    assert set(contours) == set(picked)
    assert all(isinstance(v, str) for v in contours.values())
    # Cached: a fresh MaskSet over the same npz serves them without the text.
    warm = decoded_masks(text, "fixture_contours", data_dir=str(tmp_path), max_workers=2)
    again = warm.ensure_contours(picked, max_workers=2)
    assert again == contours


def test_box_records_match_mask_keys(tmp_path):
    text = _text()
    boxes = box_records(text)
    ms = decoded_masks(text, "fixture_boxes", data_dir=str(tmp_path),
                       max_workers=1, use_cache=False)
    mask_keys = {(r.object_id, int(r.frame)) for r in ms.records if r.has_mask}
    assert mask_keys <= set(boxes)
    assert len(boxes) >= len(mask_keys) > 0
