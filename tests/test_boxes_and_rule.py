"""Geometry and descriptive coverage; classification is tested in test_semantic_pairing."""
import numpy as np
import pytest
from utils.boxes import box_area, clip_xyxy, coverage_ratio, crop_window, intersection_area, xyxy_to_yolo, yolo_to_xyxy

def test_intersection_and_area() -> None:
    assert box_area((0, 0, 4, 5)) == 20
    assert intersection_area((0, 0, 4, 4), (2, 2, 6, 6)) == 4
    assert intersection_area((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_yolo_conversion_roundtrip_and_clip() -> None:
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 1000, 500)
    assert box == pytest.approx((400.0, 150.0, 600.0, 350.0))
    assert xyxy_to_yolo(box, 1000, 500) == pytest.approx((0.5, 0.5, 0.2, 0.4))
    assert clip_xyxy((-5, -5, 1200, 600), 1000, 500) == (0.0, 0.0, 1000.0, 500.0)


def test_crop_window_covers_box_and_stays_inside() -> None:
    l, t, r, b = crop_window((10.2, 20.7, 30.1, 40.9), 100, 100)
    assert (l, t, r, b) == (10, 20, 31, 41)
    assert crop_window((-3, -3, 500, 500), 100, 80) == (0, 0, 100, 80)
    l, t, r, b = crop_window((50.0, 50.0, 50.0, 50.0), 100, 100)
    assert r > l and b > t
    assert crop_window((10, 10, 20, 20), 100, 100, margin_px=5) == (5, 5, 25, 25)


@pytest.mark.parametrize("seed", range(5))
def test_random_coverage_bounds(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        f = (*rng.uniform(0, 50, 2), *rng.uniform(60, 120, 2))
        s = (*rng.uniform(0, 80, 2), *rng.uniform(40, 140, 2))
        assert 0.0 <= coverage_ratio(f, s) <= 1.0
