import numpy as np

from src.prediction.regions import detect_candidate_intervals, merge_intervals


def test_no_valley_returns_none():
    pds = np.array([0.0, 0.1, 0.2, 0.0], dtype=np.float32)
    regions = detect_candidate_intervals(pds, min_pds=0.5, min_region_length=2, min_persistence_bp=2, step_size=1)
    assert regions == []


def test_single_persistent_valley_detected():
    pds = np.array([0.0, 0.6, 0.7, 0.65, 0.0], dtype=np.float32)
    regions = detect_candidate_intervals(pds, min_pds=0.5, min_region_length=3, min_persistence_bp=3, step_size=1)
    assert regions == [(1, 3)]


def test_adjacent_valleys_merge():
    intervals = [(1, 3), (5, 8)]
    merged = merge_intervals(intervals, merge_distance=2, step_size=1)
    assert merged == [(1, 8)]
