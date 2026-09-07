from __future__ import annotations

import numpy as np


def _windows_to_bp(num_windows: int, step_size: int, window_size: int) -> int:
    if num_windows <= 0:
        return 0
    return int(window_size + (num_windows - 1) * step_size)


def detect_candidate_intervals(
    pds: np.ndarray,
    min_pds: float,
    min_region_length: int,
    min_persistence_bp: int,
    step_size: int,
    window_size: int = 1,
) -> list[tuple[int, int]]:
    positive = np.isfinite(pds) & (pds >= min_pds)
    padded = np.concatenate([[False], positive, [False]])
    delta = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(delta > 0)
    ends = np.flatnonzero(delta < 0) - 1

    regions: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        length_windows = end - start + 1
        length_bp = _windows_to_bp(length_windows, step_size, window_size)
        if length_bp < min_region_length or length_bp < min_persistence_bp:
            continue
        regions.append((int(start), int(end)))
    return regions


def merge_intervals(intervals: list[tuple[int, int]], merge_distance: int, step_size: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    max_gap_windows = merge_distance // max(step_size, 1)
    for start, end in sorted_intervals[1:]:
        ps, pe = merged[-1]
        if start <= pe + max_gap_windows + 1:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged
