from __future__ import annotations

import numpy as np

_BASE_MAP = np.full(256, 4, dtype=np.uint8)
for idx, base in enumerate("ACGT"):
    _BASE_MAP[ord(base)] = idx


def encode_dinucleotides(sequence: str, window_size: int, step_size: int = 1) -> np.ndarray:
    arr = _BASE_MAP[np.frombuffer(sequence.encode(), dtype=np.uint8)]
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_size)[::step_size]
    left = windows[:, :-1]
    right = windows[:, 1:]
    return (left * 4 + right).astype(np.int16)


def has_ambiguous_base(sequence: str, window_size: int, step_size: int = 1) -> np.ndarray:
    arr = _BASE_MAP[np.frombuffer(sequence.encode(), dtype=np.uint8)] == 4
    return np.lib.stride_tricks.sliding_window_view(arr, window_size)[::step_size].any(axis=1)
