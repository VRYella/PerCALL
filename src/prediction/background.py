from __future__ import annotations

import numpy as np


def estimate_local_background(profile: np.ndarray, flank_size: int) -> np.ndarray:
    n = len(profile)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out

    for i in range(n):
        left_start = max(0, i - flank_size)
        left = profile[left_start:i]
        right_end = min(n, i + flank_size + 1)
        right = profile[i + 1:right_end]
        flank = np.concatenate([left, right]) if left.size or right.size else np.array([], dtype=np.float32)
        finite = flank[np.isfinite(flank)]
        if finite.size:
            out[i] = np.float32(np.median(finite))
    return out
