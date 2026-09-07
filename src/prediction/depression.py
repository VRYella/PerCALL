from __future__ import annotations

import numpy as np

from src.utils.constants import EPSILON


def calculate_perplexity_depression(perplexity: np.ndarray, background: np.ndarray, normalize: bool = False) -> np.ndarray:
    if normalize:
        return ((background - perplexity) / (background + EPSILON)).astype(np.float32)
    return (background - perplexity).astype(np.float32)
