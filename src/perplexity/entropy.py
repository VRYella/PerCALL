from __future__ import annotations

import numpy as np


def shannon_entropy(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(probabilities > 0, probabilities, 1.0)
        return -np.sum(probabilities * np.log2(p), axis=1)


def perplexity_from_entropy(entropy: np.ndarray) -> np.ndarray:
    return (2.0 ** entropy).astype(np.float32)
