import numpy as np

from src.prediction.depression import calculate_perplexity_depression


def test_pds_math_known_values():
    perplexity = np.array([8.0, 7.0], dtype=np.float32)
    background = np.array([10.0, 10.0], dtype=np.float32)
    pds = calculate_perplexity_depression(perplexity, background)
    assert np.allclose(pds, np.array([2.0, 3.0], dtype=np.float32))
