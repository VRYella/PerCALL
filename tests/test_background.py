import numpy as np

from src.prediction.background import estimate_local_background


def test_background_uses_flanks_and_handles_boundaries():
    profile = np.array([10.0, 9.0, 5.0, 9.0, 10.0], dtype=np.float32)
    background = estimate_local_background(profile, flank_size=2)
    assert np.isclose(background[2], 9.5)
    assert np.isfinite(background[0])
    assert np.isfinite(background[-1])
