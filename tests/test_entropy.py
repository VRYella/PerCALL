import numpy as np

from src.perplexity.entropy import perplexity_from_entropy, shannon_entropy


def test_uniform_entropy_and_perplexity():
    probs = np.full((1, 16), 1 / 16, dtype=float)
    entropy = shannon_entropy(probs)
    assert np.isclose(entropy[0], 4.0)
    ppl = perplexity_from_entropy(entropy)
    assert np.isclose(ppl[0], 16.0)


def test_single_state_entropy():
    probs = np.zeros((1, 16), dtype=float)
    probs[0, 0] = 1.0
    entropy = shannon_entropy(probs)
    ppl = perplexity_from_entropy(entropy)
    assert np.isclose(entropy[0], 0.0)
    assert np.isclose(ppl[0], 1.0)
