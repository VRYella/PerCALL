from src.models.dataclasses import PerplexityConfig
from src.prediction.regulatory_predictor import predict_regulatory_regions


def test_synthetic_low_perplexity_segment_detected():
    background = "ACGT" * 125
    low_complexity = "A" * 200
    sequence = background + low_complexity + background
    cfg = PerplexityConfig(
        perplexity_window=17,
        min_region_length=80,
        max_region_length=400,
        min_perplexity_depression=0.2,
        min_persistence_bp=50,
    )
    result = predict_regulatory_regions("synthetic", sequence, cfg)
    assert len(result.candidate_regions) >= 1
    assert any(region.start < 600 and region.end > 500 for region in result.candidate_regions)


def test_long_repetitive_sequence_not_entirely_called():
    sequence = "A" * 2000
    cfg = PerplexityConfig(min_region_length=100, max_region_length=500, min_perplexity_depression=0.0, min_persistence_bp=100)
    result = predict_regulatory_regions("repeat", sequence, cfg)
    assert len(result.candidate_regions) >= 1
    assert all(region.length <= cfg.max_region_length for region in result.candidate_regions)


def test_oversized_positive_interval_is_bounded_to_max_length():
    sequence = "A" * 2500
    cfg = PerplexityConfig(
        perplexity_window=17,
        min_region_length=100,
        max_region_length=300,
        min_perplexity_depression=0.0,
        min_persistence_bp=100,
    )
    result = predict_regulatory_regions("oversized", sequence, cfg)
    assert len(result.candidate_regions) >= 1
    assert all(region.length <= cfg.max_region_length for region in result.candidate_regions)
