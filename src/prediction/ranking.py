from __future__ import annotations

from src.models.dataclasses import CandidateRegion


def rank_regions(regions: list[CandidateRegion]) -> list[CandidateRegion]:
    scored = sorted(
        regions,
        key=lambda r: (-r.mean_pds, -r.max_pds, -r.persistence, r.mean_perplexity),
    )
    for rank, region in enumerate(scored, start=1):
        region.rank = rank
    return scored
