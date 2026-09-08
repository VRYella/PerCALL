from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.models.dataclasses import CandidateRegion


def add_region_highlights(fig: go.Figure, regions: list[CandidateRegion]) -> go.Figure:
    for region in regions:
        fig.add_vrect(x0=region.start, x1=region.end, fillcolor="rgba(16,185,129,0.2)", line_width=0)
    return fig


def region_summary_text(region: CandidateRegion) -> str:
    summary = (
        f"Mean DNA perplexity: {region.mean_perplexity:.2f}\n"
        f"Local background PPL: {region.background_perplexity:.2f}\n"
        f"Mean PDS: {region.mean_pds:.2f}\n"
        f"Maximum PDS: {region.max_pds:.2f}\n"
        f"Persistence: {region.persistence} bp\n"
        f"Prediction rank: {region.rank}"
    )
    if region.motif_count:
        summary += f"\nMotif hits: {region.motif_count}\nMotif summary: {region.motifs}"
    return summary
