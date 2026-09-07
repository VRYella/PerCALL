from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_perplexity_background_pds(positions: np.ndarray, perplexity: np.ndarray, background: np.ndarray, pds: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=positions, y=perplexity, mode="lines", name="Perplexity"))
    fig.add_trace(go.Scatter(x=positions, y=background, mode="lines", name="Local background"))
    fig.add_trace(go.Scatter(x=positions, y=pds, mode="lines", name="PDS", yaxis="y2"))
    fig.update_layout(
        xaxis_title="Position",
        yaxis_title="Perplexity",
        yaxis2=dict(title="PDS", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig
