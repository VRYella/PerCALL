"""
core/plotting.py
────────────────
Interactive Plotly figures for PERCALL.

Functions
─────────
plot_perplexity_profile   – Tab 2: perplexity + baseline + residual + regions
plot_motif_enrichment     – Tab 4: motif bar chart (region rate vs background)
plot_genome_browser       – Tab 5: horizontal track view
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .motifs import MOTIF_COLORS, MOTIF_LABELS

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_REGION_FILL = "rgba(248, 81, 73, 0.18)"
_REGION_LINE = "rgba(248, 81, 73, 0.80)"
_PERP_COLOR = "#58a6ff"
_BASELINE_COLOR = "#8b949e"
_RESIDUAL_COLOR = "#e3b341"
_TROUGH_COLOR = "#f85149"

# ---------------------------------------------------------------------------
# Tab 2 – Perplexity profile
# ---------------------------------------------------------------------------


def plot_perplexity_profile(
    perp: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    regions: List[dict],
    seq_offset: int = 0,
    title: str = "Perplexity Profile",
) -> go.Figure:
    """
    Create a two-panel interactive figure:
      • Upper panel  – raw perplexity (blue) and rolling baseline (grey dashed).
      • Lower panel  – residual signal (gold), with called regions shaded in red.

    Parameters
    ----------
    perp : np.ndarray   Raw perplexity values.
    baseline : np.ndarray  Rolling baseline values (same length, may have NaN).
    residual : np.ndarray  perp - baseline.
    regions : list of dict  Each dict has keys: start, end, trough, rank.
    seq_offset : int  Positional offset added to x-coordinates (for display).
    title : str  Figure title.
    """
    positions = np.arange(seq_offset, seq_offset + len(perp))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Perplexity & Baseline", "Residual Signal"),
        row_heights=[0.55, 0.45],
    )

    # --- upper panel: raw perplexity -----------------------------------------
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=perp,
            mode="lines",
            name="Perplexity",
            line=dict(color=_PERP_COLOR, width=1.0),
            hovertemplate="Position: %{x}<br>Perplexity: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=baseline,
            mode="lines",
            name="Baseline",
            line=dict(color=_BASELINE_COLOR, width=1.2, dash="dot"),
            hovertemplate="Position: %{x}<br>Baseline: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- lower panel: residual -----------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=residual,
            mode="lines",
            name="Residual",
            line=dict(color=_RESIDUAL_COLOR, width=1.0),
            hovertemplate="Position: %{x}<br>Residual: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Zero reference line
    fig.add_hline(y=0, line_color=_BASELINE_COLOR, line_dash="dash", line_width=0.8, row=2, col=1)

    # --- region shading -------------------------------------------------------
    for r in regions:
        x0, x1 = r["start"] + seq_offset, r["end"] + seq_offset
        label = f"Region {r['rank']}"
        for row in (1, 2):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=_REGION_FILL,
                line_color=_REGION_LINE,
                line_width=1.0,
                annotation_text=label if row == 1 else "",
                annotation_position="top left",
                annotation_font_size=9,
                row=row,
                col=1,
            )
        # Trough marker on upper panel
        fig.add_vline(
            x=r["trough"] + seq_offset,
            line_color=_TROUGH_COLOR,
            line_dash="dot",
            line_width=0.9,
            row=1,
            col=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_xaxes(title_text="Position (bp)", row=2, col=1, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Perplexity", row=1, col=1, gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Residual", row=2, col=1, gridcolor="#e0e0e0")

    return fig


# ---------------------------------------------------------------------------
# Tab 4 – Motif enrichment bar chart
# ---------------------------------------------------------------------------


def plot_motif_enrichment(
    region_counts: Dict[str, int],
    background_counts: Dict[str, int],
    total_region_bp: int,
    total_seq_bp: int,
    active_motifs: List[str],
) -> go.Figure:
    """
    Grouped bar chart: motif hits per kb inside called regions vs whole sequence.

    Parameters
    ----------
    region_counts : dict  Motif hit counts within called regions.
    background_counts : dict  Motif hit counts across the full sequence.
    total_region_bp : int  Total bp spanned by called regions.
    total_seq_bp : int  Total sequence length.
    active_motifs : list[str]  Ordered list of motif keys to display.
    """
    region_kb = max(total_region_bp / 1000.0, 1e-9)
    seq_kb = max(total_seq_bp / 1000.0, 1e-9)

    labels = [MOTIF_LABELS.get(m, m) for m in active_motifs]
    bg_rates = [background_counts.get(m, 0) / seq_kb for m in active_motifs]
    reg_rates = [region_counts.get(m, 0) / region_kb for m in active_motifs]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Whole-sequence rate",
            x=labels,
            y=bg_rates,
            marker_color="#4a9eff",
            hovertemplate="%{x}<br>Background: %{y:.2f} hits/kb<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Inside low-perplexity regions",
            x=labels,
            y=reg_rates,
            marker_color="#f85149",
            hovertemplate="%{x}<br>In regions: %{y:.2f} hits/kb<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Non-B DNA Motif Enrichment: Regions vs Background",
        xaxis_title="Motif",
        yaxis_title="Hits per kb",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        margin=dict(l=60, r=30, t=80, b=100),
        xaxis=dict(tickangle=-30),
    )
    return fig


def plot_motif_distribution(
    motif_positions: Dict[str, List[Tuple[int, int]]],
    seq_len: int,
    active_motifs: List[str],
) -> go.Figure:
    """
    Horizontal bar chart showing relative positional density of each motif.
    Each motif gets a row; hits are plotted as colored ticks.
    """
    fig = go.Figure()
    y_labels = [MOTIF_LABELS.get(m, m) for m in active_motifs]

    for i, motif in enumerate(active_motifs):
        positions = motif_positions.get(motif, [])
        if not positions:
            continue
        xs = [(s + e) / 2 for s, e in positions]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[y_labels[i]] * len(xs),
                mode="markers",
                marker=dict(
                    color=MOTIF_COLORS.get(motif, "#888888"),
                    symbol="line-ns",
                    size=12,
                    line=dict(width=2, color=MOTIF_COLORS.get(motif, "#888888")),
                ),
                name=y_labels[i],
                hovertemplate="Position: %{x}<extra>" + y_labels[i] + "</extra>",
            )
        )

    fig.update_layout(
        title="Motif Distribution along Sequence",
        xaxis_title="Position (bp)",
        xaxis=dict(range=[0, seq_len]),
        height=max(200, 50 + 40 * len(active_motifs)),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        margin=dict(l=130, r=30, t=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 5 – Genome browser track view
# ---------------------------------------------------------------------------


def plot_genome_browser(
    seq_len: int,
    regions: List[dict],
    motif_positions: Dict[str, List[Tuple[int, int]]],
    active_motifs: List[str],
    seq_label: str = "Sequence",
) -> go.Figure:
    """
    Horizontal multi-track view:
      Track 0 – full sequence bar
      Track 1 – called regulatory regions
      Tracks 2+ – one per motif type

    Parameters
    ----------
    seq_len : int  Total sequence length.
    regions : list of dict  Region dicts (start, end, rank).
    motif_positions : dict  Motif name → list of (start, end) tuples.
    active_motifs : list[str]  Motifs to show as tracks.
    seq_label : str  Label for the sequence track.
    """
    track_labels = [seq_label, "Regions"] + [
        MOTIF_LABELS.get(m, m) for m in active_motifs
    ]
    n_tracks = len(track_labels)
    track_height = 0.7  # fraction of each track row used by the bar

    fig = go.Figure()

    # Y positions: highest track at top
    def _y(track_idx: int) -> float:
        return float(n_tracks - track_idx)

    # --- Track 0: full sequence bar -----------------------------------------
    fig.add_shape(
        type="rect",
        x0=0, x1=seq_len,
        y0=_y(0) - track_height / 2,
        y1=_y(0) + track_height / 2,
        fillcolor="#d0d7de",
        line=dict(width=0),
    )

    # --- Track 1: called regions --------------------------------------------
    for r in regions:
        fig.add_shape(
            type="rect",
            x0=r["start"], x1=r["end"],
            y0=_y(1) - track_height / 2,
            y1=_y(1) + track_height / 2,
            fillcolor=_REGION_FILL,
            line=dict(color=_REGION_LINE, width=1),
        )
        fig.add_annotation(
            x=(r["start"] + r["end"]) / 2,
            y=_y(1),
            text=f"R{r['rank']}",
            showarrow=False,
            font=dict(size=9, color="#f85149"),
            yanchor="middle",
        )

    # --- Motif tracks --------------------------------------------------------
    for ti, motif in enumerate(active_motifs, start=2):
        color = MOTIF_COLORS.get(motif, "#888888")
        hits = motif_positions.get(motif, [])
        for start, end in hits:
            width = max(end - start, 3)
            fig.add_shape(
                type="rect",
                x0=start, x1=start + width,
                y0=_y(ti) - track_height / 3,
                y1=_y(ti) + track_height / 3,
                fillcolor=color,
                line=dict(width=0),
                opacity=0.75,
            )

    # Invisible scatter for y-axis labels
    fig.add_trace(
        go.Scatter(
            x=[seq_len / 2] * n_tracks,
            y=[_y(i) for i in range(n_tracks)],
            mode="text",
            text=track_labels,
            textposition="middle left",
            textfont=dict(size=10),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Genome Browser View",
        xaxis=dict(title="Position (bp)", range=[0, seq_len], gridcolor="#e0e0e0"),
        yaxis=dict(
            tickvals=[_y(i) for i in range(n_tracks)],
            ticktext=track_labels,
            range=[0.3, n_tracks + 0.7],
            showgrid=False,
        ),
        height=max(250, 80 + 45 * n_tracks),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        margin=dict(l=130, r=30, t=60, b=50),
        showlegend=False,
    )
    return fig
