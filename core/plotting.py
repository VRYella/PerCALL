"""
core/plotting.py
────────────────
Interactive Plotly figures for PERCALL.

Functions
─────────
plot_perplexity_profile   – Tab 3: perplexity + baseline + residual + regions
plot_gc_profile           – Tab 2: sliding-window GC content along sequence
plot_region_distribution  – Tab 6: region width & score distribution histograms
plot_motif_enrichment     – Tab 4: motif bar chart (region rate vs background)
plot_motif_distribution   – Tab 4: motif positional density ticks
plot_genome_browser       – Tab 5: horizontal multi-track view
plot_sequence_domain_map  – Tab 3: compact region/motif architecture map
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .motifs import MOTIF_COLORS, MOTIF_LABELS

# ---------------------------------------------------------------------------
# Dark-theme colour palette  (electric-blue / emerald / amber accent)
# ---------------------------------------------------------------------------
_PERP_COLOR    = "#00d4ff"   # electric blue
_BASELINE_COLOR = "#8b949e"  # muted grey
_RESIDUAL_COLOR = "#e3b341"  # amber / gold
_TROUGH_COLOR  = "#ff4d6d"   # rose-red
_GC_COLOR      = "#00ff9d"   # emerald
_REGION_FILL   = "rgba(0, 212, 255, 0.10)"
_REGION_LINE   = "rgba(0, 212, 255, 0.70)"

_DARK_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="#c9d1d9", family="Inter, system-ui, sans-serif", size=11),
    hoverlabel=dict(bgcolor="#0d1b2a", font_color="#e6edf3", bordercolor="#00d4ff"),
)

_GRID_X = dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.15)")
_GRID_Y = dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.15)")


# ---------------------------------------------------------------------------
# Tab 3 – Perplexity profile
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
      • Lower panel  – residual signal (amber), with called regions shaded in blue.

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
            line=dict(color=_PERP_COLOR, width=1.2),
            hovertemplate="Position: %{x}<br>Perplexity: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=baseline,
            mode="lines",
            name="Baseline",
            line=dict(color=_BASELINE_COLOR, width=1.3, dash="dot"),
            hovertemplate="Position: %{x}<br>Baseline: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- lower panel: residual -----------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=residual,
            mode="lines",
            name="Residual",
            line=dict(color=_RESIDUAL_COLOR, width=1.1),
            hovertemplate="Position: %{x}<br>Residual: %{y:.4f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color=_BASELINE_COLOR, line_dash="dash",
                  line_width=0.8, row=2, col=1)

    # --- region shading -------------------------------------------------------
    for r in regions:
        x0 = r["start"] + seq_offset
        x1 = r["end"] + seq_offset
        label = f"R{r['rank']}"
        for row in (1, 2):
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=_REGION_FILL,
                line_color=_REGION_LINE,
                line_width=1.2,
                annotation_text=label if row == 1 else "",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=_PERP_COLOR,
                row=row, col=1,
            )
        fig.add_vline(
            x=r["trough"] + seq_offset,
            line_color=_TROUGH_COLOR, line_dash="dot", line_width=1.0,
            row=1, col=1,
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#00d4ff")),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_xaxes(title_text="Position (bp)", row=2, col=1, **_GRID_X)
    fig.update_yaxes(title_text="Perplexity", row=1, col=1, **_GRID_Y)
    fig.update_yaxes(title_text="Residual", row=2, col=1, **_GRID_Y)
    fig.update_annotations(font_color="#8b9bb4")

    return fig


# ---------------------------------------------------------------------------
# Tab 2 – GC content profile
# ---------------------------------------------------------------------------


def plot_gc_profile(
    seq: str,
    regions: Optional[List[dict]] = None,
    window: int = 50,
    title: str = "GC Content Profile",
) -> go.Figure:
    """
    Sliding-window GC% plotted along the sequence.

    Parameters
    ----------
    seq : str   DNA sequence (uppercase).
    regions : list of dict | None   Regions to shade (start, end, rank).
    window : int  Window size for GC calculation (default 50).
    title : str   Figure title.
    """
    n = len(seq)
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    is_gc = ((arr == ord('G')) | (arr == ord('C'))).astype(np.float32)

    if n < window:
        gc = np.array([is_gc.mean() * 100])
        positions = np.array([n // 2])
    else:
        # Efficient sliding sum
        cumsum = np.concatenate([[0.0], np.cumsum(is_gc)])
        gc = (cumsum[window:] - cumsum[:-window]) / window * 100.0
        positions = np.arange(window // 2, window // 2 + len(gc))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=gc,
            mode="lines",
            name="GC%",
            line=dict(color=_GC_COLOR, width=1.4),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 157, 0.06)",
            hovertemplate="Position: %{x}<br>GC: %{y:.1f}%<extra></extra>",
        )
    )
    # 50% reference line
    fig.add_hline(y=50, line_color="#8b949e", line_dash="dash", line_width=0.8)

    if regions:
        for r in regions:
            fig.add_vrect(
                x0=r["start"], x1=r["end"],
                fillcolor=_REGION_FILL, line_color=_REGION_LINE, line_width=1.0,
                annotation_text=f"R{r['rank']}", annotation_position="top left",
                annotation_font_size=9, annotation_font_color=_PERP_COLOR,
            )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=13, color=_GC_COLOR)),
        height=280,
        xaxis=dict(title="Position (bp)", **_GRID_X),
        yaxis=dict(title="GC Content (%)", range=[0, 100], **_GRID_Y),
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=False,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 6 – Region distribution histograms
# ---------------------------------------------------------------------------


def plot_region_distribution(
    regions_df,
    title: str = "Region Distribution",
) -> go.Figure:
    """
    Two-panel histogram: region widths (left) and mean residual scores (right).

    Parameters
    ----------
    regions_df : pd.DataFrame   Must have columns 'Width' and 'Mean Residual'.
    title : str   Figure title.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Region Width Distribution", "Mean Residual Distribution"),
    )

    widths = regions_df["Width"].dropna() if "Width" in regions_df.columns else []
    scores = regions_df["Mean Residual"].dropna() if "Mean Residual" in regions_df.columns else []

    if len(widths):
        fig.add_trace(
            go.Histogram(
                x=widths, name="Width (bp)",
                marker_color=_PERP_COLOR, opacity=0.75,
                hovertemplate="Width: %{x} bp<br>Count: %{y}<extra></extra>",
            ),
            row=1, col=1,
        )

    if len(scores):
        fig.add_trace(
            go.Histogram(
                x=scores, name="Mean Residual",
                marker_color=_RESIDUAL_COLOR, opacity=0.75,
                hovertemplate="Residual: %{x:.4f}<br>Count: %{y}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#c9d1d9")),
        height=300,
        showlegend=False,
        margin=dict(l=60, r=30, t=80, b=50),
        bargap=0.05,
    )
    fig.update_xaxes(row=1, col=1, title_text="Width (bp)", **_GRID_X)
    fig.update_xaxes(row=1, col=2, title_text="Mean Residual", **_GRID_X)
    fig.update_yaxes(title_text="Count", **_GRID_Y)
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
            x=labels, y=bg_rates,
            marker_color=_BASELINE_COLOR,
            opacity=0.8,
            hovertemplate="%{x}<br>Background: %{y:.2f} hits/kb<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Inside low-perplexity regions",
            x=labels, y=reg_rates,
            marker_color=_PERP_COLOR,
            opacity=0.9,
            hovertemplate="%{x}<br>In regions: %{y:.2f} hits/kb<extra></extra>",
        )
    )

    fig.update_layout(
        **_DARK_LAYOUT,
        barmode="group",
        title=dict(text="Non-B DNA Motif Enrichment: Regions vs Background",
                   font=dict(size=13, color="#c9d1d9")),
        xaxis=dict(title="Motif", tickangle=-30, **_GRID_X),
        yaxis=dict(title="Hits per kb", **_GRID_Y),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=30, t=80, b=100),
    )
    return fig


def plot_motif_distribution(
    motif_positions: Dict[str, List[Tuple[int, int]]],
    seq_len: int,
    active_motifs: List[str],
) -> go.Figure:
    """
    Horizontal tick chart showing relative positional density of each motif.
    """
    fig = go.Figure()
    y_labels = [MOTIF_LABELS.get(m, m) for m in active_motifs]

    for i, motif in enumerate(active_motifs):
        positions = motif_positions.get(motif, [])
        if not positions:
            continue
        xs = [(s + e) / 2 for s, e in positions]
        color = MOTIF_COLORS.get(motif, "#888888")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[y_labels[i]] * len(xs),
                mode="markers",
                marker=dict(
                    color=color, symbol="line-ns", size=13,
                    line=dict(width=2, color=color),
                ),
                name=y_labels[i],
                hovertemplate="Position: %{x}<extra>" + y_labels[i] + "</extra>",
            )
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Motif Distribution along Sequence",
                   font=dict(size=13, color="#c9d1d9")),
        xaxis=dict(title="Position (bp)", range=[0, seq_len], **_GRID_X),
        yaxis=dict(**_GRID_Y),
        height=max(200, 50 + 40 * len(active_motifs)),
        showlegend=False,
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
    """
    track_labels = [seq_label, "Regions"] + [
        MOTIF_LABELS.get(m, m) for m in active_motifs
    ]
    n_tracks = len(track_labels)
    track_height = 0.65

    fig = go.Figure()

    def _y(track_idx: int) -> float:
        return float(n_tracks - track_idx)

    # Track 0: full sequence bar
    fig.add_shape(
        type="rect",
        x0=0, x1=seq_len,
        y0=_y(0) - track_height / 2,
        y1=_y(0) + track_height / 2,
        fillcolor="rgba(139,155,180,0.25)",
        line=dict(width=0),
    )

    # Track 1: called regions
    for r in regions:
        fig.add_shape(
            type="rect",
            x0=r["start"], x1=r["end"],
            y0=_y(1) - track_height / 2,
            y1=_y(1) + track_height / 2,
            fillcolor=_REGION_FILL,
            line=dict(color=_REGION_LINE, width=1.2),
        )
        fig.add_annotation(
            x=(r["start"] + r["end"]) / 2, y=_y(1),
            text=f"R{r['rank']}", showarrow=False,
            font=dict(size=9, color=_PERP_COLOR), yanchor="middle",
        )

    # Motif tracks
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
                opacity=0.80,
            )

    # Invisible scatter for y-axis labels
    fig.add_trace(
        go.Scatter(
            x=[seq_len / 2] * n_tracks,
            y=[_y(i) for i in range(n_tracks)],
            mode="text",
            text=track_labels,
            textposition="middle left",
            textfont=dict(size=10, color="#c9d1d9"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Genome Browser View", font=dict(size=13, color="#c9d1d9")),
        xaxis=dict(title="Position (bp)", range=[0, seq_len], **_GRID_X),
        yaxis=dict(
            tickvals=[_y(i) for i in range(n_tracks)],
            ticktext=track_labels,
            range=[0.3, n_tracks + 0.7],
            showgrid=False,
        ),
        height=max(260, 85 + 48 * n_tracks),
        margin=dict(l=140, r=30, t=60, b=50),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 3 – Sequence domain / architecture map
# ---------------------------------------------------------------------------


def plot_sequence_domain_map(
    seq_len: int,
    regions: List[dict],
    title: str = "Region Architecture Map",
) -> go.Figure:
    """
    Compact horizontal bar showing the sequence with called regions overlaid.

    Parameters
    ----------
    seq_len : int  Total sequence length.
    regions : list of dict  Region dicts (start, end, rank, mean_residual).
    title : str   Figure title.
    """
    fig = go.Figure()

    # Full sequence bar
    fig.add_shape(
        type="rect", x0=0, x1=seq_len, y0=0.25, y1=0.75,
        fillcolor="rgba(139,155,180,0.2)", line=dict(color="rgba(139,155,180,0.4)", width=1),
    )

    # Region rectangles with colour intensity scaled by abs(mean_residual)
    scores = [abs(r["mean_residual"]) for r in regions] if regions else [1]
    max_score = max(scores) if scores else 1.0

    for r in regions:
        intensity = abs(r["mean_residual"]) / max(max_score, 1e-9)
        alpha = 0.3 + 0.6 * intensity
        fig.add_shape(
            type="rect",
            x0=r["start"], x1=r["end"],
            y0=0.1, y1=0.9,
            fillcolor=f"rgba(0, 212, 255, {alpha:.2f})",
            line=dict(color=_REGION_LINE, width=1.0),
        )
        fig.add_annotation(
            x=(r["start"] + r["end"]) / 2, y=0.5,
            text=f"R{r['rank']}", showarrow=False,
            font=dict(size=9, color="#060b18"), yanchor="middle",
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=12, color="#c9d1d9")),
        height=120,
        xaxis=dict(title="Position (bp)", range=[0, seq_len], **_GRID_X),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig
