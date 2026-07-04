from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Scientific colour palette (white theme) ────────────────────────────────
_BG      = "#ffffff"
_GRID    = "rgba(148,163,184,0.22)"
_TEXT    = "#1f2937"
_MUTED   = "#475569"
_BLUE    = "#1E3A8A"
_TEAL    = "#0F766E"
_GREEN   = "#10B981"
_AMBER   = "#F59E0B"
_CRIMSON = "#DC2626"
_INDIGO  = "#4F46E5"

_BASE_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, Segoe UI, sans-serif", size=16),
    title_font=dict(size=24, color=_BLUE, family="Space Grotesk, Inter, sans-serif"),
    margin=dict(l=72, r=36, t=72, b=60),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=16),
    ),
    hoverlabel=dict(font=dict(size=15)),
)


def _apply_base(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, title=f"<b>{title}</b>", height=height)
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor="rgba(203,213,225,0.6)", linewidth=1,
        tickfont=dict(color=_MUTED, size=16),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(226,232,240,0.7)", zeroline=False,
        linecolor="rgba(203,213,225,0.6)", linewidth=1,
        tickfont=dict(color=_MUTED, size=16),
        title_font=dict(size=18),
    )
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data available", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False,
                       font=dict(color=_MUTED, size=14))
    return _apply_base(fig, title)


def _valley_bounds(domain: dict) -> tuple[int, int]:
    return int(domain.get("Signal_Start", domain.get("Start", 0))), \
           int(domain.get("Signal_End",   domain.get("End",   0)))


def _add_valley_shading(fig: go.Figure, domains: list[dict]) -> None:
    for i, d in enumerate(domains):
        s0, s1 = _valley_bounds(d)
        valley_id = d.get("ID", f"PV_{i+1:06d}")
        score     = d.get("ValleyScore", 0.0)
        fig.add_vrect(
            x0=s0, x1=s1,
            fillcolor="rgba(30,58,138,0.12)",
            line_color=_BLUE, line_width=0.8, opacity=0.8,
            annotation_text=valley_id,
            annotation_position="top left",
            annotation_font=dict(size=10, color=_BLUE),
        )


# ─── Plot 1: Raw Perplexity Landscape ────────────────────────────────────────

def plot_perplexity_landscape(di: np.ndarray, domains: list[dict]) -> go.Figure:
    """Raw dinucleotide perplexity profile with detected valleys shaded."""
    if len(di) == 0:
        return _empty_figure("Raw Dinucleotide Perplexity Landscape")
    x = np.arange(len(di))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=di, mode="lines",
        name="Di Perplexity (raw)",
        line=dict(color=_TEAL, width=1.4),
        opacity=0.85,
    ))
    _add_valley_shading(fig, domains)
    fig.update_layout(xaxis_title="Signal position (nt)", yaxis_title="Perplexity")
    return _apply_base(fig, "Raw Dinucleotide Perplexity Landscape", height=380)


# ─── Plot 2: Smoothed Perplexity Landscape ───────────────────────────────────

def plot_smoothed_perplexity(
    di: np.ndarray,
    smoothed_di: np.ndarray,
    domains: list[dict],
) -> go.Figure:
    """Raw and Savitzky-Golay smoothed perplexity with detected valleys."""
    if len(di) == 0 and len(smoothed_di) == 0:
        return _empty_figure("Smoothed Perplexity Landscape")
    fig = go.Figure()
    if len(di) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(di)), y=di, mode="lines",
            name="Raw perplexity",
            line=dict(color=_TEAL, width=1.0, dash="dot"),
            opacity=0.4,
        ))
    if len(smoothed_di) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(smoothed_di)), y=smoothed_di, mode="lines",
            name="SG-smoothed perplexity",
            line=dict(color=_BLUE, width=2.2),
            opacity=0.9,
        ))
    _add_valley_shading(fig, domains)
    fig.update_layout(xaxis_title="Signal position (nt)", yaxis_title="Perplexity")
    return _apply_base(fig, "Savitzky–Golay Smoothed Perplexity", height=380)


# ─── Plot 3: Three-Window Illustration ───────────────────────────────────────

def plot_three_window(
    smoothed_di: np.ndarray,
    pds: np.ndarray,
    domain: dict,
    flank_size: int = 100,
    spacer_size: int = 50,
) -> go.Figure:
    """Visualise the three-window PDS layout for a selected valley.

    Shows the upstream flank, spacer, candidate, spacer, and downstream flank
    with colour-coded background bands on the smoothed-perplexity trace.
    """
    if len(smoothed_di) == 0:
        return _empty_figure("Three-Window Illustration")

    s0, s1 = _valley_bounds(domain)
    n = len(smoothed_di)

    # Window boundaries
    up_s  = max(0, s0 - spacer_size - flank_size)
    up_e  = max(0, s0 - spacer_size)
    dn_s  = min(n - 1, s1 + spacer_size)
    dn_e  = min(n - 1, s1 + spacer_size + flank_size)

    # Plot range with padding
    pad  = max(20, flank_size // 4)
    x0_plot = max(0, up_s - pad)
    x1_plot = min(n - 1, dn_e + pad)
    x_range = np.arange(x0_plot, x1_plot + 1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Smoothed Perplexity", "PDS"))

    # Row 1: smoothed perplexity + window bands
    fig.add_trace(go.Scatter(
        x=x_range, y=smoothed_di[x0_plot: x1_plot + 1], mode="lines",
        name="Smoothed perplexity",
        line=dict(color=_BLUE, width=2.2),
    ), row=1, col=1)

    band_defs = [
        (up_s,  up_e,  "rgba(245,158,11,0.18)",  "Upstream flank"),
        (up_e,  s0,    "rgba(148,163,184,0.12)",  "Spacer"),
        (s0,    s1,    "rgba(30,58,138,0.18)",    "Candidate valley"),
        (s1,    dn_s,  "rgba(148,163,184,0.12)",  "Spacer"),
        (dn_s,  dn_e,  "rgba(245,158,11,0.18)",  "Downstream flank"),
    ]
    shown = set()
    for bx0, bx1, fill, label in band_defs:
        fig.add_vrect(
            x0=bx0, x1=bx1, fillcolor=fill,
            line_width=0,
            annotation_text=label if label not in shown else "",
            annotation_position="top left",
            annotation_font=dict(size=9, color=_MUTED),
            row=1, col=1,
        )
        shown.add(label)

    # Row 2: PDS
    if len(pds) > 0:
        pds_slice = pds[x0_plot: x1_plot + 1].astype(float)
        pos_pds = np.where(pds_slice > 0, pds_slice, 0.0)
        fig.add_trace(go.Scatter(
            x=x_range, y=pos_pds, mode="lines",
            name="PDS",
            line=dict(color=_GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.18)",
        ), row=2, col=1)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.12)", line_width=0, row=2, col=1)
        fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"), row=2, col=1)

    fig.update_yaxes(title_text="Perplexity", row=1, col=1)
    fig.update_yaxes(title_text="PDS", row=2, col=1)
    fig.update_xaxes(title_text="Signal position (nt)", row=2, col=1)
    fig.update_layout(**_BASE_LAYOUT, title=f"<b>Three-Window Illustration — {domain.get('ID', 'valley')}</b>",
                      height=500)
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor="rgba(203,213,225,0.6)", linewidth=1,
        tickfont=dict(color=_MUTED, size=16),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(226,232,240,0.7)", zeroline=False,
        linecolor="rgba(203,213,225,0.6)", linewidth=1,
        tickfont=dict(color=_MUTED, size=16),
        title_font=dict(size=18),
    )
    return fig


# ─── Plot 4: PDS Landscape ───────────────────────────────────────────────────

def plot_pds_landscape(pds: np.ndarray, domains: list[dict]) -> go.Figure:
    """Genome-wide PDS profile with detected valleys highlighted."""
    if len(pds) == 0:
        return _empty_figure("PDS Landscape")
    x = np.arange(len(pds))
    pds_f = pds.astype(float)
    pos   = np.where(pds_f > 0, pds_f, 0.0)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(go.Scatter(
        x=x, y=pds_f, mode="lines",
        name="PDS",
        line=dict(color=_TEAL, width=1.3),
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pos, mode="lines",
        name="PDS (positive)",
        line=dict(color=_GREEN, width=2.0),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
    ))
    _add_valley_shading(fig, domains)
    fig.update_layout(xaxis_title="Signal position (nt)", yaxis_title="PDS")
    return _apply_base(fig, "Perplexity Depression Score (PDS) Landscape", height=400)


# ─── Plot 5: Valley Ranking ───────────────────────────────────────────────────

def plot_valley_ranking(domains: list[dict]) -> go.Figure:
    """Bar chart of valleys ranked by ValleyScore (highest = best)."""
    if not domains:
        return _empty_figure("Valley Ranking")
    ranked = sorted(domains, key=lambda d: d.get("ValleyScore", 0.0), reverse=True)
    ids    = [d.get("ID", "PV") for d in ranked]
    scores = [d.get("ValleyScore", 0.0) for d in ranked]
    pds    = [d.get("PDSMean", 0.0) for d in ranked]
    lens   = [d.get("Length", 0) for d in ranked]

    fig = go.Figure(go.Bar(
        x=ids, y=scores,
        marker=dict(
            color=pds,
            colorscale=[[0.0, "#e0e7ff"], [0.5, _TEAL], [1.0, _BLUE]],
            colorbar=dict(title="PDSMean"),
        ),
        customdata=np.array([[l, p] for l, p in zip(lens, pds)], dtype=object),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "ValleyScore: %{y:.4f}<br>"
            "Length: %{customdata[0]} bp<br>"
            "PDSMean: %{customdata[1]:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Valley Score")
    return _apply_base(fig, "Valley Ranking by ValleyScore", height=380)


# ─── Plot 6: Motif Architecture ──────────────────────────────────────────────

def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    """Bar chart of motif counts per valley."""
    if not domains:
        return _empty_figure("Motif Mapping")
    ranked = sorted(domains, key=lambda d: d.get("MotifCount", 0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[d.get("ID", "PV") for d in ranked],
        y=[d.get("MotifCount", 0) for d in ranked],
        marker=dict(color=_INDIGO),
        customdata=[d.get("Motifs", "") or "—" for d in ranked],
        hovertemplate="%{x}<br>MotifCount: %{y}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Motif count")
    return _apply_base(fig, "Motif Mapping", height=360)


# ─── Workflow diagram ─────────────────────────────────────────────────────────

def plot_algorithm_workflow() -> go.Figure:
    labels = [
        "DNA Input",
        "Dinucleotide Perplexity\n(window = 17 nt)",
        "Savitzky–Golay Smoothing\n(window=21, order=3)",
        "Perplexity Depression Score (PDS)\nThree-window contrast",
        "Bounded Kadane\nValley Detection",
        "Valley Expansion\n(PDS > 0 or > 20% peak)",
        "Valley Merging\n(gap ≤ 100 bp)",
        "Valley Metrics\n& ValleyScore",
        "Optional Motif\nAnnotation",
        "Downloads",
    ]
    colors = [
        _BLUE, _BLUE, _AMBER, _TEAL, _GREEN, _GREEN, _GREEN, _INDIGO, _TEAL, _CRIMSON,
    ]
    n = len(labels)
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels, pad=18, thickness=16,
            line=dict(color=_GRID, width=1),
            color=colors[:n],
        ),
        link=dict(
            source=list(range(n - 1)),
            target=list(range(1, n)),
            value=[4] * (n - 1),
            color=["rgba(30,58,138,0.15)"] * (n - 1),
        ),
    ))
    return _apply_base(fig, "REGPLEX v13 Algorithm Workflow", height=540)
