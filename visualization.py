from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared design tokens
# ---------------------------------------------------------------------------
_BG = "#071126"
_BG2 = "#0d1f3a"
_GRID = "rgba(120,157,201,0.12)"
_TEXT = "#e7edf7"
_MUTED = "#9fb3cc"
_CYAN = "#35d0ff"
_PURPLE = "#8a7dff"
_GOLD = "#ffc857"
_GREEN = "#36f7b0"
_PINK = "#ff6b9d"

# Class colour mapping (I = gold, II = cyan, III = purple, IV = teal)
_CLASS_COLOR = {"I": _GOLD, "II": _CYAN, "III": _PURPLE, "IV": _GREEN}
# Pre-defined rgba fill colours for domain class vrects (opacity 0.12)
_CLASS_FILL = {
    "I":   "rgba(255,200,87,0.12)",
    "II":  "rgba(53,208,255,0.12)",
    "III": "rgba(138,125,255,0.12)",
    "IV":  "rgba(54,247,176,0.12)",
}
_CLASS_DEFAULT = "#4a90e2"

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=_BG,
    plot_bgcolor=_BG2,
    font=dict(family="'Inter', 'Segoe UI', sans-serif", color=_TEXT, size=12),
    title_font=dict(size=15, color=_CYAN),
    margin=dict(l=60, r=30, t=55, b=50),
    xaxis=dict(
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(color=_MUTED)
    ),
    yaxis=dict(
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(color=_MUTED)
    ),
    legend=dict(bgcolor="rgba(7,17,38,0.6)", bordercolor=_GRID, borderwidth=1),
    hoverlabel=dict(
        bgcolor="#0d1f3a", bordercolor=_CYAN,
        font=dict(color=_TEXT, size=12),
    ),
)


def _apply_base(fig: go.Figure, title: str = "", height: int = 0) -> go.Figure:
    layout = dict(_BASE_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color=_CYAN))
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Figure 1 · Perplexity Profile
# ---------------------------------------------------------------------------
def plot_p1_profile(p1: np.ndarray) -> go.Figure:
    x = np.arange(len(p1))
    fig = go.Figure()
    # Gradient fill under the curve
    fig.add_trace(go.Scatter(
        x=x, y=p1, mode="lines",
        name="P1",
        line=dict(color=_CYAN, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(53,208,255,0.10)",
        hovertemplate="Position: %{x}<br>P1: %{y:.4f}<extra></extra>",
    ))
    _apply_base(fig, "Fig. 1 · Perplexity Profile (P1)", height=300)
    fig.update_layout(
        xaxis_title="Genomic Position",
        yaxis_title="P1 (bits)",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 · PDI Profile
# ---------------------------------------------------------------------------
def plot_pdi_profile(pdi: np.ndarray, domains: list[dict]) -> go.Figure:
    x = np.arange(len(pdi))
    fig = go.Figure()
    # Zero reference line
    fig.add_hline(y=0, line=dict(color=_GRID, width=1, dash="dot"))
    # Positive PDI (elevated) fill
    pos = np.where(pdi > 0, pdi, 0)
    neg = np.where(pdi < 0, pdi, 0)
    fig.add_trace(go.Scatter(
        x=x, y=pos, mode="lines",
        name="PDI ↑",
        line=dict(color=_PURPLE, width=1),
        fill="tozeroy",
        fillcolor="rgba(138,125,255,0.18)",
        hovertemplate="Position: %{x}<br>PDI: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=neg, mode="lines",
        name="PDI ↓",
        line=dict(color=_PINK, width=0.8),
        fill="tozeroy",
        fillcolor="rgba(255,107,157,0.08)",
        showlegend=False,
        hovertemplate="Position: %{x}<br>PDI: %{y:.4f}<extra></extra>",
    ))
    # Domain highlights (colour by class)
    for d in domains:
        cls = d.get("Class", "IV")
        col = _CLASS_COLOR.get(cls, _CLASS_DEFAULT)
        fill = _CLASS_FILL.get(cls, "rgba(74,144,226,0.12)")
        fig.add_vrect(
            x0=d["Start"], x1=d["End"],
            fillcolor=fill,
            line_width=1,
            line_color=col,
            opacity=0.5,
            annotation_text=cls,
            annotation_position="top left",
            annotation_font=dict(size=9, color=col),
        )
    _apply_base(fig, "Fig. 2 · Perplexity Depression Index (PDI)", height=320)
    fig.update_layout(
        xaxis_title="Genomic Position",
        yaxis_title="PDI",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 · Domain Map
# ---------------------------------------------------------------------------
def plot_domain_map(seq_length: int, domains: list[dict]) -> go.Figure:
    fig = go.Figure()
    # Chromosome backbone
    fig.add_shape(
        type="rect", x0=0, x1=seq_length, y0=0.42, y1=0.58,
        fillcolor="rgba(120,157,201,0.15)",
        line=dict(color=_GRID, width=1),
        layer="below",
    )
    # Tick every ~10 % of length
    tick_every = max(1, seq_length // 10)
    for tick in range(0, seq_length + 1, tick_every):
        fig.add_shape(
            type="line", x0=tick, x1=tick, y0=0.38, y1=0.42,
            line=dict(color=_MUTED, width=0.8),
        )
    # Domain blocks, coloured by class, with hover
    for d in domains:
        cls = d.get("Class", "IV")
        col = _CLASS_COLOR.get(cls, _CLASS_DEFAULT)
        rcs = d.get("RCS", 0.0)
        alpha = 0.45 + 0.55 * rcs
        fig.add_shape(
            type="rect",
            x0=d["Start"], x1=d["End"], y0=0.28, y1=0.72,
            fillcolor=col,
            opacity=float(alpha),
            line=dict(color=col, width=1),
        )
        # Invisible scatter for hover tooltip
        fig.add_trace(go.Scatter(
            x=[(d["Start"] + d["End"]) / 2],
            y=[0.5],
            mode="markers",
            marker=dict(color=col, size=1, opacity=0),
            name=d["Domain_ID"],
            hovertemplate=(
                f"<b>{d['Domain_ID']}</b> (Class {cls})<br>"
                f"Start: {d['Start']:,}  End: {d['End']:,}<br>"
                f"Length: {d['Length']:,} bp<br>"
                f"RCS: {rcs:.4f}<extra></extra>"
            ),
        ))
    _apply_base(fig, "Fig. 3 · Regulatory Domain Map", height=240)
    fig.update_layout(
        xaxis_title="Genomic Coordinate (bp)",
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4 · Domain Ranking
# ---------------------------------------------------------------------------
def plot_domain_ranking(domains: list[dict]) -> go.Figure:
    if not domains:
        fig = go.Figure()
        return _apply_base(fig, "Fig. 4 · Domain Ranking (RCS)")
    sorted_d = sorted(domains, key=lambda d: d.get("RCS", 0), reverse=True)
    labels = [d["Domain_ID"] for d in sorted_d]
    rcs = [d.get("RCS", 0) for d in sorted_d]
    classes = [d.get("Class", "IV") for d in sorted_d]
    colors = [_CLASS_COLOR.get(c, _CLASS_DEFAULT) for c in classes]
    fig = go.Figure(go.Bar(
        x=labels, y=rcs,
        marker=dict(
            color=rcs,
            colorscale=[[0, _PURPLE], [0.5, _CYAN], [1.0, _GOLD]],
            showscale=True,
            colorbar=dict(
                title="RCS",
                tickfont=dict(color=_MUTED),
                titlefont=dict(color=_MUTED),
            ),
            line=dict(color=colors, width=1.2),
        ),
        hovertemplate=(
            "<b>%{x}</b><br>RCS: %{y:.4f}<extra></extra>"
        ),
    ))
    _apply_base(fig, "Fig. 4 · Domain Ranking by Regulatory Complexity Score", height=340)
    fig.update_layout(
        xaxis_title="Domain ID",
        yaxis_title="RCS (normalised)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color=_MUTED)),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5 · Domain Statistics
# ---------------------------------------------------------------------------
def plot_domain_statistics(domains: list[dict]) -> go.Figure:
    subplot_titles = [
        "Domain Length Distribution",
        "Mean PDI Distribution",
        "P1 Variance Distribution",
        "RCS Distribution",
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    hist_cfg = dict(
        opacity=0.85,
        marker_line_width=0.6,
    )
    datasets = [
        ([d["Length"] for d in domains], _CYAN, "Length (bp)", 1, 1),
        ([d.get("Mean_PDI", 0) for d in domains], _PURPLE, "Mean PDI", 1, 2),
        ([d.get("Variance_P1", 0) for d in domains], _GOLD, "Variance P1", 2, 1),
        ([d.get("RCS", 0) for d in domains], _GREEN, "RCS", 2, 2),
    ]
    for vals, col, name, row, col_num in datasets:
        fig.add_trace(
            go.Histogram(
                x=vals, name=name,
                marker=dict(color=col, **hist_cfg),
                hovertemplate=f"{name}: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=row, col=col_num,
        )
    _apply_base(fig, "Fig. 5 · Domain Statistics", height=500)
    fig.update_layout(showlegend=False)
    # Style subplot axes
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(color=_MUTED), row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1
        )
        fig.update_yaxes(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(color=_MUTED), row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1
        )
    for ann in fig.layout.annotations:
        ann.font.color = _MUTED
        ann.font.size = 12
    return fig


# ---------------------------------------------------------------------------
# Figure 6 · Motif Architecture
# ---------------------------------------------------------------------------
def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    if not domains:
        fig = go.Figure()
        return _apply_base(fig, "Fig. 6 · Motif Architecture")
    sorted_d = sorted(domains, key=lambda d: d.get("Motif_Count", 0), reverse=True)
    labels = [d["Domain_ID"] for d in sorted_d]
    counts = [d.get("Motif_Count", 0) for d in sorted_d]
    motif_names = [d.get("Motifs", "") or "—" for d in sorted_d]
    classes = [d.get("Class", "IV") for d in sorted_d]
    colors = [_CLASS_COLOR.get(c, _CLASS_DEFAULT) for c in classes]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker=dict(color=colors, opacity=0.85, line=dict(color=_GRID, width=0.8)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Motif count: %{y}<br>"
            "Motifs: %{customdata}<extra></extra>"
        ),
        customdata=motif_names,
    ))
    _apply_base(fig, "Fig. 6 · Non-B DNA Motif Architecture", height=340)
    fig.update_layout(
        xaxis_title="Domain ID",
        yaxis_title="Motif Hit Count",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color=_MUTED)),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 7 · Algorithm Illustration (Sankey)
# ---------------------------------------------------------------------------
def plot_algorithm_illustration() -> go.Figure:
    labels = ["DNA Input", "10-mer P1", "PDI Signal", "Kadane Scan", "Ranked Domains", "Motif Hits"]
    node_colors = [_CYAN, _PURPLE, _GOLD, _GREEN, _PINK, _CYAN]
    link_colors = [
        "rgba(53,208,255,0.25)",
        "rgba(138,125,255,0.25)",
        "rgba(255,200,87,0.25)",
        "rgba(54,247,176,0.25)",
        "rgba(255,107,157,0.25)",
    ]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=labels,
            color=node_colors,
            pad=28,
            thickness=22,
            line=dict(color=_GRID, width=0.5),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[1, 2, 3, 4, 5],
            value=[4, 4, 4, 4, 4],
            color=link_colors,
            hovertemplate="Flow: %{source.label} → %{target.label}<extra></extra>",
        ),
    ))
    _apply_base(fig, "Fig. 7 · REGPLEX Algorithm Pipeline", height=340)
    fig.update_layout(font_size=12)
    return fig
