from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_BG = "#ffffff"
_GRID = "rgba(100,116,139,0.18)"
_TEXT = "#0f172a"
_MUTED = "#475569"
_BLUE = "#2563eb"
_SKY = "#0ea5e9"
_TEAL = "#14b8a6"
_GREEN = "#16a34a"
_ORANGE = "#f97316"
_RED = "#ef4444"
_FILL = "rgba(37,99,235,0.12)"

_BASE_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Arial, Helvetica, sans-serif", size=12),
    title_font=dict(size=16, color=_TEXT),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=_GRID, borderwidth=1),
)


def _apply_base(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, title=title, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=_GRID, tickfont=dict(color=_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=_GRID, tickfont=dict(color=_MUTED))
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return _apply_base(fig, title)


def _domain_signal_bounds(domain: dict) -> tuple[int, int]:
    return int(domain.get("Signal_Start", domain["Start"])), int(domain.get("Signal_End", domain["End"]))


def plot_p1_profile(p1: np.ndarray) -> go.Figure:
    if len(p1) == 0:
        return _empty_figure("Figure 1 · Raw Perplexity Profile")
    x = np.arange(len(p1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=p1,
        mode="lines",
        name="P1",
        line=dict(color=_BLUE, width=1.8),
        fill="tozeroy",
        fillcolor=_FILL,
        hovertemplate="Position %{x}<br>P1 %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="P1")
    return _apply_base(fig, "Figure 1 · Raw Perplexity Profile")


def plot_p2_landscape(p2: np.ndarray) -> go.Figure:
    if len(p2) == 0:
        return _empty_figure("Figure 2 · Smoothed Landscape")
    x = np.arange(len(p2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=p2,
        mode="lines",
        name="Smoothed P1",
        line=dict(color=_SKY, width=2),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.10)",
        hovertemplate="Position %{x}<br>Smoothed P1 %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Smoothed perplexity")
    return _apply_base(fig, "Figure 2 · Smoothed Landscape")


def plot_three_window_illustration(
    p2: np.ndarray,
    pvs: np.ndarray,
    params: dict,
    domains: list[dict],
    candidate_windows: np.ndarray,
) -> go.Figure:
    if len(p2) == 0:
        return _empty_figure("Figure 3 · Three-Window Valley Illustration")
    if domains:
        anchor = int((domains[0]["Signal_Start"] + domains[0]["Signal_End"]) // 2)
    elif np.isfinite(pvs).any():
        anchor = int(np.nanargmax(pvs))
    else:
        anchor = len(p2) // 2
    candidate_window = int(candidate_windows[anchor]) if anchor < len(candidate_windows) and candidate_windows[anchor] > 0 else int(params.get("candidate_window", 100))
    upstream = int(params.get("upstream_window", 100))
    downstream = int(params.get("downstream_window", 100))
    spacer = int(params.get("spacer", 50))
    candidate_start = anchor - (candidate_window // 2)
    candidate_end = candidate_start + candidate_window - 1
    upstream_start = candidate_start - spacer - upstream
    upstream_end = upstream_start + upstream - 1
    downstream_start = candidate_end + spacer + 1
    downstream_end = downstream_start + downstream - 1
    left = max(0, upstream_start - 25)
    right = min(len(p2) - 1, downstream_end + 25)
    x = np.arange(left, right + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=p2[left:right + 1],
        mode="lines",
        name="P2",
        line=dict(color=_BLUE, width=2),
        hovertemplate="Position %{x}<br>P2 %{y:.4f}<extra></extra>",
    ))
    regions = [
        (upstream_start, upstream_end, "Upstream", "rgba(37,99,235,0.12)"),
        (candidate_start - spacer, candidate_start - 1, "Spacer", "rgba(148,163,184,0.10)"),
        (candidate_start, candidate_end, "Candidate", "rgba(20,184,166,0.16)"),
        (candidate_end + 1, candidate_end + spacer, "Spacer", "rgba(148,163,184,0.10)"),
        (downstream_start, downstream_end, "Downstream", "rgba(249,115,22,0.12)"),
    ]
    for start, end, label, fill in regions:
        fig.add_vrect(x0=max(left, start), x1=min(right, end), fillcolor=fill, line_width=0)
        fig.add_annotation(x=(max(left, start) + min(right, end)) / 2, y=1.04, yref="paper", text=label, showarrow=False, font=dict(color=_MUTED, size=11))
    if 0 <= anchor < len(pvs) and np.isfinite(pvs[anchor]):
        fig.add_trace(go.Scatter(
            x=[anchor],
            y=[p2[anchor]],
            mode="markers",
            marker=dict(color=_RED, size=9),
            name="Valley anchor",
            hovertemplate=f"Anchor {anchor}<br>PVS {pvs[anchor]:.4f}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="P2")
    return _apply_base(fig, "Figure 3 · Three-Window Valley Illustration")


def plot_pvs_profile(pvs: np.ndarray, domains: list[dict]) -> go.Figure:
    if len(pvs) == 0:
        return _empty_figure("Figure 4 · Genome-wide Valley Score")
    x = np.arange(len(pvs))
    positive = np.where(pvs > 0, pvs, 0)
    negative = np.where(pvs < 0, pvs, 0)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(go.Scatter(
        x=x,
        y=positive,
        mode="lines",
        name="Positive PVS",
        line=dict(color=_GREEN, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(22,163,74,0.14)",
        hovertemplate="Position %{x}<br>PVS %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=negative,
        mode="lines",
        name="Negative PVS",
        line=dict(color=_ORANGE, width=1.1),
        fill="tozeroy",
        fillcolor="rgba(249,115,22,0.10)",
        hovertemplate="Position %{x}<br>PVS %{y:.4f}<extra></extra>",
    ))
    for domain in domains:
        start, end = _domain_signal_bounds(domain)
        opacity = 0.15 + (0.45 * float(domain.get("Confidence", 0.0)))
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(37,99,235,0.12)", line_color=_BLUE, opacity=opacity)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="PVS")
    return _apply_base(fig, "Figure 4 · Genome-wide Valley Score")


def plot_kadane_domains(pvs: np.ndarray, domains: list[dict]) -> go.Figure:
    if len(pvs) == 0:
        return _empty_figure("Figure 5 · Kadane Domain Selection")
    x = np.arange(len(pvs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=pvs,
        mode="lines",
        name="PVS",
        line=dict(color="rgba(15,23,42,0.35)", width=1.1),
        hovertemplate="Position %{x}<br>PVS %{y:.4f}<extra></extra>",
    ))
    for domain in domains:
        start, end = _domain_signal_bounds(domain)
        fig.add_trace(go.Scatter(
            x=np.arange(start, end + 1),
            y=pvs[start:end + 1],
            mode="lines",
            name=domain["Domain_ID"],
            line=dict(color=_BLUE, width=2 + (3 * float(domain.get("Confidence", 0.0)))),
            hovertemplate=(
                f"{domain['Domain_ID']}<br>Signal {start}-{end}<br>"
                f"Mean PVS {domain.get('Mean_PVS', float('nan')):.4f}<br>"
                f"Confidence {domain.get('Confidence', 0.0):.4f}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="PVS")
    return _apply_base(fig, "Figure 5 · Kadane Domain Selection")


def plot_domain_ranking(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Figure 6 · Domain Ranking")
    ranked = sorted(domains, key=lambda domain: domain.get("Confidence", 0.0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[domain["Domain_ID"] for domain in ranked],
        y=[domain.get("Confidence", 0.0) for domain in ranked],
        marker=dict(
            color=[domain.get("Mean_PVS", 0.0) for domain in ranked],
            colorscale=[[0.0, "#bfdbfe"], [0.5, _SKY], [1.0, _BLUE]],
            colorbar=dict(title="Mean PVS"),
        ),
        customdata=np.array([[domain.get("Length", 0), domain.get("Persistence", 0.0)] for domain in ranked], dtype=object),
        hovertemplate=(
            "%{x}<br>Confidence %{y:.4f}<br>Length %{customdata[0]} bp<br>"
            "Persistence %{customdata[1]:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="Domain ID", yaxis_title="Confidence (0-1)")
    return _apply_base(fig, "Figure 6 · Domain Ranking", height=360)


def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Figure 7 · Motif Architecture")
    ranked = sorted(domains, key=lambda domain: domain.get("Motif_Count", 0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[domain["Domain_ID"] for domain in ranked],
        y=[domain.get("Motif_Count", 0) for domain in ranked],
        marker=dict(color=_TEAL),
        customdata=[domain.get("Motifs", "") or "—" for domain in ranked],
        hovertemplate="%{x}<br>Motif count %{y}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Domain ID", yaxis_title="Motif count")
    return _apply_base(fig, "Figure 7 · Motif Architecture", height=340)


def plot_algorithm_workflow() -> go.Figure:
    labels = [
        "DNA sequence",
        "P1 (10-mer dinucleotide perplexity)",
        "Smoothed landscape",
        "Three-window valley analysis",
        "PVS (Contrast × Symmetry)",
        "Bounded Kadane",
        "Valley domains",
        "Confidence scoring",
        "Optional motif annotation",
    ]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=20,
            thickness=18,
            line=dict(color=_GRID, width=1),
            color=[_BLUE, _SKY, _TEAL, "#60a5fa", _GREEN, _ORANGE, _BLUE, _TEAL, _SKY],
        ),
        link=dict(
            source=list(range(len(labels) - 1)),
            target=list(range(1, len(labels))),
            value=[4] * (len(labels) - 1),
            color=["rgba(37,99,235,0.18)"] * (len(labels) - 1),
        ),
    ))
    return _apply_base(fig, "Figure 8 · Complete Algorithm Workflow", height=360)
