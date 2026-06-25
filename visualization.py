from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_BG = "#ffffff"
_GRID = "rgba(148,163,184,0.22)"
_TEXT = "#1f2937"
_MUTED = "#475569"
_BLUE = "#1E3A8A"
_TEAL = "#0F766E"
_GREEN = "#16A34A"
_AMBER = "#F59E0B"
_CRIMSON = "#DC2626"
_BLUE_ALT = "#2563eb"
_BLUE_DARK = "#1d4ed8"

_BASE_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, Segoe UI, sans-serif", size=13),
    title_font=dict(size=18, color=_TEXT),
    margin=dict(l=62, r=30, t=66, b=52),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=_GRID, borderwidth=1),
)


def _apply_base(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, title=title, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=_GRID, tickfont=dict(color=_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=_GRID, tickfont=dict(color=_MUTED))
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return _apply_base(fig, title)


def _domain_signal_bounds(domain: dict) -> tuple[int, int]:
    return int(domain.get("Signal_Start", domain.get("Start", 0))), int(domain.get("Signal_End", domain.get("End", 0)))


def plot_p1_profile(p1: np.ndarray) -> go.Figure:
    if len(p1) == 0:
        return _empty_figure("Perplexity Profile")
    x = np.arange(len(p1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p1, mode="lines", name="P1", line=dict(color=_BLUE, width=1.9)))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="P1")
    return _apply_base(fig, "Perplexity Profile")


def plot_p2_landscape(landscape: np.ndarray) -> go.Figure:
    if len(landscape) == 0:
        return _empty_figure("Perplexity Landscape")
    x = np.arange(len(landscape))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=landscape,
        mode="lines",
        name="Landscape",
        line=dict(color=_TEAL, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(15,118,110,0.10)",
    ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Landscape perplexity")
    return _apply_base(fig, "Perplexity Landscape")


def plot_three_window_illustration(
    landscape: np.ndarray,
    lpc: np.ndarray,
    params: dict,
    domains: list[dict],
    candidate_windows: np.ndarray,
) -> go.Figure:
    if len(landscape) == 0:
        return _empty_figure("Three-Window Contrast")
    if domains:
        anchor = int((domains[0]["Signal_Start"] + domains[0]["Signal_End"]) // 2)
    elif np.isfinite(lpc).any():
        anchor = int(np.nanargmax(lpc))
    else:
        anchor = len(landscape) // 2

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
    right = min(len(landscape) - 1, downstream_end + 25)
    x = np.arange(left, right + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=landscape[left:right + 1], mode="lines", name="Landscape", line=dict(color=_BLUE, width=2.2)))

    regions = [
        (upstream_start, upstream_end, "Upstream", "rgba(30,58,138,0.10)"),
        (candidate_start - spacer, candidate_start - 1, "Spacer", "rgba(148,163,184,0.12)"),
        (candidate_start, candidate_end, "Candidate", "rgba(15,118,110,0.16)"),
        (candidate_end + 1, candidate_end + spacer, "Spacer", "rgba(148,163,184,0.12)"),
        (downstream_start, downstream_end, "Downstream", "rgba(245,158,11,0.14)"),
    ]
    for start, end, label, fill in regions:
        if end < left or start > right:
            continue
        s = max(left, start)
        e = min(right, end)
        fig.add_vrect(x0=s, x1=e, fillcolor=fill, line_width=0)
        fig.add_annotation(x=(s + e) / 2, y=1.04, yref="paper", text=label, showarrow=False, font=dict(color=_MUTED, size=11))

    fig.update_layout(xaxis_title="Signal position", yaxis_title="Landscape perplexity")
    return _apply_base(fig, "Three-Window Contrast")


def plot_pvs_profile(lpc: np.ndarray, domains: list[dict]) -> go.Figure:
    if len(lpc) == 0:
        return _empty_figure("Local Contrast")
    x = np.arange(len(lpc))
    pos = np.where(lpc > 0, lpc, 0)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(go.Scatter(x=x, y=pos, mode="lines", name="LPC", line=dict(color=_GREEN, width=1.9), fill="tozeroy", fillcolor="rgba(22,163,74,0.14)"))
    for domain in domains:
        start, end = _domain_signal_bounds(domain)
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(30,58,138,0.16)", line_color=_BLUE, opacity=0.30)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="LPC")
    return _apply_base(fig, "Local Contrast")


def plot_kadane_domains(lpc: np.ndarray, domains: list[dict]) -> go.Figure:
    if len(lpc) == 0:
        return _empty_figure("Kadane Segments")
    x = np.arange(len(lpc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=lpc, mode="lines", name="LPC", line=dict(color="rgba(31,41,55,0.35)", width=1.15)))
    for domain in domains:
        start, end = _domain_signal_bounds(domain)
        fig.add_trace(go.Scatter(
            x=np.arange(start, end + 1),
            y=lpc[start:end + 1],
            mode="lines",
            name=domain.get("ID", "Valley"),
            line=dict(color=_BLUE, width=2.5),
            showlegend=False,
        ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="LPC")
    return _apply_base(fig, "Kadane Segments")


def plot_domain_ranking(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Valley Ranking")
    ranked = sorted(domains, key=lambda d: d.get("ValleyScore", 0.0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[d.get("ID", "PV") for d in ranked],
        y=[d.get("ValleyScore", 0.0) for d in ranked],
        marker=dict(color=[d.get("MeanLPC", 0.0) for d in ranked], colorscale=[[0.0, "#cbd5e1"], [0.5, _TEAL], [1.0, _BLUE]], colorbar=dict(title="Mean LPC")),
        customdata=np.array([[d.get("Length", 0), d.get("Persistence", 0.0)] for d in ranked], dtype=object),
        hovertemplate="%{x}<br>ValleyScore %{y:.4f}<br>Length %{customdata[0]} bp<br>Persistence %{customdata[1]:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Valley Score")
    return _apply_base(fig, "Valley Ranking", height=390)


def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Motif Mapping")
    ranked = sorted(domains, key=lambda d: d.get("MotifCount", 0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[d.get("ID", "PV") for d in ranked],
        y=[d.get("MotifCount", 0) for d in ranked],
        marker=dict(color=_TEAL),
        customdata=[d.get("Motifs", "") or "—" for d in ranked],
        hovertemplate="%{x}<br>MotifCount %{y}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Motif count")
    return _apply_base(fig, "Motif Mapping", height=360)


def plot_algorithm_workflow() -> go.Figure:
    labels = [
        "DNA sequence",
        "P1 (10-mer dinucleotide perplexity)",
        "Perplexity landscape",
        "Local Perplexity Contrast",
        "Adaptive Window Selection",
        "Bounded Kadane",
        "Perplexity valleys",
        "Motif Annotation",
    ]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=20, thickness=17, line=dict(color=_GRID, width=1), color=[_BLUE, _TEAL, _BLUE_ALT, _GREEN, _AMBER, _BLUE_DARK, _BLUE, _TEAL]),
        link=dict(source=list(range(len(labels) - 1)), target=list(range(1, len(labels))), value=[4] * (len(labels) - 1), color=["rgba(30,58,138,0.20)"] * (len(labels) - 1)),
    ))
    return _apply_base(fig, "REGPLEX Workflow", height=410)
