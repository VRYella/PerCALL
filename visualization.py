from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BG = "#ffffff"
_GRID = "rgba(148,163,184,0.22)"
_TEXT = "#1f2937"
_MUTED = "#475569"
_BLUE = "#1E3A8A"
_TEAL = "#0F766E"
_GREEN = "#10B981"
_AMBER = "#F59E0B"
_CRIMSON = "#DC2626"
_INDIGO = "#4F46E5"

_BASE_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, Segoe UI, sans-serif", size=17),
    title_font=dict(size=25, color=_BLUE, family="Space Grotesk, Inter, sans-serif"),
    margin=dict(l=72, r=36, t=72, b=60),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=16)),
    hoverlabel=dict(font=dict(size=15)),
)


def _apply_base(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, title=f"<b>{title}</b>", height=height)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(203,213,225,0.6)", linewidth=1, tickfont=dict(color=_MUTED, size=16), title_font=dict(size=18))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(226,232,240,0.7)", zeroline=False, linecolor="rgba(203,213,225,0.6)", linewidth=1, tickfont=dict(color=_MUTED, size=16), title_font=dict(size=18))
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=_MUTED, size=14))
    return _apply_base(fig, title)


def _region_bounds(region: dict) -> tuple[int, int]:
    return int(region.get("Signal_Start", region.get("Start", 0))), int(region.get("Signal_End", region.get("End", 0)))


def _add_region_shading(fig: go.Figure, regions: list[dict]) -> None:
    for i, region in enumerate(regions):
        s0, s1 = _region_bounds(region)
        region_id = region.get("Region_ID", f"LPR_{i+1:06d}")
        fig.add_vrect(
            x0=s0,
            x1=s1,
            fillcolor="rgba(30,58,138,0.12)",
            line_color=_BLUE,
            line_width=0.8,
            opacity=0.8,
            annotation_text=region_id,
            annotation_position="top left",
            annotation_font=dict(size=10, color=_BLUE),
        )


def plot_perplexity_landscape(di: np.ndarray, regions: list[dict]) -> go.Figure:
    if len(di) == 0:
        return _empty_figure("Raw Dinucleotide Perplexity")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(di)), y=di, mode="lines", name="Di Perplexity (raw)", line=dict(color=_TEAL, width=1.4), opacity=0.85))
    _add_region_shading(fig, regions)
    fig.update_layout(xaxis_title="Signal position (nt)", yaxis_title="Perplexity")
    return _apply_base(fig, "Raw Dinucleotide Perplexity", height=390)


def plot_smoothed_perplexity(di: np.ndarray, smoothed_di: np.ndarray, regions: list[dict]) -> go.Figure:
    if len(di) == 0 and len(smoothed_di) == 0:
        return _empty_figure("Smoothed Perplexity")
    fig = go.Figure()
    if len(di) > 0:
        fig.add_trace(go.Scatter(x=np.arange(len(di)), y=di, mode="lines", name="Raw perplexity", line=dict(color=_TEAL, width=1.0, dash="dot"), opacity=0.4))
    if len(smoothed_di) > 0:
        fig.add_trace(go.Scatter(x=np.arange(len(smoothed_di)), y=smoothed_di, mode="lines", name="SG-smoothed perplexity", line=dict(color=_BLUE, width=2.2), opacity=0.9))
    _add_region_shading(fig, regions)
    fig.update_layout(xaxis_title="Signal position (nt)", yaxis_title="Perplexity")
    return _apply_base(fig, "Savitzky–Golay Smoothed Perplexity", height=390)


def plot_three_window(smoothed_di: np.ndarray, pds: np.ndarray, region: dict, flank_size: int = 100, spacer_size: int = 50) -> go.Figure:
    if len(smoothed_di) == 0:
        return _empty_figure("Three-Window PDS Layout")

    s0, s1 = _region_bounds(region)
    n = len(smoothed_di)
    up_s, up_e = max(0, s0 - spacer_size - flank_size), max(0, s0 - spacer_size)
    dn_s, dn_e = min(n - 1, s1 + spacer_size), min(n - 1, s1 + spacer_size + flank_size)
    pad = max(20, flank_size // 4)
    x0_plot, x1_plot = max(0, up_s - pad), min(n - 1, dn_e + pad)
    x_range = np.arange(x0_plot, x1_plot + 1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Smoothed Perplexity", "PDS"))
    fig.add_trace(go.Scatter(x=x_range, y=smoothed_di[x0_plot:x1_plot + 1], mode="lines", name="Smoothed perplexity", line=dict(color=_BLUE, width=2.2)), row=1, col=1)

    bands = [
        (up_s, up_e, "rgba(245,158,11,0.18)", "Upstream flank"),
        (up_e, s0, "rgba(148,163,184,0.12)", "Spacer"),
        (s0, s1, "rgba(30,58,138,0.18)", "Candidate region"),
        (s1, dn_s, "rgba(148,163,184,0.12)", "Spacer"),
        (dn_s, dn_e, "rgba(245,158,11,0.18)", "Downstream flank"),
    ]
    shown = set()
    for bx0, bx1, fill, label in bands:
        fig.add_vrect(x0=bx0, x1=bx1, fillcolor=fill, line_width=0, annotation_text=label if label not in shown else "", annotation_position="top left", annotation_font=dict(size=9, color=_MUTED), row=1, col=1)
        shown.add(label)

    if len(pds) > 0:
        pds_slice = pds[x0_plot:x1_plot + 1].astype(float)
        pos_pds = np.where(pds_slice > 0, pds_slice, 0.0)
        fig.add_trace(go.Scatter(x=x_range, y=pos_pds, mode="lines", name="PDS", line=dict(color=_GREEN, width=2), fill="tozeroy", fillcolor="rgba(16,185,129,0.18)"), row=2, col=1)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.12)", line_width=0, row=2, col=1)
        fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"), row=2, col=1)

    fig.update_yaxes(title_text="Perplexity", row=1, col=1)
    fig.update_yaxes(title_text="PDS", row=2, col=1)
    fig.update_xaxes(title_text="Signal position (nt)", row=2, col=1)
    fig.update_layout(**_BASE_LAYOUT, title=f"<b>Three-Window Illustration — {region.get('Region_ID', 'LPR')}</b>", height=520)
    return fig


def plot_pds_landscape(pds: np.ndarray, regions: list[dict]) -> go.Figure:
    if len(pds) == 0:
        return _empty_figure("PDS Landscape")
    pds_f = pds.astype(float)
    pos = np.where(pds_f > 0, pds_f, 0.0)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(go.Scatter(x=np.arange(len(pds)), y=pds_f, mode="lines", name="PDS", line=dict(color=_TEAL, width=1.3), opacity=0.55))
    fig.add_trace(go.Scatter(x=np.arange(len(pds)), y=pos, mode="lines", name="PDS (positive)", line=dict(color=_GREEN, width=2.0), fill="tozeroy", fillcolor="rgba(16,185,129,0.15)"))
    _add_region_shading(fig, regions)
    fig.update_layout(xaxis_title="Signal position (nt)", yaxis_title="PDS")
    return _apply_base(fig, "Perplexity Depression Score (PDS)", height=410)


def plot_region_ranking(regions: list[dict]) -> go.Figure:
    if not regions:
        return _empty_figure("Region Ranking")
    ranked = sorted(regions, key=lambda r: r.get("Region_Score", 0.0), reverse=True)
    ids = [r.get("Region_ID", "LPR") for r in ranked]
    scores = [r.get("Region_Score", 0.0) for r in ranked]
    pds = [r.get("Perplexity_Depression_Score", 0.0) for r in ranked]
    lens = [r.get("Length", 0) for r in ranked]

    fig = go.Figure(go.Bar(
        x=ids,
        y=scores,
        marker=dict(color=pds, colorscale=[[0.0, "#e0e7ff"], [0.5, _TEAL], [1.0, _BLUE]], colorbar=dict(title="Perplexity Depression Score")),
        customdata=np.array([[l, p] for l, p in zip(lens, pds)], dtype=object),
        hovertemplate="<b>%{x}</b><br>Region_Score: %{y:.4f}<br>Length: %{customdata[0]} bp<br>PDS: %{customdata[1]:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Region_ID", yaxis_title="Region Score")
    return _apply_base(fig, "Low Perplexity Region Ranking", height=390)


def plot_motif_architecture(regions: list[dict]) -> go.Figure:
    if not regions:
        return _empty_figure("Motif Annotation")
    ranked = sorted(regions, key=lambda r: r.get("Motif_Count", 0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[r.get("Region_ID", "LPR") for r in ranked],
        y=[r.get("Motif_Count", 0) for r in ranked],
        marker=dict(color=_INDIGO),
        customdata=[r.get("Motifs", "") or "—" for r in ranked],
        hovertemplate="%{x}<br>Motif_Count: %{y}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Region_ID", yaxis_title="Motif count")
    return _apply_base(fig, "Motif Annotation by Region", height=370)


def plot_algorithm_workflow() -> go.Figure:
    labels = [
        "DNA",
        "Dinucleotide Perplexity\n(window = 17 nt)",
        "Savitzky–Golay Smoothing",
        "Perplexity Depression Score (PDS)",
        "Bounded Kadane Optimization",
        "Region Expansion",
        "Region Merging",
        "Low Perplexity Region Ranking",
        "Optional Motif Annotation",
        "Downloads",
    ]
    colors = [_BLUE, _BLUE, _AMBER, _TEAL, _GREEN, _GREEN, _GREEN, _INDIGO, _TEAL, _CRIMSON]
    n = len(labels)
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=18, thickness=16, line=dict(color=_GRID, width=1), color=colors[:n]),
        link=dict(source=list(range(n - 1)), target=list(range(1, n)), value=[4] * (n - 1), color=["rgba(30,58,138,0.15)"] * (n - 1)),
    ))
    return _apply_base(fig, "REGPLEX Workflow", height=560)


