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

_SCALE_COLORS = [_BLUE, _TEAL, _GREEN, _AMBER, _CRIMSON, _BLUE_ALT, _BLUE_DARK, _MUTED]

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
    fig.update_xaxes(
        showgrid=True, gridcolor=_GRID, zeroline=False,
        linecolor=_GRID, tickfont=dict(color=_MUTED),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=_GRID, zeroline=False,
        linecolor=_GRID, tickfont=dict(color=_MUTED),
    )
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data available", x=0.5, y=0.5,
        xref="paper", yref="paper", showarrow=False,
    )
    return _apply_base(fig, title)


def _valley_bounds(domain: dict) -> tuple[int, int]:
    return (
        int(domain.get("Signal_Start", domain.get("Start", 0))),
        int(domain.get("Signal_End", domain.get("End", 0))),
    )


# ---------------------------------------------------------------------------
# 1 — Raw Perplexity
# ---------------------------------------------------------------------------
def plot_p1_profile(p1: np.ndarray) -> go.Figure:
    if len(p1) == 0:
        return _empty_figure("Perplexity Profile (P1)")
    x = np.arange(len(p1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p1, mode="lines", name="P1",
        line=dict(color=_BLUE, width=1.9),
    ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="P1")
    return _apply_base(fig, "Perplexity Profile (P1)")


# ---------------------------------------------------------------------------
# 2 — Multi-scale Landscapes
# ---------------------------------------------------------------------------
def plot_multiscale_landscapes(
    landscapes: dict, domains: list[dict]
) -> go.Figure:
    if not landscapes:
        return _empty_figure("Multi-scale Perplexity Landscapes")
    n = max((len(v) for v in landscapes.values()), default=0)
    if n == 0:
        return _empty_figure("Multi-scale Perplexity Landscapes")
    fig = go.Figure()
    for i, s in enumerate(sorted(landscapes)):
        arr = landscapes[s]
        col = _SCALE_COLORS[i % len(_SCALE_COLORS)]
        fig.add_trace(go.Scatter(
            x=np.arange(len(arr)), y=arr,
            mode="lines", name=f"{s} bp",
            line=dict(color=col, width=1.8), opacity=0.85,
        ))
    for d in domains:
        s0, s1 = _valley_bounds(d)
        fig.add_vrect(
            x0=s0, x1=s1,
            fillcolor="rgba(30,58,138,0.10)", line_width=0,
        )
    fig.update_layout(
        xaxis_title="Signal position",
        yaxis_title="Landscape perplexity",
    )
    return _apply_base(fig, "Multi-scale Perplexity Landscapes", height=400)


# ---------------------------------------------------------------------------
# 3 — Consensus LPC
# ---------------------------------------------------------------------------
def plot_consensus_lpc(
    consensus_lpc: np.ndarray, domains: list[dict]
) -> go.Figure:
    if len(consensus_lpc) == 0:
        return _empty_figure("Consensus LPC")
    x = np.arange(len(consensus_lpc))
    pos = np.where(consensus_lpc > 0, consensus_lpc, 0)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(go.Scatter(
        x=x, y=pos, mode="lines", name="Consensus LPC",
        line=dict(color=_GREEN, width=1.9),
        fill="tozeroy", fillcolor="rgba(22,163,74,0.14)",
    ))
    for d in domains:
        s0, s1 = _valley_bounds(d)
        fig.add_vrect(
            x0=s0, x1=s1,
            fillcolor="rgba(30,58,138,0.16)",
            line_color=_BLUE, opacity=0.30,
        )
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Consensus LPC")
    return _apply_base(fig, "Consensus LPC")


# ---------------------------------------------------------------------------
# 4 — Kadane Segments
# ---------------------------------------------------------------------------
def plot_kadane_domains(
    consensus_lpc: np.ndarray, domains: list[dict]
) -> go.Figure:
    if len(consensus_lpc) == 0:
        return _empty_figure("Kadane Segments")
    x = np.arange(len(consensus_lpc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=consensus_lpc, mode="lines", name="Consensus LPC",
        line=dict(color="rgba(31,41,55,0.35)", width=1.15),
    ))
    for domain in domains:
        s0, s1 = _valley_bounds(domain)
        fig.add_trace(go.Scatter(
            x=np.arange(s0, s1 + 1),
            y=consensus_lpc[s0: s1 + 1],
            mode="lines",
            name=domain.get("ID", "Valley"),
            line=dict(color=_BLUE, width=2.5),
            showlegend=False,
        ))
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Consensus LPC")
    return _apply_base(fig, "Kadane Segments")


# ---------------------------------------------------------------------------
# 5 — Scale Support Heatmap
# ---------------------------------------------------------------------------
def plot_scale_support_heatmap(
    lpc_profiles: dict, domains: list[dict], scales: list[int]
) -> go.Figure:
    if not domains or not lpc_profiles:
        return _empty_figure("Scale Support Heatmap")
    valley_ids = [d.get("ID", f"PV_{i:06d}") for i, d in enumerate(domains)]
    z: list[list[float]] = []
    for d in domains:
        s0, s1 = _valley_bounds(d)
        row: list[float] = []
        for sc in scales:
            lpc = lpc_profiles.get(sc)
            if lpc is None or s1 >= len(lpc):
                row.append(float("nan"))
                continue
            seg = lpc[s0: s1 + 1]
            finite = seg[np.isfinite(seg)]
            row.append(float(np.mean(finite)) if finite.size > 0 else float("nan"))
        z.append(row)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{sc} bp" for sc in scales],
        y=valley_ids,
        colorscale=[[0.0, "#cbd5e1"], [0.5, _TEAL], [1.0, _BLUE]],
        colorbar=dict(title="Mean LPC"),
        zmid=0,
        hovertemplate=(
            "Valley: %{y}<br>Scale: %{x}<br>Mean LPC: %{z:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="Observation scale", yaxis_title="Valley")
    h = max(320, 60 + 30 * len(domains))
    return _apply_base(fig, "Scale Support Heatmap", height=h)


# ---------------------------------------------------------------------------
# 6 — Valley Ranking
# ---------------------------------------------------------------------------
def plot_domain_ranking(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Valley Ranking")
    ranked = sorted(domains, key=lambda d: d.get("ValleyScore", 0.0), reverse=True)
    fig = go.Figure(go.Bar(
        x=[d.get("ID", "PV") for d in ranked],
        y=[d.get("ValleyScore", 0.0) for d in ranked],
        marker=dict(
            color=[d.get("MeanLPC", 0.0) for d in ranked],
            colorscale=[[0.0, "#cbd5e1"], [0.5, _TEAL], [1.0, _BLUE]],
            colorbar=dict(title="Mean LPC"),
        ),
        customdata=np.array(
            [[d.get("Length", 0), d.get("Persistence", 0.0),
              d.get("ScaleSupport", 0.0)] for d in ranked],
            dtype=object,
        ),
        hovertemplate=(
            "%{x}<br>ValleyScore %{y:.4f}<br>Length %{customdata[0]} bp"
            "<br>Persistence %{customdata[1]:.4f}"
            "<br>ScaleSupport %{customdata[2]:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Valley Score")
    return _apply_base(fig, "Valley Ranking", height=390)


# ---------------------------------------------------------------------------
# 7 — Motif Architecture
# ---------------------------------------------------------------------------
def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Motif Mapping")
    ranked = sorted(
        domains, key=lambda d: d.get("MotifCount", 0), reverse=True
    )
    fig = go.Figure(go.Bar(
        x=[d.get("ID", "PV") for d in ranked],
        y=[d.get("MotifCount", 0) for d in ranked],
        marker=dict(color=_TEAL),
        customdata=[d.get("Motifs", "") or "—" for d in ranked],
        hovertemplate="%{x}<br>MotifCount %{y}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Motif count")
    return _apply_base(fig, "Motif Mapping", height=360)


# ---------------------------------------------------------------------------
# 8 — Complete Workflow
# ---------------------------------------------------------------------------
def plot_algorithm_workflow() -> go.Figure:
    labels = [
        "DNA sequence",
        "P1 (10-mer dinucleotide perplexity)",
        "Multi-scale Landscapes [25–400 bp]",
        "Per-scale Three-window LPC",
        "Normalized LPC Profiles",
        "Consensus LPC (nanmedian)",
        "Bounded Kadane + Valley Expansion",
        "Valley Merging",
        "Perplexity Valleys",
        "Motif Annotation",
    ]
    n = len(labels)
    colors = [
        _BLUE, _TEAL, _BLUE_ALT, _GREEN, _AMBER,
        _BLUE_DARK, _TEAL, _BLUE, _TEAL, _GREEN,
    ]
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
            color=["rgba(30,58,138,0.18)"] * (n - 1),
        ),
    ))
    return _apply_base(fig, "REGPLEX v9 Workflow", height=460)
