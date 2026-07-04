from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

_BG = "#ffffff"
_GRID = "rgba(148,163,184,0.22)"
_TEXT = "#1f2937"
_MUTED = "#475569"
_BLUE = "#1E3A8A"
_TEAL = "#0F766E"
_GREEN = "#10B981"
_AMBER = "#F59E0B"
_CRIMSON = "#DC2626"
_SLATE = "#64748B"

_BASE_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, Segoe UI, sans-serif", size=13),
    title_font=dict(size=16, color=_TEXT),
    margin=dict(l=62, r=30, t=56, b=52),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=_GRID, borderwidth=1),
)


def _apply_base(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, title=title, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=_GRID,
                     tickfont=dict(color=_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, linecolor=_GRID,
                     tickfont=dict(color=_MUTED))
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data available", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False,
                       font=dict(color=_MUTED, size=14))
    return _apply_base(fig, title)


def _valley_signal_bounds(domain: dict) -> tuple[int, int]:
    return (
        int(domain.get("Signal_Start", domain.get("Start", 0))),
        int(domain.get("Signal_End", domain.get("End", 0))),
    )


def _add_valley_rects(fig: go.Figure, domains: list[dict]) -> None:
    for d in domains:
        s, e = _valley_signal_bounds(d)
        fig.add_vrect(x0=s, x1=e,
                      fillcolor="rgba(30,58,138,0.10)",
                      line_color=_BLUE, line_width=0.8,
                      opacity=0.6)


# ---------------------------------------------------------------------------
# Plot 1 – Raw perplexity landscape
# ---------------------------------------------------------------------------

def plot_perplexity_landscape(di: np.ndarray, domains: list[dict]) -> go.Figure:
    """Raw dinucleotide perplexity profile with valley highlights."""
    if len(di) == 0:
        return _empty_figure("Dinucleotide Perplexity Landscape")
    fig = go.Figure()
    x = np.arange(len(di))
    fig.add_trace(go.Scatter(
        x=x, y=di, mode="lines",
        name="Di-perplexity (raw)",
        line=dict(color=_BLUE, width=1.2),
        opacity=0.7,
    ))
    _add_valley_rects(fig, domains)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Perplexity")
    return _apply_base(fig, "Dinucleotide Perplexity Landscape", height=360)


# ---------------------------------------------------------------------------
# Plot 2 – Smoothed perplexity landscape
# ---------------------------------------------------------------------------

def plot_smoothed_perplexity_landscape(
    di: np.ndarray,
    smoothed_di: np.ndarray,
    domains: list[dict],
) -> go.Figure:
    """Savitzky-Golay smoothed perplexity with detected valley highlights."""
    if len(di) == 0 and len(smoothed_di) == 0:
        return _empty_figure("Smoothed Perplexity Landscape")
    fig = go.Figure()
    if len(di) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(di)), y=di, mode="lines",
            name="Raw", line=dict(color=_BLUE, width=1.0, dash="dot"), opacity=0.35,
        ))
    if len(smoothed_di) > 0:
        fig.add_trace(go.Scatter(
            x=np.arange(len(smoothed_di)), y=smoothed_di, mode="lines",
            name="SG-smoothed", line=dict(color=_TEAL, width=2.2), opacity=0.95,
        ))
    _add_valley_rects(fig, domains)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Perplexity")
    return _apply_base(fig, "Smoothed Perplexity Landscape (Savitzky–Golay)", height=380)


# ---------------------------------------------------------------------------
# Plot 3 – Three-window illustration
# ---------------------------------------------------------------------------

def plot_three_window(
    smoothed_di: np.ndarray,
    valley: dict,
    flank_size: int = 100,
    spacer_size: int = 50,
) -> go.Figure:
    """Three-window illustration for a selected valley.

    Shows upstream / spacer / candidate / spacer / downstream regions
    with shaded backgrounds and horizontal mean-value reference lines.
    """
    if len(smoothed_di) == 0 or valley is None:
        return _empty_figure("Three-Window Context")

    n = len(smoothed_di)
    s = int(valley.get("Signal_Start", valley.get("Start", 0)))
    e = int(valley.get("Signal_End", valley.get("End", 0)))

    # Window boundaries
    up_e = max(0, s - spacer_size)
    up_s = max(0, up_e - flank_size)
    dn_s = min(n, e + spacer_size)
    dn_e = min(n, dn_s + flank_size)

    # Display range
    view_s = max(0, up_s - 10)
    view_e = min(n, dn_e + 10)
    x = np.arange(view_s, view_e)
    y = smoothed_di[view_s:view_e]

    fig = go.Figure()

    # Shaded region backgrounds
    fig.add_vrect(x0=up_s, x1=up_e,
                  fillcolor="rgba(30,58,138,0.08)", line_width=0,
                  annotation_text="Upstream", annotation_position="top left",
                  annotation_font=dict(size=11, color=_BLUE))
    if up_e < s:
        fig.add_vrect(x0=up_e, x1=s, fillcolor="rgba(100,116,139,0.05)", line_width=0,
                      annotation_text="Spacer", annotation_position="top left",
                      annotation_font=dict(size=10, color=_MUTED))
    fig.add_vrect(x0=s, x1=e,
                  fillcolor="rgba(15,118,110,0.14)", line_color=_TEAL, line_width=1,
                  annotation_text="Candidate", annotation_position="top right",
                  annotation_font=dict(size=11, color=_TEAL))
    if e < dn_s:
        fig.add_vrect(x0=e, x1=dn_s, fillcolor="rgba(100,116,139,0.05)", line_width=0)
    fig.add_vrect(x0=dn_s, x1=dn_e,
                  fillcolor="rgba(30,58,138,0.08)", line_width=0,
                  annotation_text="Downstream", annotation_position="top right",
                  annotation_font=dict(size=11, color=_BLUE))

    # Smoothed signal
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        name="SG-smoothed perplexity",
        line=dict(color=_TEXT, width=1.8),
    ))

    # Horizontal mean reference lines
    def _safe_mean(arr: np.ndarray) -> float | None:
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)) if v.size else None

    up_mean = _safe_mean(smoothed_di[up_s:up_e])
    cand_mean = _safe_mean(smoothed_di[s:e])
    dn_mean = _safe_mean(smoothed_di[dn_s:dn_e])

    for label, x0, x1, mean_val, color in [
        ("UpstreamMean", up_s, up_e, up_mean, _BLUE),
        ("CandidateMean", s, e, cand_mean, _TEAL),
        ("DownstreamMean", dn_s, dn_e, dn_mean, _BLUE),
    ]:
        if mean_val is not None:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=mean_val, y1=mean_val,
                          line=dict(color=color, width=2, dash="dash"))
            fig.add_annotation(x=(x0 + x1) / 2, y=mean_val,
                                text=f"{label}: {mean_val:.2f}",
                                showarrow=False, yshift=10,
                                font=dict(size=10, color=color))

    valley_id = valley.get("ID", "")
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Perplexity")
    return _apply_base(fig, f"Three-Window Context — {valley_id}", height=400)


# ---------------------------------------------------------------------------
# Plot 4 – PDS landscape
# ---------------------------------------------------------------------------

def plot_pds_landscape(pds: np.ndarray, domains: list[dict]) -> go.Figure:
    """Genome-wide PDS profile with detected valley highlights."""
    if len(pds) == 0:
        return _empty_figure("Perplexity Depression Score (PDS) Landscape")
    x = np.arange(len(pds))
    pos = np.where(np.isfinite(pds) & (pds > 0), pds, 0.0)
    neg = np.where(np.isfinite(pds) & (pds <= 0), pds, 0.0)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(go.Scatter(
        x=x, y=pos, mode="lines",
        name="PDS > 0 (valley)",
        line=dict(color=_TEAL, width=1.8),
        fill="tozeroy", fillcolor="rgba(15,118,110,0.18)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=neg, mode="lines",
        name="PDS ≤ 0 (background)",
        line=dict(color=_SLATE, width=1.0),
        opacity=0.5,
    ))
    for d in domains:
        s, e = _valley_signal_bounds(d)
        fig.add_vrect(x0=s, x1=e,
                      fillcolor="rgba(30,58,138,0.18)", line_color=_BLUE, line_width=1,
                      opacity=0.7)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="PDS")
    return _apply_base(fig, "Perplexity Depression Score (PDS) Landscape", height=380)


# ---------------------------------------------------------------------------
# Plot 5 – Valley ranking
# ---------------------------------------------------------------------------

def plot_valley_ranking(domains: list[dict]) -> go.Figure:
    """Bar chart of valleys ranked by ValleyScore."""
    if not domains:
        return _empty_figure("Valley Ranking")
    ranked = sorted(domains, key=lambda d: d.get("ValleyScore", 0.0), reverse=True)
    ids = [d.get("ID", f"PV_{i:06d}") for i, d in enumerate(ranked, 1)]
    scores = [d.get("ValleyScore", 0.0) for d in ranked]
    pds_means = [d.get("PDSMean", 0.0) for d in ranked]
    lengths = [d.get("Length", 0) for d in ranked]
    persistence = [d.get("Persistence", 0.0) for d in ranked]

    colors = [
        f"rgba(15,118,110,{0.4 + 0.6 * s / max(scores, default=1)})" for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=ids,
        y=scores,
        marker=dict(color=colors, line=dict(color=_TEAL, width=1)),
        customdata=list(zip(lengths, pds_means, persistence)),
        hovertemplate=(
            "%{x}<br>"
            "ValleyScore: %{y:.4f}<br>"
            "Length: %{customdata[0]} bp<br>"
            "PDSMean: %{customdata[1]:.3f}<br>"
            "Persistence: %{customdata[2]:.2f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Valley Score")
    return _apply_base(fig, "Valley Ranking by Score", height=380)


# ---------------------------------------------------------------------------
# Plot 6 – Algorithm workflow (Sankey)
# ---------------------------------------------------------------------------

def plot_algorithm_workflow() -> go.Figure:
    """REGPLEX v13 algorithm workflow as a Sankey diagram."""
    labels = [
        "DNA Input",
        "Dinucleotide Perplexity\n(17 nt window)",
        "Savitzky–Golay Smoothing\n(21 bp, order 3)",
        "Perplexity Depression Score\n(three-window contrast)",
        "Bounded Kadane Detection\n(100–1000 bp)",
        "Valley Expansion\n(PDS threshold)",
        "Valley Merging\n(gap ≤ 100 bp)",
        "Valley Metrics\n& Biological Filter",
        "ValleyScore Ranking",
        "Optional Motif\nAnnotation",
        "Downloads\n(CSV/Excel/BED/GFF/FASTA/JSON)",
    ]
    colors = [
        _BLUE, _BLUE,
        _AMBER,
        _TEAL,
        _GREEN, _GREEN, _GREEN,
        _CRIMSON,
        _CRIMSON,
        _TEAL,
        _BLUE,
    ]
    n = len(labels)
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels, pad=16, thickness=16,
            line=dict(color=_GRID, width=1),
            color=colors,
        ),
        link=dict(
            source=list(range(n - 1)),
            target=list(range(1, n)),
            value=[5] * (n - 1),
            color=["rgba(30,58,138,0.12)"] * (n - 1),
        ),
    ))
    return _apply_base(fig, "REGPLEX v13 — Algorithm Workflow", height=500)
