from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_BG = "#ffffff"
_GRID = "rgba(148,163,184,0.22)"
_TEXT = "#1f2937"
_MUTED = "#475569"
_BLUE = "#1E3A8A"
_TEAL = "#0F766E"
_GREEN = "#10B981"
_AMBER = "#F59E0B"
_CRIMSON = "#DC2626"

_LAYER_COLORS = {"mono": _BLUE, "di": _TEAL, "tri": _GREEN}
_SCALE_COLORS = [_BLUE, _TEAL, _GREEN, _AMBER, _CRIMSON, "#2563eb", "#1d4ed8", _MUTED]

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


def _valley_bounds(domain: dict) -> tuple[int, int]:
    return int(domain.get("Signal_Start", domain.get("Start", 0))), int(domain.get("Signal_End", domain.get("End", 0)))


def plot_smoothed_perplexity(
    mono: np.ndarray,
    di: np.ndarray,
    tri: np.ndarray,
    smoothed_mono: np.ndarray,
    smoothed_di: np.ndarray,
    smoothed_tri: np.ndarray,
    domains: list[dict],
) -> go.Figure:
    """Plot raw vs Savitzky–Golay smoothed perplexity for each layer."""
    if len(di) == 0 and len(smoothed_di) == 0:
        return _empty_figure("Raw vs Savitzky–Golay Smoothed Perplexity")

    fig = go.Figure()
    raw_pairs = [("Mono", mono, _BLUE), ("Di", di, _TEAL), ("Tri", tri, _GREEN)]
    smooth_pairs = [("Mono SG", smoothed_mono, _BLUE), ("Di SG", smoothed_di, _TEAL), ("Tri SG", smoothed_tri, _GREEN)]

    for (name, arr, color), (sname, sarr, scolor) in zip(raw_pairs, smooth_pairs):
        if len(arr) > 0:
            x = np.arange(len(arr))
            fig.add_trace(go.Scatter(
                x=x, y=arr, mode="lines", name=f"{name} Raw",
                line=dict(color=color, width=1, dash="dot"), opacity=0.45, legendgroup=name,
            ))
        if len(sarr) > 0:
            xs = np.arange(len(sarr))
            fig.add_trace(go.Scatter(
                x=xs, y=sarr, mode="lines", name=f"{sname} Smoothed",
                line=dict(color=scolor, width=2.2), opacity=0.9, legendgroup=name,
            ))

    for d in domains:
        s0, s1 = _valley_bounds(d)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.10)", line_width=0)

    fig.update_layout(xaxis_title="Signal position", yaxis_title="Perplexity")
    return _apply_base(fig, "Raw vs Savitzky–Golay Smoothed Perplexity", height=420)


def plot_perplexity_layers(mono: np.ndarray, di: np.ndarray, tri: np.ndarray) -> go.Figure:
    if len(di) == 0:
        return _empty_figure("Layer Perplexity Profiles")
    x = np.arange(len(di))
    fig = go.Figure()
    for name, arr in (("Mono", mono), ("Di", di), ("Tri", tri)):
        if len(arr) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=x[: len(arr)],
                y=arr,
                mode="lines",
                name=f"{name} Perplexity",
                line=dict(color=_LAYER_COLORS[name.lower()], width=1.8),
                opacity=0.9,
            )
        )
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Perplexity")
    return _apply_base(fig, "Layer Perplexity Profiles", height=390)


def plot_multiscale_landscapes(landscapes: dict[int, np.ndarray], domains: list[dict], layer_name: str) -> go.Figure:
    if not landscapes:
        return _empty_figure(f"{layer_name} Multi-scale Landscapes")
    fig = go.Figure()
    for i, s in enumerate(sorted(landscapes)):
        arr = landscapes[s]
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(arr)),
                y=arr,
                mode="lines",
                name=f"{s} bp",
                line=dict(color=_SCALE_COLORS[i % len(_SCALE_COLORS)], width=1.6),
                opacity=0.85,
            )
        )
    for d in domains:
        s0, s1 = _valley_bounds(d)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.10)", line_width=0)
    fig.update_layout(xaxis_title="Signal position", yaxis_title=f"{layer_name} landscape perplexity")
    return _apply_base(fig, f"{layer_name} Multi-scale Landscapes", height=400)


def plot_layer_consensus(layer_consensus: dict[str, np.ndarray], domains: list[dict]) -> go.Figure:
    if not layer_consensus:
        return _empty_figure("Layer Consensus LPC")
    fig = go.Figure()
    for layer, arr in layer_consensus.items():
        x = np.arange(len(arr))
        pos = np.where(arr > 0, arr, 0)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pos,
                mode="lines",
                name=f"{layer.capitalize()} Consensus",
                line=dict(color=_LAYER_COLORS.get(layer, _MUTED), width=1.8),
            )
        )
    for d in domains:
        s0, s1 = _valley_bounds(d)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.08)", line_width=0)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="Normalized LPC")
    return _apply_base(fig, "Layer Consensus LPC", height=390)


def plot_consensus_lpc(consensus_lpc: np.ndarray, domains: list[dict]) -> go.Figure:
    if len(consensus_lpc) == 0:
        return _empty_figure("ConsensusLPC")
    x = np.arange(len(consensus_lpc))
    pos = np.where(consensus_lpc > 0, consensus_lpc, 0)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, dash="dot"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pos,
            mode="lines",
            name="ConsensusLPC",
            line=dict(color=_GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.15)",
        )
    )
    for d in domains:
        s0, s1 = _valley_bounds(d)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.16)", line_color=_BLUE, opacity=0.30)
    fig.update_layout(xaxis_title="Signal position", yaxis_title="ConsensusLPC")
    return _apply_base(fig, "Final Ensemble ConsensusLPC")


def plot_kadane_domains(
    consensus_lpc: np.ndarray,
    domains: list[dict],
    kadane_core: tuple[int | None, int | None] = (None, None),
    candidates: list[tuple[int, int]] | None = None,
) -> go.Figure:
    if len(consensus_lpc) == 0:
        return _empty_figure("Kadane Refinement and Final Valleys")
    x = np.arange(len(consensus_lpc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=consensus_lpc, mode="lines", name="ConsensusLPC", line=dict(color="rgba(31,41,55,0.35)", width=1.2)))

    # Raw candidates (light background)
    if candidates:
        for i, (cs, ce) in enumerate(candidates):
            fig.add_vrect(
                x0=cs, x1=ce,
                fillcolor="rgba(245,158,11,0.06)",
                line_width=0,
                annotation_text="cand" if i == 0 else None,
                annotation_position="top left" if i == 0 else None,
            )

    # Final domains (accepted)
    for domain in domains:
        s0, s1 = _valley_bounds(domain)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.18)", line_color=_BLUE, line_width=1, opacity=0.7)

    # Global best Kadane core
    ks, ke = kadane_core
    if ks is not None and ke is not None:
        fig.add_vrect(
            x0=ks, x1=ke,
            fillcolor="rgba(245,158,11,0.30)",
            line_color=_AMBER,
            annotation_text="Best Kadane core",
            annotation_position="top left",
        )
    fig.update_layout(xaxis_title="Signal position", yaxis_title="ConsensusLPC")
    return _apply_base(fig, "Kadane Refinement and Final Valleys", height=420)


def plot_scale_support_heatmap(lpc_profiles: dict[str, dict[int, np.ndarray]], domains: list[dict], scales: list[int]) -> go.Figure:
    if not domains or not lpc_profiles:
        return _empty_figure("Scale and Layer Support")
    valley_ids = [d.get("ID", f"PV_{i:06d}") for i, d in enumerate(domains, 1)]
    x_labels = [f"{layer}:{sc}bp" for layer in ("mono", "di", "tri") for sc in scales]
    z: list[list[float]] = []
    for d in domains:
        s0, s1 = _valley_bounds(d)
        row: list[float] = []
        for layer in ("mono", "di", "tri"):
            for sc in scales:
                arr = lpc_profiles.get(layer, {}).get(sc)
                if arr is None or s1 >= len(arr):
                    row.append(float("nan"))
                    continue
                seg = arr[s0: s1 + 1]
                finite_vals = seg[np.isfinite(seg)]
                row.append(float(np.mean(finite_vals)) if finite_vals.size else float("nan"))
        z.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=valley_ids,
            colorscale=[[0.0, "#e2e8f0"], [0.5, _TEAL], [1.0, _BLUE]],
            zmid=0,
            colorbar=dict(title="Mean LPC"),
            hovertemplate="Valley: %{y}<br>Layer/Scale: %{x}<br>Mean LPC: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Layer / Scale", yaxis_title="Valley")
    h = max(320, 60 + 30 * len(domains))
    return _apply_base(fig, "Scale and Layer Support", height=h)


def plot_domain_ranking(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Valley Ranking")
    ranked = sorted(domains, key=lambda d: d.get("ValleyScore", 0.0), reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[d.get("ID", "PV") for d in ranked],
            y=[d.get("ValleyScore", 0.0) for d in ranked],
            marker=dict(color=[d.get("Contrast", 0.0) for d in ranked], colorscale=[[0.0, "#cbd5e1"], [0.5, _TEAL], [1.0, _BLUE]], colorbar=dict(title="Contrast")),
            customdata=np.array([[d.get("Length", 0), d.get("ScaleSupport", "0/0"), d.get("LayerSupport", "0/0")] for d in ranked], dtype=object),
            hovertemplate="%{x}<br>ValleyScore %{y:.4f}<br>Length %{customdata[0]} bp<br>ScaleSupport %{customdata[1]}<br>LayerSupport %{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Valley Score")
    return _apply_base(fig, "Valley Ranking", height=390)


def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    if not domains:
        return _empty_figure("Motif Mapping")
    ranked = sorted(domains, key=lambda d: d.get("MotifCount", 0), reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[d.get("ID", "PV") for d in ranked],
            y=[d.get("MotifCount", 0) for d in ranked],
            marker=dict(color=_TEAL),
            customdata=[d.get("Motifs", "") or "—" for d in ranked],
            hovertemplate="%{x}<br>MotifCount %{y}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Valley ID", yaxis_title="Motif count")
    return _apply_base(fig, "Motif Mapping", height=360)


def plot_vpi_profile(
    consensus_lpc: np.ndarray,
    vpi: np.ndarray,
    domains: list[dict],
    kadane_core: tuple[int | None, int | None] = (None, None),
) -> go.Figure:
    """Valley Persistence Profile — primary interpretation figure for v12.

    Plots ConsensusLPC (left axis) and VPI (right axis, 0–1) on the same
    figure, with final valley extents, per-valley Kadane cores and the global
    best Kadane core highlighted.
    """
    if len(consensus_lpc) == 0:
        return _empty_figure("Valley Persistence Profile (VPI)")

    x = np.arange(len(consensus_lpc))
    fig = go.Figure()

    # ConsensusLPC — positive region only, filled
    pos_lpc = np.where(np.isfinite(consensus_lpc) & (consensus_lpc > 0), consensus_lpc, 0.0)
    fig.add_trace(go.Scatter(
        x=x, y=pos_lpc,
        mode="lines", name="ConsensusLPC",
        line=dict(color=_GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
    ))

    # VPI on secondary axis
    if len(vpi) > 0:
        xv = np.arange(len(vpi))
        fig.add_trace(go.Scatter(
            x=xv, y=vpi.astype(float),
            mode="lines", name="VPI",
            line=dict(color=_AMBER, width=2, dash="dash"),
            yaxis="y2",
            opacity=0.85,
        ))
        # Threshold reference lines (as traces on y2)
        n_pts = len(vpi)
        fig.add_trace(go.Scatter(
            x=[0, n_pts - 1], y=[0.6, 0.6],
            mode="lines", name="VPI ≥ 0.6 (candidate)",
            line=dict(color=_AMBER, dash="dot", width=1),
            yaxis="y2", showlegend=True, opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=[0, n_pts - 1], y=[0.3, 0.3],
            mode="lines", name="VPI ≥ 0.3 (expand)",
            line=dict(color=_CRIMSON, dash="dot", width=1),
            yaxis="y2", showlegend=True, opacity=0.7,
        ))

    # Final valleys
    for domain in domains:
        s0, s1 = _valley_bounds(domain)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(30,58,138,0.16)", line_color=_BLUE, line_width=1, opacity=0.6)
        # Per-valley Kadane core (visualization only)
        ks = domain.get("KadaneCoreStart")
        ke = domain.get("KadaneCoreEnd")
        if ks is not None and ke is not None:
            fig.add_vrect(x0=ks, x1=ke, fillcolor="rgba(245,158,11,0.22)", line_width=0)

    # Global best Kadane core
    ks_g, ke_g = kadane_core
    if ks_g is not None and ke_g is not None:
        fig.add_vrect(
            x0=ks_g, x1=ke_g,
            fillcolor="rgba(245,158,11,0.32)",
            line_color=_AMBER, line_width=1.5,
            annotation_text="Best Kadane core",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Signal position",
        yaxis_title="ConsensusLPC",
        yaxis2=dict(
            title="VPI (0–1)",
            overlaying="y",
            side="right",
            range=[0.0, 1.05],
            showgrid=False,
            tickfont=dict(color=str(_AMBER)),
        ),
    )
    return _apply_base(fig, "Valley Persistence Profile (VPI)", height=430)


def plot_algorithm_workflow() -> go.Figure:
    labels = [
        "DNA",
        "Mono Perplexity",
        "Di Perplexity",
        "Tri Perplexity",
        "Savitzky–Golay Smoothing",
        "Multi-scale Landscapes",
        "Three-window LPC",
        "Layer Consensus",
        "Ensemble ConsensusLPC",
        "Valley Persistence Index (VPI)",
        "Gap-Tolerant Candidates (VPI ≥ 0.6)",
        "VPI Expansion (VPI > 0.3)",
        "Kadane Core (viz only)",
        "ConsensusLPC Persistence Filter",
        "VPI Persistence Filter (≥ 80 %)",
        "Prominence Filter (75th pct)",
        "Score & NMS",
        "Merged Domains",
        "Motif Annotation",
    ]
    colors = [
        _BLUE, _BLUE, _TEAL, _GREEN,
        _AMBER,
        _AMBER, _AMBER, _BLUE, _TEAL,
        _AMBER,
        _GREEN, _GREEN, _AMBER, _CRIMSON, _CRIMSON, _CRIMSON,
        _CRIMSON, _BLUE, _TEAL,
    ]
    n = len(labels)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, pad=14, thickness=14, line=dict(color=_GRID, width=1), color=colors[:n]),
            link=dict(source=list(range(n - 1)), target=list(range(1, n)), value=[4] * (n - 1), color=["rgba(30,58,138,0.15)"] * (n - 1)),
        )
    )
    return _apply_base(fig, "REGPLEX v12 Workflow", height=560)
