from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_p1_profile(p1: np.ndarray) -> go.Figure:
    x = np.arange(len(p1))
    fig = go.Figure(go.Scatter(x=x, y=p1, mode="lines", name="P1"))
    fig.update_layout(title="Figure 1 · Perplexity Profile", xaxis_title="Position", yaxis_title="P1")
    return fig


def plot_pdi_profile(pdi: np.ndarray, domains: list[dict]) -> go.Figure:
    x = np.arange(len(pdi))
    fig = go.Figure(go.Scatter(x=x, y=pdi, mode="lines", name="PDI"))
    for d in domains:
        fig.add_vrect(x0=d["Start"], x1=d["End"], fillcolor="rgba(37,99,235,0.15)", line_width=0)
    fig.update_layout(title="Figure 2 · PDI Profile", xaxis_title="Position", yaxis_title="PDI")
    return fig


def plot_domain_map(seq_length: int, domains: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=seq_length, y0=0.45, y1=0.55, fillcolor="lightgray", line_width=0)
    for d in domains:
        fig.add_shape(type="rect", x0=d["Start"], x1=d["End"], y0=0.3, y1=0.7, fillcolor="royalblue", line_width=0)
    fig.update_layout(title="Figure 3 · Domain Map", xaxis_title="Coordinate", yaxis=dict(visible=False), height=220)
    return fig


def plot_domain_ranking(domains: list[dict]) -> go.Figure:
    ranks = [d["Domain_ID"] for d in domains]
    rcs = [d["RCS"] for d in domains]
    fig = go.Figure(go.Bar(x=ranks, y=rcs, name="RCS"))
    fig.update_layout(title="Figure 4 · Domain Ranking", xaxis_title="Domain", yaxis_title="RCS")
    return fig


def plot_domain_statistics(domains: list[dict]) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Length", "Mean PDI", "Variance P1", "RCS"])
    fig.add_trace(go.Histogram(x=[d["Length"] for d in domains], name="Length"), row=1, col=1)
    fig.add_trace(go.Histogram(x=[d["Mean_PDI"] for d in domains], name="PDI"), row=1, col=2)
    fig.add_trace(go.Histogram(x=[d["Variance_P1"] for d in domains], name="Var"), row=2, col=1)
    fig.add_trace(go.Histogram(x=[d["RCS"] for d in domains], name="RCS"), row=2, col=2)
    fig.update_layout(title="Figure 5 · Domain Statistics", showlegend=False)
    return fig


def plot_motif_architecture(domains: list[dict]) -> go.Figure:
    labels = [d["Domain_ID"] for d in domains]
    counts = [d.get("Motif_Count", 0) for d in domains]
    fig = go.Figure(go.Bar(x=labels, y=counts, name="Motifs"))
    fig.update_layout(title="Figure 6 · Motif Architecture", xaxis_title="Domain", yaxis_title="Motif Count")
    return fig


def plot_algorithm_illustration() -> go.Figure:
    labels = ["DNA", "P1", "PDI", "Kadane", "Domains", "Motifs"]
    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels),
            link=dict(source=[0, 1, 2, 3, 4], target=[1, 2, 3, 4, 5], value=[1, 1, 1, 1, 1]),
        )
    )
    fig.update_layout(title="Figure 7 · Algorithm Illustration")
    return fig
