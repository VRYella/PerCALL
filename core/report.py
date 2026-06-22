"""
core/report.py
──────────────
Publication-quality PDF report generation for PERCALL results.

Uses fpdf2 (FPDF2) to build a multi-section PDF containing:
  • Cover page with run parameters
  • Sequence statistics table
  • Regulatory regions table
  • Motif summary table
  • Embedded figures (perplexity profile, motif enrichment)

fpdf2 is a pure-Python library — no external system dependencies.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Figure helpers (matplotlib, saved to bytes for embedding)
# ---------------------------------------------------------------------------


def _perplexity_figure_bytes(
    perp: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    regions: List[dict],
    title: str = "",
) -> bytes:
    """Return a PNG of the perplexity profile as raw bytes."""
    positions = np.arange(len(perp))

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [0.55, 0.45]},
    )
    fig.patch.set_facecolor("white")

    ax1, ax2 = axes

    ax1.plot(positions, perp, color="#2196F3", lw=0.9, label="Perplexity")
    ax1.plot(positions, baseline, color="#9e9e9e", lw=1.1, ls="--", label="Baseline")
    ax1.set_ylabel("Perplexity", fontsize=8)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, lw=0.4, alpha=0.5)

    ax2.plot(positions, residual, color="#FF8C00", lw=0.9, label="Residual")
    ax2.axhline(0, color="#9e9e9e", ls="--", lw=0.7)
    ax2.set_ylabel("Residual", fontsize=8)
    ax2.set_xlabel("Position (bp)", fontsize=8)
    ax2.grid(True, lw=0.4, alpha=0.5)

    for r in regions:
        for ax in (ax1, ax2):
            ax.axvspan(r["start"], r["end"], color="#f8514926", lw=0)
        ax1.axvline(r["trough"], color="#f85149", lw=0.7, ls=":")

    if title:
        fig.suptitle(title, fontsize=10, y=1.01)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _motif_enrichment_figure_bytes(
    region_counts: Dict[str, int],
    background_counts: Dict[str, int],
    total_region_bp: int,
    total_seq_bp: int,
    active_motifs: List[str],
    motif_labels: Dict[str, str],
) -> bytes:
    """Return a PNG of the motif enrichment bar chart as raw bytes."""
    from .motifs import MOTIF_COLORS

    region_kb = max(total_region_bp / 1000.0, 1e-9)
    seq_kb = max(total_seq_bp / 1000.0, 1e-9)

    labels = [motif_labels.get(m, m) for m in active_motifs]
    bg_rates = [background_counts.get(m, 0) / seq_kb for m in active_motifs]
    reg_rates = [region_counts.get(m, 0) / region_kb for m in active_motifs]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.bar(x - w / 2, bg_rates, width=w, label="Whole-sequence", color="#4a9eff")
    ax.bar(x + w / 2, reg_rates, width=w, label="Inside regions", color="#f85149")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Hits per kb", fontsize=8)
    ax.set_title("Non-B DNA Motif Enrichment", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------


def generate_pdf(
    params: Dict,
    seq_stats: List[Dict],
    regions_df_records: List[Dict],
    motif_counts: Dict[str, int],
    perp: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    regions: Optional[List[dict]] = None,
    region_counts: Optional[Dict[str, int]] = None,
    background_counts: Optional[Dict[str, int]] = None,
    total_region_bp: int = 0,
    total_seq_bp: int = 0,
    active_motifs: Optional[List[str]] = None,
    motif_labels: Optional[Dict[str, str]] = None,
    seq_label: str = "",
) -> bytes:
    """
    Build and return a PDF report as raw bytes.

    Parameters
    ----------
    params : dict
        Run parameters (window, min_len, etc.).
    seq_stats : list of dict
        One row per input sequence (header, length, gc_pct, n_regions).
    regions_df_records : list of dict
        Regulatory regions table rows.
    motif_counts : dict
        Overall motif counts in the analysed sequence.
    perp / baseline / residual / regions
        Arrays and region list for generating the profile figure.
    region_counts / background_counts / total_* / active_motifs / motif_labels
        Data for the motif enrichment figure.
    seq_label : str
        Label used in the profile figure title.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError(
            "fpdf2 is required for PDF report generation.  "
            "Install it with:  pip install fpdf2"
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ------------------------------------------------------------------
    # Cover / header
    # ------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 100, 200)
    pdf.cell(0, 10, "PERCALL Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0, 6,
        f"PERplexity-based Regulatory Region CALLer  |  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True,
        align="C",
    )
    pdf.ln(6)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    _section_header(pdf, "Run Parameters")
    param_items = [
        ("Window size", params.get("window", 10)),
        ("Min region length (bp)", params.get("min_len", 50)),
        ("Max region length (bp)", params.get("max_len", 300)),
        ("Baseline window (bp)", params.get("baseline_win", 200)),
        ("Number of regions", params.get("top_k", 5)),
        ("Score threshold", params.get("score_cutoff", -0.05)),
        ("Active motifs", ", ".join(active_motifs or [])),
    ]
    _two_col_table(pdf, param_items)
    pdf.ln(4)

    # ------------------------------------------------------------------
    # Sequence statistics
    # ------------------------------------------------------------------
    _section_header(pdf, "Sequence Statistics")
    col_widths = [70, 35, 20, 30]
    headers = ["Header", "Length (bp)", "GC%", "Regions"]
    _table_header(pdf, headers, col_widths)
    for row in seq_stats[:50]:  # cap at 50 rows to prevent excessive PDF page count
        _table_row(
            pdf,
            [
                str(row.get("header", ""))[:40],
                str(row.get("length", "")),
                f"{row.get('gc_pct', 0):.1f}",
                str(row.get("n_regions", 0)),
            ],
            col_widths,
        )
    pdf.ln(4)

    # ------------------------------------------------------------------
    # Regulatory regions
    # ------------------------------------------------------------------
    _section_header(pdf, "Regulatory Regions")
    col_widths2 = [12, 22, 22, 18, 22, 28, 16, 30]
    headers2 = ["Rank", "Start", "End", "Width", "Trough", "Mean Res.", "GC%", "Motifs"]
    _table_header(pdf, headers2, col_widths2)
    for row in regions_df_records[:100]:
        _table_row(
            pdf,
            [
                str(row.get("rank", "")),
                str(row.get("start", "")),
                str(row.get("end", "")),
                str(row.get("width", "")),
                str(row.get("trough", "")),
                str(row.get("mean_residual", "")),
                str(row.get("gc_pct", "")),
                str(row.get("motifs", ""))[:20],
            ],
            col_widths2,
        )
    pdf.ln(4)

    # ------------------------------------------------------------------
    # Motif summary
    # ------------------------------------------------------------------
    _section_header(pdf, "Motif Summary")
    _two_col_table(
        pdf,
        [(motif_labels.get(k, k) if motif_labels else k, str(v))
         for k, v in motif_counts.items() if v > 0],
    )
    pdf.ln(4)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if perp is not None and baseline is not None and residual is not None:
        _section_header(pdf, "Perplexity Profile")
        _embed_figure(
            pdf,
            _perplexity_figure_bytes(
                perp, baseline, residual, regions or [],
                title=f"PERCALL — {seq_label}",
            ),
        )

    if (
        region_counts is not None
        and background_counts is not None
        and active_motifs
        and motif_labels
    ):
        _section_header(pdf, "Motif Enrichment")
        _embed_figure(
            pdf,
            _motif_enrichment_figure_bytes(
                region_counts,
                background_counts,
                total_region_bp,
                total_seq_bp,
                active_motifs,
                motif_labels,
            ),
        )

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# PDF helper utilities
# ---------------------------------------------------------------------------


def _section_header(pdf, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 100, 200)
    pdf.cell(0, 7, text, ln=True)
    pdf.set_draw_color(30, 100, 200)
    pdf.set_line_width(0.3)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)


def _two_col_table(pdf, rows: List[tuple]) -> None:
    pdf.set_font("Helvetica", "", 9)
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(65, 6, str(label), border=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, str(value), ln=True)


def _table_header(pdf, headers: List[str], col_widths: List[int]) -> None:
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 230, 245)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln()


def _table_row(pdf, values: List[str], col_widths: List[int]) -> None:
    pdf.set_font("Helvetica", "", 8)
    for v, w in zip(values, col_widths):
        pdf.cell(w, 5, str(v), border=1)
    pdf.ln()


def _embed_figure(pdf, img_bytes: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(img_bytes)
        tmp_path = tf.name
    try:
        pdf.image(tmp_path, x=pdf.get_x(), y=pdf.get_y(), w=170)
        pdf.ln(85)  # approximate figure height
    finally:
        os.unlink(tmp_path)
