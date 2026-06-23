"""
core/depression.py
──────────────────
Local Depression Framework — the PRIMARY NOVELTY of REGPLEX v3.

THREE-WINDOW MODEL
──────────────────
For every genomic position *d* (treated as the domain centre):

  ┌──────────────────┐  spacer  ┌──────────────┐  spacer  ┌──────────────────┐
  │  Upstream flank  │  ──────  │    Domain    │  ──────  │ Downstream flank │
  │  (flank_win bp)  │          │ (domain_win) │          │  (flank_win bp)  │
  └──────────────────┘          └──────────────┘          └──────────────────┘

DEPRESSION RATIOS
─────────────────
P1_Depression = mean(P1_flanks) / (mean(P1_domain) + ε)
P2_Depression = mean(P2_flanks) / (mean(P2_domain) + ε)

Values > 1 indicate the domain is LOWER complexity than its context —
a locally depressed regulatory architecture.

COMPOSITE DEPRESSION SIGNAL
────────────────────────────
Composite = w1 × P1_Depression + w2 × P2_Depression

REGULATORY DEPRESSION INDEX (RDI)
──────────────────────────────────
For each detected domain [gs, ge]:
  P1_Contrast  = mean(P1_flanks) / (mean(P1_domain) + ε)
  P2_Contrast  = mean(P2_flanks) / (mean(P2_domain) + ε)
  LengthFactor = log(domain_length)
  MotifFactor  = 1 + motif_count
  RDI_raw      = P1_Contrast × P2_Contrast × LengthFactor × MotifFactor
  RDI          = normalise(RDI_raw) → [0, 1]

DOMAIN CLASSES
──────────────
  Class I   : RDI > 0.80
  Class II  : 0.60 – 0.80
  Class III : 0.40 – 0.60
  Class IV  : < 0.40

PERFORMANCE
───────────
All operations use NumPy prefix sums — O(n) complexity, no Python loops
over positions.  Suitable for chromosome-scale analysis.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prefix_stats(arr: np.ndarray):
    """
    Build cumulative-sum and valid-count prefix arrays for an array that may
    contain NaN values.

    Returns
    -------
    cval   : float64 array of length n+1.  cval[j] = sum of arr[0..j-1] (NaN→0).
    cvalid : float64 array of length n+1.  cvalid[j] = count of finite arr[0..j-1].
    """
    valid = np.isfinite(arr).astype(np.float64)
    vals = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    n = len(arr)
    cval = np.empty(n + 1, dtype=np.float64)
    cvalid = np.empty(n + 1, dtype=np.float64)
    cval[0] = 0.0
    cvalid[0] = 0.0
    np.cumsum(vals, out=cval[1:])
    np.cumsum(valid, out=cvalid[1:])
    return cval, cvalid


def _vec_mean(
    cval: np.ndarray,
    cvalid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Vectorised mean of arr[starts[i]:ends[i]] for all *i*, using prefix sums.
    Returns NaN where no valid values exist in the window.
    """
    n_valid = cvalid[ends] - cvalid[starts]
    total = cval[ends] - cval[starts]
    safe = np.where(n_valid > 0, n_valid, 1.0)
    means = (total / safe).astype(np.float32)
    means[n_valid == 0] = np.nan
    return means


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_depression_profile(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    flank_win: int = 100,
    spacer: int = 50,
    domain_win: int = 100,
    p1_weight: float = 0.5,
    p2_weight: float = 0.5,
    eps: float = 1e-6,
) -> dict:
    """
    Compute the genome-wide Local Depression Profile.

    For every position *d*, evaluates whether the central domain is
    significantly LOWER complexity than its flanking context — the hallmark
    of a locally depressed regulatory architecture.

    Parameters
    ----------
    p1 : np.ndarray
        First-order perplexity profile (float32, may contain NaN).
    p2 : np.ndarray
        Second-order perplexity profile (same length as p1, may contain NaN).
    flank_win : int
        Width of each flanking window in positions (default 100).
    spacer : int
        Gap in positions between the domain edge and each flank
        (default 50, user-adjustable 0–200).
    domain_win : int
        Width of the domain window in positions (default 100).
    p1_weight : float
        Weight of P1_Depression in the composite signal (default 0.5).
    p2_weight : float
        Weight of P2_Depression in the composite signal (default 0.5).
    eps : float
        Small constant preventing division by zero (default 1e-6).

    Returns
    -------
    dict with arrays (all length n, float32, NaN at edges / invalid windows):
        p1_dep         – P1 depression ratio  = mean(P1_flanks) / mean(P1_domain).
        p2_dep         – P2 depression ratio  = mean(P2_flanks) / mean(P2_domain).
        composite      – Weighted composite depression.
        p1_domain_mean – Mean P1 in the domain window at each centre.
        p2_domain_mean – Mean P2 in the domain window at each centre.
        p1_flank_mean  – Mean P1 in both flanks at each centre.
        p2_flank_mean  – Mean P2 in both flanks at each centre.

    Notes
    -----
    Minimum sequence length to produce any valid output:
    ``2 × (flank_win + spacer) + domain_win``
    (defaults: 2 × 150 + 100 = 400 positions).
    """
    n = len(p1)
    domain_half = domain_win // 2
    # Distance from domain centre to furthest edge of a flank:
    margin = domain_half + spacer + flank_win

    out = {
        k: np.full(n, np.nan, dtype=np.float32)
        for k in (
            "p1_dep", "p2_dep", "composite",
            "p1_domain_mean", "p2_domain_mean",
            "p1_flank_mean", "p2_flank_mean",
        )
    }

    d_min = margin
    d_max = n - margin   # exclusive upper bound

    if d_min >= d_max:
        return out

    # ── Prefix sums ───────────────────────────────────────────────────────
    p1c, p1v = _prefix_stats(p1)
    p2c, p2v = _prefix_stats(p2)

    d = np.arange(d_min, d_max, dtype=np.int64)

    # ── Window boundary arrays (all vectorised) ──────────────────────────
    dom_start = d - domain_half          # inclusive
    dom_end   = d + domain_half          # exclusive → domain_win elements

    up_end   = d - domain_half - spacer  # exclusive
    up_start = up_end - flank_win        # inclusive → flank_win elements

    dn_start = d + domain_half + spacer  # inclusive
    dn_end   = dn_start + flank_win      # exclusive → flank_win elements

    # ── P1 means ─────────────────────────────────────────────────────────
    p1_dom  = _vec_mean(p1c, p1v, dom_start, dom_end)
    p1_up   = _vec_mean(p1c, p1v, up_start,  up_end)
    p1_dn   = _vec_mean(p1c, p1v, dn_start,  dn_end)

    both_p1  = np.isfinite(p1_up) & np.isfinite(p1_dn)
    p1_flank = np.where(both_p1,
                         (p1_up + p1_dn) / 2,
                         np.where(np.isfinite(p1_up), p1_up, p1_dn)
                         ).astype(np.float32)

    # ── P2 means ─────────────────────────────────────────────────────────
    p2_dom  = _vec_mean(p2c, p2v, dom_start, dom_end)
    p2_up   = _vec_mean(p2c, p2v, up_start,  up_end)
    p2_dn   = _vec_mean(p2c, p2v, dn_start,  dn_end)

    both_p2  = np.isfinite(p2_up) & np.isfinite(p2_dn)
    p2_flank = np.where(both_p2,
                         (p2_up + p2_dn) / 2,
                         np.where(np.isfinite(p2_up), p2_up, p2_dn)
                         ).astype(np.float32)

    # ── Depression ratios ─────────────────────────────────────────────────
    # Values > 1.0 → domain is lower complexity than flanks (desired signature)
    p1_dep = np.where(
        np.isfinite(p1_flank) & np.isfinite(p1_dom),
        p1_flank / (p1_dom + eps),
        np.nan,
    ).astype(np.float32)

    p2_dep = np.where(
        np.isfinite(p2_flank) & np.isfinite(p2_dom),
        p2_flank / (p2_dom + eps),
        np.nan,
    ).astype(np.float32)

    # ── Composite depression ──────────────────────────────────────────────
    both_dep = np.isfinite(p1_dep) & np.isfinite(p2_dep)
    p1_only  = np.isfinite(p1_dep) & ~np.isfinite(p2_dep)
    p2_only  = ~np.isfinite(p1_dep) & np.isfinite(p2_dep)

    comp = np.full(len(d), np.nan, dtype=np.float32)
    comp[both_dep] = (
        p1_weight * p1_dep[both_dep] + p2_weight * p2_dep[both_dep]
    )
    comp[p1_only] = p1_dep[p1_only]
    comp[p2_only] = p2_dep[p2_only]

    # ── Write to output (slice assignment avoids fancy-index copy) ────────
    sl = slice(d_min, d_max)
    out["p1_dep"][sl]          = p1_dep
    out["p2_dep"][sl]          = p2_dep
    out["composite"][sl]       = comp
    out["p1_domain_mean"][sl]  = p1_dom
    out["p2_domain_mean"][sl]  = p2_dom
    out["p1_flank_mean"][sl]   = p1_flank
    out["p2_flank_mean"][sl]   = p2_flank

    return out


def compute_rdi(
    regions: list,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    spacer: int = 50,
    flank_win: int = 100,
    eps: float = 1e-6,
) -> list:
    """
    Compute the Regulatory Depression Index (RDI) for each detected domain
    and add the results to the region dicts in-place.

    RDI formula
    -----------
    For each domain [gs, ge]:
      P1_Contrast  = mean(P1_flanks) / (mean(P1_domain) + ε)
      P2_Contrast  = mean(P2_flanks) / (mean(P2_domain) + ε)
      LengthFactor = log(domain_length)
      MotifFactor  = 1 + motif_count
      RDI_raw      = P1_Contrast × P2_Contrast × LengthFactor × MotifFactor

    RDI is normalised to [0, 1] across all regions in the call.

    Parameters
    ----------
    regions : list of dict
        Region dicts produced by ``find_regions``.
    p1, p2 : np.ndarray
        Perplexity profiles for the same sequence.
    spacer : int
        Spacer gap (same value as used in the depression profile, default 50).
    flank_win : int
        Flank window width (default 100).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Same list of dicts, with the following keys added to each entry:
        rdi, rdi_class, mean_p1, mean_p2,
        mean_p1_flanks, mean_p2_flanks,
        p1_contrast, p2_contrast, motif_count
    """
    n = len(p1)
    p1c, p1v = _prefix_stats(p1)
    p2c, p2v = _prefix_stats(p2)

    def _wm(cv, cvalid, a, b):
        """Scalar mean of arr[a:b]; returns NaN if empty or all-NaN."""
        a, b = max(int(a), 0), min(int(b), n)
        if b <= a:
            return float("nan")
        nv = cvalid[b] - cvalid[a]
        if nv == 0.0:
            return float("nan")
        return float((cv[b] - cv[a]) / nv)

    def _favg(u, d):
        """Average of two potentially-NaN flank means."""
        fu, fd = np.isfinite(u), np.isfinite(d)
        if fu and fd:
            return (u + d) / 2.0
        if fu:
            return float(u)
        if fd:
            return float(d)
        return float("nan")

    raw_rdi: list[float] = []

    for reg in regions:
        gs, ge = reg["start"], reg["end"]

        # Domain means
        mp1_dom = _wm(p1c, p1v, gs, ge + 1)
        mp2_dom = _wm(p2c, p2v, gs, ge + 1)

        # Flank window positions
        up_end   = gs - spacer
        up_start = up_end - flank_win
        dn_start = ge + 1 + spacer
        dn_end   = dn_start + flank_win

        # Flank means
        mp1_fl = _favg(
            _wm(p1c, p1v, up_start, up_end),
            _wm(p1c, p1v, dn_start, dn_end),
        )
        mp2_fl = _favg(
            _wm(p2c, p2v, up_start, up_end),
            _wm(p2c, p2v, dn_start, dn_end),
        )

        # Contrasts (ratio; fall back to 1.0 when data unavailable)
        p1c_val = (mp1_fl / (mp1_dom + eps)
                   if np.isfinite(mp1_fl) and np.isfinite(mp1_dom) else 1.0)
        p2c_val = (mp2_fl / (mp2_dom + eps)
                   if np.isfinite(mp2_fl) and np.isfinite(mp2_dom) else 1.0)

        motif_count = len(
            [m for m in reg.get("motifs", "").split(";") if m.strip()]
        )
        length_factor = float(np.log(max(reg["width"], 1)))
        motif_factor  = 1.0 + motif_count

        raw_rdi.append(p1c_val * p2c_val * length_factor * motif_factor)

        reg["mean_p1"]        = round(mp1_dom, 4) if np.isfinite(mp1_dom) else None
        reg["mean_p2"]        = round(mp2_dom, 4) if np.isfinite(mp2_dom) else None
        reg["mean_p1_flanks"] = round(mp1_fl, 4)  if np.isfinite(mp1_fl)  else None
        reg["mean_p2_flanks"] = round(mp2_fl, 4)  if np.isfinite(mp2_fl)  else None
        reg["p1_contrast"]    = round(p1c_val, 4)
        reg["p2_contrast"]    = round(p2c_val, 4)
        reg["motif_count"]    = motif_count

    # ── Normalise RDI to [0, 1] across all regions ────────────────────────
    if raw_rdi:
        mn   = min(raw_rdi)
        mx   = max(raw_rdi)
        span = max(mx - mn, 1e-9)
    else:
        mn = mx = span = 1.0

    for i, reg in enumerate(regions):
        rdi = float((raw_rdi[i] - mn) / span) if raw_rdi else 0.0
        reg["rdi"] = round(rdi, 4)
        if rdi > 0.80:
            reg["rdi_class"] = "I"
        elif rdi >= 0.60:
            reg["rdi_class"] = "II"
        elif rdi >= 0.40:
            reg["rdi_class"] = "III"
        else:
            reg["rdi_class"] = "IV"

    return regions
