#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 PERPLEXITY-KADANE  —  Optimal Low-Perplexity Region Detection in DNA
═══════════════════════════════════════════════════════════════════════════════

THE IDEA
  DUST, SEG, and RepeatMasker call low-complexity regions with a fixed entropy
  threshold applied window-by-window. Thresholding answers "is this single
  window low-complexity?" — it does not answer "where does the optimal
  contiguous low-complexity REGION begin and end?", and it is brittle to
  baseline GC drift along a sequence.

  This tool reframes region-calling as an exact OPTIMIZATION problem: find
  the contiguous stretch of the perplexity-residual signal with the MINIMUM
  MEAN value, subject only to a user-set minimum length. This is the
  "bounded-length minimum-mean subarray" problem, solved here in O(n) per
  scan with a monotonic-deque algorithm (the natural generalisation of
  Kadane's maximum-subarray trick to the length-constrained, mean-based
  case). The result is provably optimal for the stated objective — not a
  heuristic, not a threshold sweep.

  Critically: minimum-SUM subarrays (plain Kadane) are NOT well-posed here,
  because once normalised against a local baseline, residuals hover near
  zero almost everywhere — so an unbounded minimum-sum search degenerately
  swallows the entire flanking sequence (zero-valued padding never increases
  a sum). Minimum-MEAN with both a floor (MIN_LEN, user-set, default 50 bp)
  and a ceiling (MAX_LEN) is the correct, well-posed formulation, and is the
  central methodological contribution of this tool.

  The user controls exactly ONE biologically meaningful parameter: MIN_LEN.
  Everything else — depth, exact width up to MAX_LEN, number of regions — is
  decided by the optimisation, not by a tunable threshold.

  Each detected region is optionally screened for non-B DNA motifs (G4,
  i-motif, Z-DNA, STRs, direct repeats, polyA/T, triplex-forming purine
  runs), connecting the information-theoretic signal to known destabilising
  / structure-forming DNA elements — testing the hypothesis that regions of
  minimal sequence perplexity preferentially host non-canonical structure.

INPUT   multi-FASTA file
OUTPUT  <prefix>_regions.csv       — PRIMARY: one row per detected region
        <prefix>_summary.csv      — one row per sequence
        <prefix>_profile.png      — example sequence trace with regions shaded
        <prefix>_motif_summary.png — motif rate inside regions vs background
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, argparse, textwrap, gc, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from numpy.lib.stride_tricks import sliding_window_view

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS — MIN_LEN is the one parameter meant to be routinely changed
# ══════════════════════════════════════════════════════════════════════════════

WINDOW   = 10     # k-mer size for perplexity (dinucleotide-based)
MIN_LEN  = 50     # ← THE ONE FREE PARAMETER: minimum region length (bp)
MAX_LEN  = 300     # ceiling on region width (mathematically required — see header)

BASELINE_WIN = 200    # rolling baseline window (auto-adapts to local GC drift)
TOP_K        = 5      # max regions reported per sequence (paper's own analysis
                       # focuses on the single dominant trough per promoter;
                       # 5 captures secondary regions without excessive scanning)
SCORE_CUTOFF = -0.05  # region mean residual must be below this to be reported
MIN_SEQ_LEN  = 100    # sequences shorter than this are skipped

# Core-promoter restriction: Kadane is run only within [TSS+CORE_WINDOW[0],
# TSS+CORE_WINDOW[1]], not the full flanking sequence. This mirrors the
# paper's own comparative analysis (promoter defined as [-100,-1] relative
# to TSS) and is necessary because an unrestricted scan across several kb
# of flanking sequence will sometimes find a deeper dip in a distal repeat
# element than in the core promoter itself — a real, separate signal, but
# not the one this tool is built to characterise. Set to None to disable
# restriction and scan the full sequence (legacy/exploratory mode).
CORE_WINDOW  = (-200, 50)   # (start, end) bp relative to TSS/TLS at sequence centre

REPORT_MOTIFS = True   # non-B DNA motif annotation inside each region
TOP_N_SEQUENCES = 20   # how many top-ranked sequences to print/report

FIG_DPI = 150

# ══════════════════════════════════════════════════════════════════════════════
CENTER = (WINDOW - 1) // 2

MAP = np.full(256, 4, dtype=np.uint8)
for _b, _i in zip("ACGT", range(4)):
    MAP[ord(_b)] = _i

MOTIFS = {
    'G4'           : re.compile(r'G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}'),
    'iMotif'       : re.compile(r'C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}'),
    # Z-DNA: regex approximation of alternating purine-pyrimidine dinucleotide
    # repeats, the classical Z-DNA-forming sequence grammar (Rich et al. 1984).
    # This is a SIMPLIFICATION: it cannot replicate the weighted 10-mer
    # dinucleotide propensity scoring + Kadane scan used in Non-B DNA Finder
    # (Yella, Gummadi & Kumar, 2026), which excludes destabilizing TA steps
    # via a filtered table of 107 high-confidence 10-mers (score >= 50). The
    # four alternating-step patterns below (CG/GC/CA/TG repeats) are the
    # dinucleotide steps with the highest Z-DNA-forming propensity in that
    # scoring table; TA steps are deliberately excluded here for consistency
    # with that filtering rationale, even though regex cannot apply a
    # continuous score. For publication-grade Z-DNA calls, the scoring-table
    # method should be used instead — see Non-B DNA Finder.
    'ZDNA'         : re.compile(r'(?:CG){4,}|(?:GC){4,}|(?:CA){4,}|(?:TG){4,}'),
    # eGZ-DNA: left-handed Z-DNA conformation formed within expanded CGG/GGC
    # trinucleotide repeats (Fakharzadeh et al. 2022); n>=4 uninterrupted
    # units per Non-B DNA Finder's detection criterion.
    'eGZDNA'       : re.compile(r'(?:CGG){4,}|(?:GGC){4,}'),
    'STR'          : re.compile(r'([ACGT]{1,6})\1{3,}'),
    'PolyA'        : re.compile(r'A{7,}'),
    'PolyT'        : re.compile(r'T{7,}'),
    'DirectRepeat' : re.compile(r'([ACGT]{4,10})[ACGT]{0,10}\1'),
    'Triplex'      : re.compile(r'[AG]{10,}'),
}


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def parse_fasta(filepath):
    header, parts = None, []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(parts).upper()
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, ''.join(parts).upper()


# ══════════════════════════════════════════════════════════════════════════════
# Perplexity
# ══════════════════════════════════════════════════════════════════════════════

def compute_perplexity(seq):
    """Sliding WINDOW-mer perplexity via dinucleotide frequencies. NaN where N present."""
    x     = MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = (x == 4)
    x_c   = np.where(has_n, 0, x)

    w  = sliding_window_view(x_c,   WINDOW)
    wn = sliding_window_view(has_n, WINDOW)
    a, b_ = w[:, :-1], w[:, 1:]
    din = (a * 4 + b_).astype(np.int16)

    n = len(w)
    c = np.zeros((n, 16), dtype=np.int16)
    np.add.at(c, (np.arange(n)[:, None], din), 1)

    p    = np.where(c == 0, 1.0, c / (WINDOW - 1)).astype(np.float32)
    H    = -np.sum(p * np.log2(p), axis=1)
    perp = (2.0 ** H).astype(np.float32)
    perp[wn.any(axis=1)] = np.nan
    return perp


def local_residual(perp):
    """
    Residual against local rolling baseline — adapts to GC drift automatically.
    min_periods=BASELINE_WIN (not 1) is required: with min_periods=1, the
    rolling mean near sequence edges is computed from very few points and
    becomes noisy, producing spurious extreme residuals at the boundaries
    that are an artifact of baseline estimation, not biology. Edge positions
    where a full-width baseline cannot be computed are masked to NaN.
    """
    bl = pd.Series(perp).rolling(BASELINE_WIN, center=True,
                                  min_periods=BASELINE_WIN).mean().values
    return perp - bl


# ══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM
# Bounded-length minimum-mean contiguous subarray (Kadane-family, O(n) per scan)
# ══════════════════════════════════════════════════════════════════════════════

def min_mean_subarray_bounded(arr, min_len, max_len):
    """
    Returns (start, end_inclusive, mean) of the contiguous run with the
    smallest MEAN value, subject to min_len <= length <= max_len.

    Solved in O(n) with a monotonic deque over the prefix-sum array — the
    standard, provably-optimal technique for the length-bounded minimum
    average subarray problem. NaNs are treated as hard breaks by the caller
    (segments are processed independently — see find_regions).
    """
    n = len(arr)
    if n < min_len:
        return None, None, np.inf

    cums = np.empty(n + 1, dtype=np.float64)
    cums[0] = 0.0
    cums[1:] = np.cumsum(arr)

    best_mean = np.inf
    best_i = best_j = -1
    dq = deque()

    for j in range(n):
        i_min = j - max_len + 1
        i_max = j - min_len + 1
        if i_max < 0:
            continue
        while dq and cums[dq[-1]] >= cums[i_max]:
            dq.pop()
        dq.append(i_max)
        while dq and dq[0] < i_min:
            dq.popleft()
        if dq:
            i = dq[0]
            length = j - i + 1
            mean_val = (cums[j + 1] - cums[i]) / length
            if mean_val < best_mean:
                best_mean = mean_val
                best_i, best_j = i, j

    return best_i, best_j, best_mean


def find_regions(residual, seq_slice):
    """
    Iteratively apply the bounded minimum-mean algorithm to find up to TOP_K
    non-overlapping low-perplexity regions. NaN runs (from N-containing
    windows) are processed as independent segments so a single N cannot
    bridge two otherwise-unrelated regions.
    """
    arr = residual.copy().astype(np.float64)
    n   = len(arr)
    regions = []

    # mark NaN as +inf sentinel-friendly: process per finite segment
    nan_mask = np.isnan(arr)

    for rank in range(1, TOP_K + 1):
        # find segments of finite values not yet masked to exactly 0-with-flag
        # (we use a parallel "active" mask so masked-out regions aren't re-picked)
        pass

        best_overall = (np.inf, None, None)
        i = 0
        while i < n:
            if nan_mask[i]:
                i += 1
                continue
            j = i
            while j < n and not nan_mask[j]:
                j += 1
            seg = arr[i:j]
            if len(seg) >= MIN_LEN:
                s, e, m = min_mean_subarray_bounded(seg, MIN_LEN, MAX_LEN)
                if s is not None and m < best_overall[0]:
                    best_overall = (m, i + s, i + e)
            i = j

        m, gs, ge = best_overall
        if gs is None or m >= SCORE_CUTOFF:
            break

        width    = ge - gs + 1
        trough_i = gs + int(np.nanargmin(arr[gs:ge+1]))
        reg_seq  = seq_slice[gs : ge + WINDOW]

        row = {
            'rank'        : rank,
            'start_pos'   : int(gs),
            'end_pos'     : int(ge),
            'trough_pos'  : int(trough_i),
            'width_bp'    : width,
            'mean_residual': round(float(m), 5),
            'gc_pct'      : round(100 * (reg_seq.count('G') + reg_seq.count('C'))
                                   / max(len(reg_seq), 1), 2),
        }

        if REPORT_MOTIFS:
            found = [name for name, pat in MOTIFS.items() if pat.search(reg_seq)]
            row['nonB_motifs']   = ';'.join(found) if found else ''
            row['n_motif_types'] = len(found)

        regions.append(row)
        arr[gs:ge+1] = np.nan       # mask: exclude from future scans
        nan_mask[gs:ge+1] = True

    return regions


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def process_fasta(filepath, prefix, start_idx=0, end_idx=None, checkpoint_every=1000):
    species = os.path.basename(filepath).split('_cleaned')[0]
    print(f"\n{'='*72}")
    print(f"  PERPLEXITY-KADANE  |  Low-perplexity region detector")
    print(f"  File      : {os.path.basename(filepath)}")
    print(f"  MIN_LEN   : {MIN_LEN} bp   (the only user-set parameter)")
    print(f"  MAX_LEN   : {MAX_LEN} bp   (ceiling; mathematically required)")
    print(f"  Motifs    : {'ON' if REPORT_MOTIFS else 'OFF'}")
    print(f"{'='*72}")

    records, skipped = [], 0
    for header, seq in parse_fasta(filepath):
        if len(seq) < MIN_SEQ_LEN + WINDOW:
            skipped += 1; continue
        records.append((header, seq))

    print(f"\n  Loaded {len(records)+skipped} | Skipped short: {skipped} | Valid: {len(records)}")
    if not records:
        print("  ERROR: no valid sequences."); return

    if end_idx is None:
        end_idx = len(records)
    records_slice = records[start_idx:end_idx]
    print(f"  Processing slice [{start_idx}:{end_idx}]  ({len(records_slice)} sequences)")

    region_rows, summary_rows = [], []
    example_perp = example_regions = example_header = None

    for local_idx, (header, seq) in enumerate(records_slice):
        idx = start_idx + local_idx
        perp = compute_perplexity(seq)
        if np.all(np.isnan(perp)):
            continue
        res = local_residual(perp)

        if CORE_WINDOW is not None:
            # Sequence is assumed centred on TSS/TLS at position len(seq)//2
            # (standard EPD-style convention: e.g. -3000..+3000 -> centre = TSS).
            # CORE_WINDOW = (start_rel, end_rel) gives bp offsets from that centre.
            tss_raw_idx = len(seq) // 2 - CENTER   # index into perp/res arrays
            w_start = max(0, tss_raw_idx + CORE_WINDOW[0])
            w_end   = min(len(res), tss_raw_idx + CORE_WINDOW[1])
            res_window = res[w_start:w_end]
            seq_window = seq[w_start : w_end + WINDOW]
            regions = find_regions(res_window, seq_window)
            for r in regions:
                # shift coordinates back to whole-sequence frame, then to TSS-relative
                r['start_pos']  = r['start_pos']  + w_start - tss_raw_idx
                r['end_pos']    = r['end_pos']    + w_start - tss_raw_idx
                r['trough_pos'] = r['trough_pos'] + w_start - tss_raw_idx
        else:
            regions = find_regions(res, seq)

        if example_perp is None and len(regions) > 0:
            example_perp, example_regions, example_header = perp, regions, header

        for r in regions:
            r.update({'seq_idx': idx, 'header': header[:80]})
            region_rows.append(r)

        summary_rows.append({
            'seq_idx'   : idx,
            'header'    : header[:80],
            'seq_length': len(seq),
            'mean_perp' : round(float(np.nanmean(perp)), 4),
            'sd_perp'   : round(float(np.nanstd(perp)),  4),
            'min_perp'  : round(float(np.nanmin(perp)),  4),
            'n_regions' : len(regions),
            'total_region_bp'   : sum(r['width_bp'] for r in regions),
            'pct_seq_in_regions': round(100 * sum(r['width_bp'] for r in regions)
                                         / len(seq), 2),
            'best_mean_residual': round(regions[0]['mean_residual'], 5) if regions else np.nan,
        })

        if (local_idx + 1) % checkpoint_every == 0 or local_idx == len(records_slice) - 1:
            print(f"  [{idx+1:>5}/{len(records)}]  regions so far: {len(region_rows)}")
            # checkpoint to disk
            pd.DataFrame(region_rows).to_csv(f"{prefix}_regions_chunk_{start_idx}_{end_idx}.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(f"{prefix}_summary_chunk_{start_idx}_{end_idx}.csv", index=False)

    df_reg  = pd.DataFrame(region_rows)
    df_summ = pd.DataFrame(summary_rows)

    core = ['seq_idx','header','rank','start_pos','end_pos','trough_pos',
            'width_bp','mean_residual','gc_pct']
    opt  = [c for c in ['nonB_motifs','n_motif_types'] if c in df_reg.columns]
    if not df_reg.empty:
        df_reg = df_reg[core + opt]

    df_reg.to_csv( f"{prefix}_regions_chunk_{start_idx}_{end_idx}.csv",  index=False)
    df_summ.to_csv(f"{prefix}_summary_chunk_{start_idx}_{end_idx}.csv",  index=False)

    print(f"\n  Chunk [{start_idx}:{end_idx}] CSVs written:")
    print(f"    {prefix}_regions_chunk_{start_idx}_{end_idx}.csv   ({len(df_reg):,} regions)")
    print(f"    {prefix}_summary_chunk_{start_idx}_{end_idx}.csv   ({len(df_summ):,} sequences)")

    if example_perp is not None:
        _plot_example(example_perp, example_regions, prefix, species, example_header)

    print(f"\n  Chunk done.\n")
    return df_reg, df_summ


def _report_top_sequences(df_summ, prefix, top_n=20):
    """
    Rank sequences by number of detected low-perplexity regions (n_regions),
    breaking ties by total bp covered by regions. Prints to console and
    writes <prefix>_top_sequences.csv.
    """
    if df_summ.empty:
        return
    ranked = df_summ.sort_values(
        ['n_regions', 'total_region_bp'], ascending=[False, False]
    ).reset_index(drop=True)
    ranked.insert(0, 'overall_rank', ranked.index + 1)

    top = ranked.head(top_n)
    out_path = f"{prefix}_top_sequences.csv"
    ranked.to_csv(out_path, index=False)

    print(f"\n  Top {min(top_n, len(ranked))} sequences by region count "
          f"(full ranking → {out_path}):")
    print(f"  {'rank':>4}  {'n_regions':>9}  {'total_bp':>8}  {'%seq':>6}  header")
    for _, row in top.iterrows():
        print(f"  {row['overall_rank']:>4}  {row['n_regions']:>9}  "
              f"{row['total_region_bp']:>8}  {row['pct_seq_in_regions']:>5.1f}%  "
              f"{row['header']}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

P = {'bg':'#0d1117','panel':'#161b22','line':'#58a6ff','fill':'#1f3a5f',
     'dip':'#f85149','dipf':'#3d1a1a','text':'#c9d1d9','grid':'#21262d',
     'gold':'#e3b341','green':'#3fb950'}

def _ax(ax):
    ax.set_facecolor(P['panel'])
    ax.tick_params(colors=P['text'], labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(P['grid'])
    ax.xaxis.label.set_color(P['text']); ax.yaxis.label.set_color(P['text'])
    ax.title.set_color(P['text'])
    ax.grid(True, color=P['grid'], lw=0.5, ls='--', alpha=0.6)


def _plot_example(perp, regions, prefix, species, header):
    positions = np.arange(CENTER, CENTER + len(perp))
    fig, ax = plt.subplots(figsize=(13, 4), facecolor=P['bg'])
    _ax(ax)
    ax.plot(positions, perp, color=P['line'], lw=1.0)
    for r in regions:
        ax.axvspan(r['start_pos'] + CENTER, r['end_pos'] + CENTER,
                  color=P['dipf'], alpha=0.5)
        ax.axvline(r['trough_pos'] + CENTER, color=P['gold'], lw=0.8, ls=':', alpha=0.8)
    ax.set_xlabel('Sequence position (bp)', fontsize=9)
    ax.set_ylabel('Perplexity', fontsize=9)
    ax.set_title(f'{species} — example sequence ({header[:50]})\n'
                 f'Detected low-perplexity regions (MIN_LEN={MIN_LEN} bp, MAX_LEN={MAX_LEN} bp)',
                 fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig(f"{prefix}_profile.png", dpi=FIG_DPI, facecolor=P['bg'])
    plt.close(); gc.collect()
    print(f"    {prefix}_profile.png")


def _plot_motif_enrichment(df_reg, records, prefix, species):
    motif_cols = list(MOTIFS.keys())
    in_region_counts = {m: 0 for m in motif_cols}
    for motifs_str in df_reg['nonB_motifs'].dropna():
        for m in motifs_str.split(';'):
            if m in in_region_counts:
                in_region_counts[m] += 1

    total_region_bp = df_reg['width_bp'].sum()
    total_seq_bp    = sum(len(s) for _, s in records)

    bg_counts = {m: 0 for m in motif_cols}
    for _, seq in records:
        for name, pat in MOTIFS.items():
            bg_counts[name] += len(pat.findall(seq))

    in_rate = {m: in_region_counts[m] / max(total_region_bp/1000, 1e-9) for m in motif_cols}
    bg_rate = {m: bg_counts[m]        / max(total_seq_bp/1000,    1e-9) for m in motif_cols}

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=P['bg'])
    _ax(ax)
    x = np.arange(len(motif_cols))
    w = 0.35
    ax.bar(x - w/2, [bg_rate[m] for m in motif_cols], width=w,
           color=P['fill'], label='Whole-sequence rate')
    ax.bar(x + w/2, [in_rate[m] for m in motif_cols], width=w,
           color=P['dip'], label='Inside low-perplexity regions')
    ax.set_xticks(x); ax.set_xticklabels(motif_cols, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Hits per kb', fontsize=9)
    ax.set_title(f'{species} — non-B DNA motif rate: regions vs background',
                 fontsize=10, pad=8)
    ax.legend(fontsize=8, facecolor=P['panel'], labelcolor=P['text'], framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f"{prefix}_motif_summary.png", dpi=FIG_DPI, facecolor=P['bg'])
    plt.close(); gc.collect()
    print(f"    {prefix}_motif_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point (notebook-safe)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        get_ipython(); in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        FASTA_FILE = "h.sapiens_noncoding_cleaned"
        PREFIX     = None
        fasta  = FASTA_FILE
        prefix = PREFIX or os.path.basename(fasta).split('_cleaned')[0]
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent("""\
                PERPLEXITY-KADANE — optimal low-perplexity region detector
                User sets only MIN_LEN (edit at top of script, default 50 bp).
            """))
        parser.add_argument('fasta')
        parser.add_argument('--prefix', default=None)
        parser.add_argument('--min_len', type=int, default=None, help='Override MIN_LEN (bp)')
        args = parser.parse_args()
        fasta  = args.fasta
        prefix = args.prefix or os.path.basename(fasta).split('_cleaned')[0]
        if args.min_len is not None:
            global MIN_LEN
            MIN_LEN = args.min_len

    if not os.path.isfile(fasta):
        raise FileNotFoundError(f"File not found: {fasta}")

    process_fasta(fasta, prefix)


if __name__ == '__main__':
    main()
