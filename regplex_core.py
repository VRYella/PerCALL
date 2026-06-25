from __future__ import annotations

import argparse
import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

PERPLEXITY_WINDOW = 10
SECOND_ORDER_WINDOW = 100
CANDIDATE_WINDOW = 100
UPSTREAM_WINDOW = 100
DOWNSTREAM_WINDOW = 100
SPACER = 50
ADAPTIVE_MIN_WINDOW = 50
ADAPTIVE_MAX_WINDOW = 300
MIN_DOMAIN = 50
MAX_DOMAIN = 1000
EPSILON = 1e-9

_IUPAC_DNA = set("ACGTN")
_MAP = np.full(256, 4, dtype=np.uint8)
for base, idx in zip("ACGT", range(4)):
    _MAP[ord(base)] = idx


@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    p1: np.ndarray
    p2: np.ndarray
    pvs: np.ndarray
    upstream_mean: np.ndarray
    candidate_mean: np.ndarray
    downstream_mean: np.ndarray
    upstream_difference: np.ndarray
    downstream_difference: np.ndarray
    candidate_window: np.ndarray
    domains: list[dict]
    params: dict


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = "".join(chunks).upper().replace("U", "T")
                seq = "".join(c for c in seq if c in _IUPAC_DNA)
                if seq:
                    records.append((header, seq))
            header = line[1:].strip() or "sequence"
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        seq = "".join(chunks).upper().replace("U", "T")
        seq = "".join(c for c in seq if c in _IUPAC_DNA)
        if seq:
            records.append((header, seq))
    return records


def compute_p1(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    if len(seq) < window:
        return np.array([], dtype=np.float32)
    arr = _MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = arr == 4
    masked = np.where(has_n, 0, arr)
    windows = np.lib.stride_tricks.sliding_window_view(masked, window)
    windows_n = np.lib.stride_tricks.sliding_window_view(has_n, window)
    a, b = windows[:, :-1], windows[:, 1:]
    din = (a * 4 + b).astype(np.int16)
    counts = np.zeros((len(windows), 16), dtype=np.int16)
    np.add.at(counts, (np.arange(len(windows))[:, None], din), 1)
    p = np.where(counts == 0, 1.0, counts / (window - 1)).astype(np.float32)
    entropy = -np.sum(p * np.log2(p), axis=1)
    perplexity = (2.0 ** entropy).astype(np.float32)
    perplexity[windows_n.any(axis=1)] = np.nan
    return perplexity


def _prefix(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(arr).astype(np.float64)
    values = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    values2 = np.where(np.isfinite(arr), arr.astype(np.float64) ** 2, 0.0)
    csum = np.empty(len(arr) + 1, dtype=np.float64)
    csum2 = np.empty(len(arr) + 1, dtype=np.float64)
    ccount = np.empty(len(arr) + 1, dtype=np.float64)
    csum[0] = csum2[0] = ccount[0] = 0.0
    np.cumsum(values, out=csum[1:])
    np.cumsum(values2, out=csum2[1:])
    np.cumsum(valid, out=ccount[1:])
    return csum, csum2, ccount


def _window_means_from_prefix(csum: np.ndarray, ccount: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.full(starts.shape, np.nan, dtype=np.float64)
    limit = len(csum) - 1
    valid = (starts >= 0) & (ends <= limit) & (ends > starts)
    if not np.any(valid):
        return out
    s = starts[valid]
    e = ends[valid]
    counts = ccount[e] - ccount[s]
    positive = counts > 0
    if not np.any(positive):
        return out
    idx = np.flatnonzero(valid)[positive]
    out[idx] = (csum[e[positive]] - csum[s[positive]]) / counts[positive]
    return out


def compute_p2(p1: np.ndarray, window: int = SECOND_ORDER_WINDOW) -> np.ndarray:
    if len(p1) == 0:
        return np.array([], dtype=np.float32)
    entropy1 = np.full(len(p1), np.nan, dtype=np.float64)
    finite = np.isfinite(p1) & (p1 > 0)
    entropy1[finite] = np.log2(p1[finite])
    csum, _, ccount = _prefix(entropy1)
    positions = np.arange(len(p1), dtype=np.int64)
    starts = positions - (window // 2)
    ends = starts + window
    mean_entropy = _window_means_from_prefix(csum, ccount, starts, ends)
    p2 = np.full(len(p1), np.nan, dtype=np.float64)
    valid = np.isfinite(mean_entropy)
    p2[valid] = np.power(2.0, mean_entropy[valid])
    return p2.astype(np.float32)


def _fixed_valley_profile(
    p2: np.ndarray,
    upstream_window: int,
    spacer: int,
    candidate_window: int,
    downstream_window: int,
) -> dict[str, np.ndarray]:
    n = len(p2)
    centers = np.arange(n, dtype=np.int64)
    candidate_start = centers - (candidate_window // 2)
    candidate_end = candidate_start + candidate_window
    upstream_end = candidate_start - spacer
    upstream_start = upstream_end - upstream_window
    downstream_start = candidate_end + spacer
    downstream_end = downstream_start + downstream_window
    csum, _, ccount = _prefix(p2)
    upstream_mean = _window_means_from_prefix(csum, ccount, upstream_start, upstream_end)
    candidate_mean = _window_means_from_prefix(csum, ccount, candidate_start, candidate_end)
    downstream_mean = _window_means_from_prefix(csum, ccount, downstream_start, downstream_end)
    upstream_difference = upstream_mean - candidate_mean
    downstream_difference = downstream_mean - candidate_mean
    pvs = ((upstream_mean + downstream_mean) / 2.0) - candidate_mean
    valid = (
        np.isfinite(upstream_mean)
        & np.isfinite(candidate_mean)
        & np.isfinite(downstream_mean)
        & np.isfinite(upstream_difference)
        & np.isfinite(downstream_difference)
        & np.isfinite(pvs)
    )
    for arr in (upstream_mean, candidate_mean, downstream_mean, upstream_difference, downstream_difference, pvs):
        arr[~valid] = np.nan
    candidate_window_track = np.zeros(n, dtype=np.int32)
    candidate_window_track[valid] = candidate_window
    candidate_start = candidate_start.astype(np.int32)
    candidate_end = (candidate_end - 1).astype(np.int32)
    candidate_start[~valid] = -1
    candidate_end[~valid] = -1
    return {
        "pvs": pvs.astype(np.float32),
        "upstream_mean": upstream_mean.astype(np.float32),
        "candidate_mean": candidate_mean.astype(np.float32),
        "downstream_mean": downstream_mean.astype(np.float32),
        "upstream_difference": upstream_difference.astype(np.float32),
        "downstream_difference": downstream_difference.astype(np.float32),
        "candidate_window": candidate_window_track,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
    }


def compute_valley_profile(
    p2: np.ndarray,
    mode: str = "fixed",
    upstream_window: int = UPSTREAM_WINDOW,
    spacer: int = SPACER,
    candidate_window: int = CANDIDATE_WINDOW,
    downstream_window: int = DOWNSTREAM_WINDOW,
    adaptive_min_window: int = ADAPTIVE_MIN_WINDOW,
    adaptive_max_window: int = ADAPTIVE_MAX_WINDOW,
) -> dict[str, np.ndarray]:
    if mode == "fixed":
        return _fixed_valley_profile(p2, upstream_window, spacer, candidate_window, downstream_window)
    min_window = max(1, int(adaptive_min_window))
    max_window = max(min_window, int(adaptive_max_window))
    best_profile = {
        "pvs": np.full(len(p2), np.nan, dtype=np.float32),
        "upstream_mean": np.full(len(p2), np.nan, dtype=np.float32),
        "candidate_mean": np.full(len(p2), np.nan, dtype=np.float32),
        "downstream_mean": np.full(len(p2), np.nan, dtype=np.float32),
        "upstream_difference": np.full(len(p2), np.nan, dtype=np.float32),
        "downstream_difference": np.full(len(p2), np.nan, dtype=np.float32),
        "candidate_window": np.zeros(len(p2), dtype=np.int32),
        "candidate_start": np.full(len(p2), -1, dtype=np.int32),
        "candidate_end": np.full(len(p2), -1, dtype=np.int32),
    }
    for current_window in range(min_window, max_window + 1):
        current = _fixed_valley_profile(p2, upstream_window, spacer, current_window, downstream_window)
        better = np.isfinite(current["pvs"]) & (
            ~np.isfinite(best_profile["pvs"]) | (current["pvs"] > best_profile["pvs"])
        )
        if not np.any(better):
            continue
        for key in best_profile:
            best_profile[key][better] = current[key][better]
    return best_profile


def bounded_min_mean(arr: np.ndarray, min_len: int, max_len: int) -> tuple[int | None, int | None, float]:
    n = len(arr)
    if n < min_len:
        return None, None, np.inf
    pref = np.empty(n + 1, dtype=np.float64)
    pref[0] = 0.0
    pref[1:] = np.cumsum(arr)
    best_mean = np.inf
    best_i = best_j = None
    dq: deque[int] = deque()
    for j in range(n):
        i_min = j - max_len + 1
        i_max = j - min_len + 1
        if i_max < 0:
            continue
        while dq and pref[dq[-1]] >= pref[i_max]:
            dq.pop()
        dq.append(i_max)
        while dq and dq[0] < i_min:
            dq.popleft()
        if dq:
            i = dq[0]
            mean = (pref[j + 1] - pref[i]) / (j - i + 1)
            if mean < best_mean:
                best_mean = mean
                best_i, best_j = i, j
    return best_i, best_j, float(best_mean)


def _nucleotide_bounds(signal_start: int, signal_end: int, seq_len: int) -> tuple[int, int]:
    start = max(0, int(signal_start))
    end = min(seq_len - 1, int(signal_end + PERPLEXITY_WINDOW - 1))
    return start, end


def domain_statistics(start: int, end: int, p1: np.ndarray, p2: np.ndarray, profile: dict[str, np.ndarray], seq: str) -> dict:
    signal_slice = slice(start, end + 1)
    domain_p1 = p1[signal_slice]
    domain_p2 = p2[signal_slice]
    domain_pvs = profile["pvs"][signal_slice]
    finite_p1 = domain_p1[np.isfinite(domain_p1)]
    finite_p2 = domain_p2[np.isfinite(domain_p2)]
    finite_pvs = domain_pvs[np.isfinite(domain_pvs)]
    mean_p1 = float(np.nanmean(finite_p1)) if finite_p1.size else float("nan")
    min_p1 = float(np.nanmin(finite_p1)) if finite_p1.size else float("nan")
    variance = float(np.nanvar(finite_p1)) if finite_p1.size else float("nan")
    sd = float(np.sqrt(variance)) if np.isfinite(variance) else float("nan")
    cv = float(sd / (mean_p1 + EPSILON)) if np.isfinite(sd) and np.isfinite(mean_p1) else float("nan")
    mean_p2 = float(np.nanmean(finite_p2)) if finite_p2.size else float("nan")
    mean_pvs = float(np.nanmean(finite_pvs)) if finite_pvs.size else float("nan")
    max_pvs = float(np.nanmax(finite_pvs)) if finite_pvs.size else float("nan")
    upstream_mean = float(np.nanmean(profile["upstream_mean"][signal_slice]))
    candidate_mean = float(np.nanmean(profile["candidate_mean"][signal_slice]))
    downstream_mean = float(np.nanmean(profile["downstream_mean"][signal_slice]))
    upstream_difference = float(np.nanmean(profile["upstream_difference"][signal_slice]))
    downstream_difference = float(np.nanmean(profile["downstream_difference"][signal_slice]))
    combined_valley_score = float(np.nanmean((
        profile["upstream_difference"][signal_slice] + profile["downstream_difference"][signal_slice]
    ) / 2.0))
    persistence = float(np.sum(finite_pvs > 0) / finite_pvs.size) if finite_pvs.size else 0.0
    signal_length = end - start + 1
    start_nt, end_nt = _nucleotide_bounds(start, end, len(seq))
    sequence = seq[start_nt:end_nt + 1]
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)
    stability = 1.0 / (variance + EPSILON) if np.isfinite(variance) else 0.0
    contrast = max(mean_pvs, 0.0) if np.isfinite(mean_pvs) else 0.0
    confidence_raw = contrast * persistence * math.log(max(signal_length, 2)) * stability
    candidate_windows = profile["candidate_window"][signal_slice]
    selected_window = int(np.nanmedian(candidate_windows[candidate_windows > 0])) if np.any(candidate_windows > 0) else 0
    return {
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Signal_Length": int(signal_length),
        "Start": int(start_nt),
        "End": int(end_nt),
        "Length": int(end_nt - start_nt + 1),
        "Mean_P1": mean_p1,
        "Mean_P2": mean_p2,
        "Mean_PVS": mean_pvs,
        "Max_PVS": max_pvs,
        "Min_P1": min_p1,
        "Upstream_Mean": upstream_mean,
        "Candidate_Mean": candidate_mean,
        "Downstream_Mean": downstream_mean,
        "Upstream_Difference": upstream_difference,
        "Downstream_Difference": downstream_difference,
        "Combined_Valley_Score": combined_valley_score,
        "Variance": variance,
        "SD": sd,
        "CV": cv,
        "GC_Content": gc,
        "Persistence": persistence,
        "Candidate_Window": selected_window,
        "Confidence_raw": confidence_raw,
        "Sequence": sequence,
    }


def normalize_confidence(domains: list[dict]) -> list[dict]:
    if not domains:
        return domains
    raw = np.array([max(0.0, float(d.get("Confidence_raw", 0.0))) for d in domains], dtype=np.float64)
    lo = float(raw.min())
    hi = float(raw.max())
    span = hi - lo
    if span < EPSILON:
        norm = np.where(raw > 0, 1.0, 0.0)
    else:
        norm = (raw - lo) / span
    for idx, domain in enumerate(domains, start=1):
        domain["Confidence"] = float(norm[idx - 1])
        domain["Domain_ID"] = f"REGPLEX_{idx:04d}"
        domain.pop("Confidence_raw", None)
    return domains


def find_domains(
    p1: np.ndarray,
    p2: np.ndarray,
    profile: dict[str, np.ndarray],
    seq: str,
    min_domain: int = MIN_DOMAIN,
    max_domain: int = MAX_DOMAIN,
) -> list[dict]:
    signal = profile["pvs"].astype(np.float64)
    work = -signal
    nan_mask = ~np.isfinite(work)
    n = len(work)
    domains: list[dict] = []
    while True:
        best = (np.inf, None, None)
        i = 0
        while i < n:
            if nan_mask[i]:
                i += 1
                continue
            j = i
            while j < n and not nan_mask[j]:
                j += 1
            seg = work[i:j]
            if len(seg) >= min_domain:
                s, e, mean_value = bounded_min_mean(seg, min_domain, max_domain)
                if s is not None and mean_value < best[0]:
                    best = (mean_value, i + s, i + e)
            i = j
        score, start, end = best
        if start is None or not np.isfinite(score) or score >= 0:
            break
        domains.append(domain_statistics(start, end, p1=p1, p2=p2, profile=profile, seq=seq))
        work[start:end + 1] = np.nan
        nan_mask[start:end + 1] = True
    domains.sort(key=lambda domain: domain["Signal_Start"])
    return normalize_confidence(domains)


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    params = {
        "perplexity_window": int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        "second_order_window": int(kwargs.get("second_order_window", SECOND_ORDER_WINDOW)),
        "candidate_mode": kwargs.get("candidate_mode", "fixed"),
        "candidate_window": int(kwargs.get("candidate_window", CANDIDATE_WINDOW)),
        "upstream_window": int(kwargs.get("upstream_window", UPSTREAM_WINDOW)),
        "downstream_window": int(kwargs.get("downstream_window", DOWNSTREAM_WINDOW)),
        "spacer": int(kwargs.get("spacer", SPACER)),
        "adaptive_min_window": int(kwargs.get("adaptive_min_window", ADAPTIVE_MIN_WINDOW)),
        "adaptive_max_window": int(kwargs.get("adaptive_max_window", ADAPTIVE_MAX_WINDOW)),
        "min_domain": int(kwargs.get("min_domain", MIN_DOMAIN)),
        "max_domain": int(kwargs.get("max_domain", MAX_DOMAIN)),
    }
    p1 = compute_p1(seq, params["perplexity_window"])
    p2 = compute_p2(p1, params["second_order_window"])
    profile = compute_valley_profile(
        p2,
        mode=params["candidate_mode"],
        upstream_window=params["upstream_window"],
        spacer=params["spacer"],
        candidate_window=params["candidate_window"],
        downstream_window=params["downstream_window"],
        adaptive_min_window=params["adaptive_min_window"],
        adaptive_max_window=params["adaptive_max_window"],
    )
    domains = find_domains(
        p1,
        p2,
        profile,
        seq,
        min_domain=params["min_domain"],
        max_domain=params["max_domain"],
    )
    for domain in domains:
        domain["Sequence_ID"] = sequence_id
    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        p1=p1,
        p2=p2,
        pvs=profile["pvs"],
        upstream_mean=profile["upstream_mean"],
        candidate_mean=profile["candidate_mean"],
        downstream_mean=profile["downstream_mean"],
        upstream_difference=profile["upstream_difference"],
        downstream_difference=profile["downstream_difference"],
        candidate_window=profile["candidate_window"],
        domains=domains,
        params=params,
    )


def domains_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        for domain in result.domains:
            rows.append({
                "Domain_ID": domain["Domain_ID"],
                "Sequence_ID": domain["Sequence_ID"],
                "Start": domain["Start"],
                "End": domain["End"],
                "Length": domain["Length"],
                "Signal_Start": domain["Signal_Start"],
                "Signal_End": domain["Signal_End"],
                "Signal_Length": domain["Signal_Length"],
                "Mean_P1": domain["Mean_P1"],
                "Mean_P2": domain["Mean_P2"],
                "Mean_PVS": domain["Mean_PVS"],
                "Max_PVS": domain["Max_PVS"],
                "Min_P1": domain["Min_P1"],
                "Upstream_Mean": domain["Upstream_Mean"],
                "Candidate_Mean": domain["Candidate_Mean"],
                "Downstream_Mean": domain["Downstream_Mean"],
                "Upstream_Difference": domain["Upstream_Difference"],
                "Downstream_Difference": domain["Downstream_Difference"],
                "Combined_Valley_Score": domain["Combined_Valley_Score"],
                "Variance": domain["Variance"],
                "SD": domain["SD"],
                "CV": domain["CV"],
                "GC_Content": domain["GC_Content"],
                "Persistence": domain["Persistence"],
                "Confidence": domain["Confidence"],
                "Candidate_Window": domain["Candidate_Window"],
                "Motif_Count": domain.get("Motif_Count", 0),
                "Motifs": domain.get("Motifs", ""),
                "Sequence": domain["Sequence"],
            })
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt == "csv":
        return df.to_csv(index=False).encode()
    if fmt == "tsv":
        return df.to_csv(index=False, sep="\t").encode()
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue()
    if fmt == "json":
        return df.to_json(orient="records", indent=2).encode()
    raise ValueError(f"Unsupported format: {fmt}")


def export_bed(df: pd.DataFrame) -> bytes:
    bed = df[["Sequence_ID", "Start", "End", "Domain_ID", "Confidence"]].copy()
    bed["Start"] = bed["Start"].astype(int)
    bed["End"] = bed["End"].astype(int) + 1
    return bed.to_csv(index=False, sep="\t", header=False).encode()


def export_fasta(df: pd.DataFrame) -> bytes:
    lines: list[str] = []
    for _, row in df.iterrows():
        lines.append(
            f">{row['Domain_ID']}|{row['Sequence_ID']}:{row['Start']}-{row['End']}|Confidence={row['Confidence']:.4f}"
        )
        lines.append(str(row["Sequence"]))
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def export_gff(df: pd.DataFrame, gff3: bool = False) -> bytes:
    rows: list[str] = []
    for _, row in df.iterrows():
        attr = (
            f"ID={row['Domain_ID']};Confidence={row['Confidence']:.4f};"
            f"MeanP1={row['Mean_P1']:.4f};MeanP2={row['Mean_P2']:.4f};"
            f"MeanPVS={row['Mean_PVS']:.4f};Motifs={row['Motifs']}"
        )
        rows.append("\t".join([
            str(row["Sequence_ID"]),
            "REGPLEX",
            "perplexity_valley",
            str(int(row["Start"]) + 1),
            str(int(row["End"]) + 1),
            f"{float(row['Confidence']):.4f}",
            ".",
            ".",
            attr,
        ]))
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run REGPLEX perplexity valley analysis from FASTA")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_domains.csv", help="Output CSV path")
    args = parser.parse_args()
    with open(args.fasta, encoding="utf-8") as handle:
        fasta_text = handle.read()
    records = parse_fasta(fasta_text)
    results = [analyze_sequence(header, sequence) for header, sequence in records]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} valleys to {args.out}")


if __name__ == "__main__":
    cli()
