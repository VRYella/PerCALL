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
DOMAIN_WINDOW = 100
UPSTREAM_WINDOW = 100
DOWNSTREAM_WINDOW = 100
SPACER = 50
MIN_DOMAIN = 50
MAX_DOMAIN = 1000
TOP_DOMAINS = 100
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
    pdi: np.ndarray
    domains: list[dict]


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


def _window_mean(csum: np.ndarray, ccount: np.ndarray, a: int, b: int) -> float:
    a = max(a, 0)
    b = min(b, len(csum) - 1)
    if b <= a:
        return float("nan")
    n = ccount[b] - ccount[a]
    if n <= 0:
        return float("nan")
    return float((csum[b] - csum[a]) / n)


def compute_pdi(
    p1: np.ndarray,
    domain_window: int = DOMAIN_WINDOW,
    upstream_window: int = UPSTREAM_WINDOW,
    downstream_window: int = DOWNSTREAM_WINDOW,
    spacer: int = SPACER,
) -> np.ndarray:
    n = len(p1)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out
    dom_half = domain_window // 2
    margin = max(dom_half + spacer + upstream_window, dom_half + spacer + downstream_window)
    csum, _, ccount = _prefix(p1)
    for d in range(margin, n - margin):
        dom_start = d - dom_half
        dom_end = dom_start + domain_window
        up_end = dom_start - spacer
        up_start = up_end - upstream_window
        dn_start = dom_end + spacer
        dn_end = dn_start + downstream_window
        u = _window_mean(csum, ccount, up_start, up_end)
        dm = _window_mean(csum, ccount, dom_start, dom_end)
        v = _window_mean(csum, ccount, dn_start, dn_end)
        if np.isfinite(u) and np.isfinite(dm) and np.isfinite(v):
            out[d] = np.float32(((u + v) / 2.0) - dm)
    return out


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


def find_domains(
    p1: np.ndarray,
    pdi: np.ndarray,
    seq: str,
    min_domain: int = MIN_DOMAIN,
    max_domain: int = MAX_DOMAIN,
    top_domains: int = TOP_DOMAINS,
) -> list[dict]:
    signal = (-pdi).astype(np.float64)
    work = signal.copy()
    nan_mask = ~np.isfinite(work)
    n = len(work)
    domains: list[dict] = []
    for _ in range(top_domains):
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
                s, e, m = bounded_min_mean(seg, min_domain, max_domain)
                if s is not None and m < best[0]:
                    best = (m, i + s, i + e)
            i = j
        score, start, end = best
        if start is None:
            break
        info = domain_statistics(start, end, p1=p1, pdi=pdi, seq=seq)
        info["Start"] = int(start)
        info["End"] = int(end)
        info["Length"] = int(end - start + 1)
        info["Mean_Kadane"] = float(score)
        domains.append(info)
        work[start:end + 1] = np.nan
        nan_mask[start:end + 1] = True
    return domains


def domain_statistics(start: int, end: int, p1: np.ndarray, pdi: np.ndarray, seq: str) -> dict:
    win_p1 = p1[start:end + 1]
    win_pdi = pdi[start:end + 1]
    s = seq[start:end + 1] if end < len(seq) else seq[start:]
    finite_p1 = win_p1[np.isfinite(win_p1)]
    finite_pdi = win_pdi[np.isfinite(win_pdi)]
    mean_p1 = float(np.nanmean(finite_p1)) if finite_p1.size else float("nan")
    var_p1 = float(np.nanvar(finite_p1)) if finite_p1.size else float("nan")
    sd_p1 = float(np.sqrt(var_p1)) if np.isfinite(var_p1) else float("nan")
    cv_p1 = float(sd_p1 / (mean_p1 + EPSILON)) if np.isfinite(sd_p1) and np.isfinite(mean_p1) else float("nan")
    # Persistence is the fraction of positions with positive PDI inside the domain.
    persistence = (float(np.sum(finite_pdi > 0) / len(finite_pdi)) if finite_pdi.size else 0.0)
    mean_pdi = float(np.nanmean(finite_pdi)) if finite_pdi.size else float("nan")
    length = max(1, end - start + 1)
    stability = 1.0 / (var_p1 + EPSILON) if np.isfinite(var_p1) else 0.0
    # RCS = Contrast(mean_pdi) × Persistence × LengthFactor(log L) × Stability(1/variance).
    rcs_raw = (mean_pdi if np.isfinite(mean_pdi) else 0.0) * persistence * math.log(length + 1) * stability
    gc = (s.count("G") + s.count("C")) / max(len(s), 1)
    return {
        "Mean_P1": mean_p1,
        "Min_P1": float(np.nanmin(finite_p1)) if finite_p1.size else float("nan"),
        "Max_P1": float(np.nanmax(finite_p1)) if finite_p1.size else float("nan"),
        "Mean_PDI": mean_pdi,
        "Max_PDI": float(np.nanmax(finite_pdi)) if finite_pdi.size else float("nan"),
        "Variance_P1": var_p1,
        "SD_P1": sd_p1,
        "CV_P1": cv_p1,
        "GC_Content": gc,
        "Persistence": persistence,
        "RCS_raw": rcs_raw,
        "Sequence": s,
    }


def score_and_classify(domains: list[dict]) -> list[dict]:
    if not domains:
        return domains
    raw = np.array([d.get("RCS_raw", 0.0) for d in domains], dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    for i, d in enumerate(domains):
        rcs = 0.0 if span < EPSILON else float((raw[i] - lo) / span)
        d["RCS"] = rcs
        d["Class"] = "I" if rcs > 0.80 else "II" if rcs >= 0.60 else "III" if rcs >= 0.40 else "IV"
        d["Domain_ID"] = f"REGPLEX_{i + 1:04d}"
        d.pop("RCS_raw", None)
    return domains


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    p1 = compute_p1(seq, kwargs.get("perplexity_window", PERPLEXITY_WINDOW))
    pdi = compute_pdi(
        p1,
        kwargs.get("domain_window", DOMAIN_WINDOW),
        kwargs.get("upstream_window", UPSTREAM_WINDOW),
        kwargs.get("downstream_window", DOWNSTREAM_WINDOW),
        kwargs.get("spacer", SPACER),
    )
    domains = []
    for d in find_domains(
        p1,
        pdi,
        seq,
        kwargs.get("min_domain", MIN_DOMAIN),
        kwargs.get("max_domain", MAX_DOMAIN),
        kwargs.get("top_domains", TOP_DOMAINS),
    ):
        d["Sequence_ID"] = sequence_id
        domains.append(d)
    score_and_classify(domains)
    return AnalysisResult(sequence_id=sequence_id, length=len(seq), p1=p1, pdi=pdi, domains=domains)


def domains_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        for d in res.domains:
            rows.append({
                "Domain_ID": d["Domain_ID"],
                "Sequence_ID": d["Sequence_ID"],
                "Start": d["Start"],
                "End": d["End"],
                "Length": d["Length"],
                "Mean_P1": d["Mean_P1"],
                "Min_P1": d["Min_P1"],
                "Max_P1": d["Max_P1"],
                "Mean_PDI": d["Mean_PDI"],
                "Max_PDI": d["Max_PDI"],
                "Variance_P1": d["Variance_P1"],
                "SD_P1": d["SD_P1"],
                "CV_P1": d["CV_P1"],
                "GC_Content": d["GC_Content"],
                "RCS": d["RCS"],
                "Class": d["Class"],
                "Motif_Count": d.get("Motif_Count", 0),
                "Motifs": d.get("Motifs", ""),
                "Sequence": d["Sequence"],
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
    bed = df[["Sequence_ID", "Start", "End", "Domain_ID", "RCS"]].copy()
    bed["Start"] = bed["Start"].astype(int)
    bed["End"] = bed["End"].astype(int) + 1
    return bed.to_csv(index=False, sep="\t", header=False).encode()


def export_fasta(df: pd.DataFrame) -> bytes:
    lines: list[str] = []
    for _, row in df.iterrows():
        lines.append(f">{row['Domain_ID']}|{row['Sequence_ID']}:{row['Start']}-{row['End']}|RCS={row['RCS']:.4f}")
        lines.append(str(row["Sequence"]))
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def export_gff(df: pd.DataFrame, gff3: bool = False) -> bytes:
    rows: list[str] = []
    for _, r in df.iterrows():
        attr = (
            f"ID={r['Domain_ID']};RCS={r['RCS']:.4f};Class={r['Class']};"
            f"MeanP1={r['Mean_P1']:.4f};MeanPDI={r['Mean_PDI']:.4f};"
            f"Variance={r['Variance_P1']:.4f};Motifs={r['Motifs']}"
        )
        row = [
            str(r["Sequence_ID"]),
            "REGPLEX",
            "uncertainty_collapse_domain",
            str(int(r["Start"]) + 1),
            str(int(r["End"]) + 1),
            f"{float(r['RCS']):.4f}",
            ".",
            ".",
            attr,
        ]
        rows.append("\t".join(row))
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run REGPLEX core analysis from FASTA")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_domains.csv", help="Output CSV path")
    args = parser.parse_args()
    with open(args.fasta, encoding="utf-8") as fh:
        fasta_text = fh.read()
    records = parse_fasta(fasta_text)
    results = [analyze_sequence(h, s) for h, s in records]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} domains to {args.out}")


if __name__ == "__main__":
    cli()
