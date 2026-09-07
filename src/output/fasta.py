from __future__ import annotations

import pandas as pd


def export_region_fasta(df: pd.DataFrame, sequence: str) -> bytes:
    lines: list[str] = []
    for _, row in df.iterrows():
        start = int(row["Start"])
        end = int(row["End"])
        rank = int(row["Rank"])
        lines.append(f">region_{rank}|{row['Sequence_ID']}:{start}-{end}")
        lines.append(sequence[start:end])
    return ("
".join(lines) + ("
" if lines else "")).encode()
