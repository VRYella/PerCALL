from __future__ import annotations

import pandas as pd


def export_bed(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    bed = df[["Sequence_ID", "Start", "End", "Rank", "Mean_PDS"]].copy()
    bed["End"] = bed["End"].astype(int) + 1
    return bed.to_csv(index=False, sep="\t", header=False).encode()
