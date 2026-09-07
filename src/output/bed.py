from __future__ import annotations

import pandas as pd


def export_bed(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    bed = df[["Sequence_ID", "Start", "End", "Rank", "Mean_PDS"]].copy()
    return bed.to_csv(index=False, sep="	", header=False).encode()
