from __future__ import annotations

import pandas as pd


def export_gff(df: pd.DataFrame) -> bytes:
    rows: list[str] = []
    for _, row in df.iterrows():
        attrs = (
            f"ID=region_{int(row['Rank'])};"
            f"mean_pds={float(row['Mean_PDS']):.4f};"
            f"max_pds={float(row['Max_PDS']):.4f};"
            f"background={float(row['Background_Perplexity']):.4f}"
        )
        rows.append("	".join([
            str(row["Sequence_ID"]),
            "PerCALL",
            "candidate_regulatory_region",
            str(int(row["Start"]) + 1),
            str(int(row["End"])),
            f"{float(row['Mean_PDS']):.4f}",
            ".",
            ".",
            attrs,
        ]))
    return ("##gff-version 3
" + "
".join(rows) + ("
" if rows else "")).encode()
