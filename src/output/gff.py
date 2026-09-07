from __future__ import annotations

from urllib.parse import quote

import pandas as pd


def _escape_attr(value: str) -> str:
    return quote(str(value), safe="._-:")


def export_gff(df: pd.DataFrame, gff3: bool = True) -> bytes:
    rows: list[str] = []
    for _, row in df.iterrows():
        attrs = (
            f"ID={_escape_attr(f'region_{int(row['Rank'])}')};"
            f"mean_pds={_escape_attr(f'{float(row['Mean_PDS']):.4f}')};"
            f"max_pds={_escape_attr(f'{float(row['Max_PDS']):.4f}')};"
            f"background={_escape_attr(f'{float(row['Background_Perplexity']):.4f}')}"
        )
        rows.append("\t".join([
            str(row["Sequence_ID"]),
            "PerCALL",
            "candidate_regulatory_region",
            str(int(row["Start"]) + 1),
            str(int(row["End"]) + 1),
            f"{float(row['Mean_PDS']):.4f}",
            ".",
            ".",
            attrs,
        ]))
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()
