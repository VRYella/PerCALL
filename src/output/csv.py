from __future__ import annotations

import pandas as pd

from src.models.dataclasses import PredictionResult


def results_dataframe(result: PredictionResult) -> pd.DataFrame:
    rows = []
    for region in result.candidate_regions:
        rows.append({
            "Sequence_ID": result.sequence_id,
            "Start": region.start,
            "End": region.end,
            "Length": region.length,
            "Mean_Perplexity": region.mean_perplexity,
            "Background_Perplexity": region.background_perplexity,
            "Mean_PDS": region.mean_pds,
            "Max_PDS": region.max_pds,
            "Persistence": region.persistence,
            "Rank": region.rank,
        })
    return pd.DataFrame(rows)


def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()
