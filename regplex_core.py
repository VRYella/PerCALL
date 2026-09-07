from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.dataclasses import PerplexityConfig, PredictionResult
from src.output.bed import export_bed as _export_bed
from src.output.csv import export_csv as _export_csv
from src.output.csv import results_dataframe
from src.output.fasta import export_region_fasta
from src.output.gff import export_gff as _export_gff
from src.perplexity.profile import calculate_perplexity_profile, smooth_profile
from src.prediction.background import estimate_local_background
from src.prediction.depression import calculate_perplexity_depression
from src.prediction.regulatory_predictor import predict_regulatory_regions
from src.preprocessing.fasta import parse_fasta

PERPLEXITY_WINDOW = 17
SG_WINDOW_LENGTH = 21
SG_POLY_ORDER = 3
FLANK_SIZE = 100
MIN_REGION_LENGTH = 100
MAX_REGION_LENGTH = 1000
MERGE_GAP = 100
TOP_N_DISPLAY = 20

PRIMARY_COLS = [
    "Sequence_ID",
    "Start",
    "End",
    "Length",
    "Mean_Perplexity",
    "Background_Perplexity",
    "Mean_PDS",
    "Max_PDS",
    "Persistence",
    "Rank",
]
ADVANCED_COLS: list[str] = []


@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    di: np.ndarray
    smoothed_di: np.ndarray
    pds: np.ndarray
    regions: list[dict]
    params: dict


def _config_from_kwargs(kwargs: dict) -> PerplexityConfig:
    return PerplexityConfig(
        perplexity_window=int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        step_size=int(kwargs.get("step_size", 1)),
        smoothing_window=int(kwargs.get("sg_window_length", SG_WINDOW_LENGTH)),
        smoothing_poly_order=int(kwargs.get("sg_poly_order", SG_POLY_ORDER)),
        flank_size=int(kwargs.get("flank_size", FLANK_SIZE)),
        min_region_length=int(kwargs.get("min_region_length", MIN_REGION_LENGTH)),
        max_region_length=int(kwargs.get("max_region_length", MAX_REGION_LENGTH)),
        min_perplexity_depression=float(kwargs.get("min_pds", kwargs.get("min_perplexity_depression", 0.25))),
        min_persistence_bp=int(kwargs.get("min_persistence_bp", 80)),
        merge_distance=int(kwargs.get("merge_gap", MERGE_GAP)),
    )


def compute_di_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    cfg = PerplexityConfig(perplexity_window=window)
    return calculate_perplexity_profile(seq, cfg).raw_perplexity


def smooth_perplexity(arr: np.ndarray, window_length: int = SG_WINDOW_LENGTH, poly_order: int = SG_POLY_ORDER) -> np.ndarray:
    return smooth_profile(arr, window_length, poly_order)


def compute_pds(smoothed_di: np.ndarray, flank_size: int = FLANK_SIZE, **_: int) -> np.ndarray:
    bg = estimate_local_background(smoothed_di, flank_size)
    return calculate_perplexity_depression(smoothed_di, bg)


def _prediction_to_analysis_result(prediction: PredictionResult, params: dict) -> AnalysisResult:
    regions: list[dict] = []
    for region in prediction.candidate_regions:
        regions.append(
            {
                "Sequence_ID": prediction.sequence_id,
                "Start": region.start,
                "End": region.end,
                "Length": region.length,
                "Mean_Perplexity": region.mean_perplexity,
                "Background_Perplexity": region.background_perplexity,
                "Mean_PDS": region.mean_pds,
                "Max_PDS": region.max_pds,
                "Persistence": region.persistence,
                "Rank": region.rank,
                "Perplexity_Depression_Score": region.mean_pds,
                "Region_Score": region.mean_pds,
            }
        )

    return AnalysisResult(
        sequence_id=prediction.sequence_id,
        length=prediction.sequence_length,
        di=prediction.profile.raw_perplexity,
        smoothed_di=prediction.profile.smoothed_perplexity,
        pds=prediction.pds,
        regions=regions,
        params=params,
    )


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    config = _config_from_kwargs(kwargs)
    prediction = predict_regulatory_regions(sequence_id=sequence_id, sequence=seq, config=config)
    return _prediction_to_analysis_result(prediction, params={**kwargs})


def regions_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        rows.extend(result.regions)
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, fmt: str, include_advanced: bool = False) -> bytes:
    _ = include_advanced
    if fmt.lower() in {"csv", "tsv", "json"}:
        if fmt.lower() == "csv":
            return _export_csv(df)
        if fmt.lower() == "tsv":
            return df.to_csv(index=False, sep="\t").encode()
        return df.to_json(orient="records", indent=2).encode()
    if fmt.lower() == "xlsx":
        import io

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue()
    raise ValueError(f"Unsupported format: {fmt}")


def export_bed(df: pd.DataFrame) -> bytes:
    return _export_bed(df)


def export_fasta(df: pd.DataFrame, sequence_map: dict[str, str] | None = None) -> bytes:
    if sequence_map is None:
        return b""
    chunks = []
    for sequence_id, sub_df in df.groupby("Sequence_ID"):
        chunks.append(export_region_fasta(sub_df, sequence_map.get(sequence_id, "")).decode())
    return "".join(chunks).encode()


def export_gff(df: pd.DataFrame, gff3: bool = True) -> bytes:
    _ = gff3
    return _export_gff(df)


def cli() -> None:
    parser = argparse.ArgumentParser(description="PerCALL candidate regulatory region detector")
    parser.add_argument("fasta", help="Input FASTA")
    parser.add_argument("--out", default="percall_regions.csv")
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        records = parse_fasta(handle.read())

    results = [analyze_sequence(header, sequence) for header, sequence in records]
    df = regions_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} candidate regions to {args.out}")


if __name__ == "__main__":
    cli()
