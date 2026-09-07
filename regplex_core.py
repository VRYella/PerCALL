from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.dataclasses import PerplexityConfig, PredictionResult
from src.output.bed import export_bed as _export_bed
from src.output.csv import export_csv as _export_csv
from src.output.fasta import export_region_fasta
from src.output.gff import export_gff as _export_gff
from src.perplexity.profile import calculate_perplexity_profile, smooth_profile
from src.prediction.background import estimate_local_background
from src.prediction.depression import calculate_perplexity_depression
from src.prediction.regulatory_predictor import predict_regulatory_regions
from src.preprocessing.fasta import parse_fasta
from src.preprocessing.sequence import clean_sequence

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
    unsupported = [k for k in ("spacer_size", "min_candidate", "max_candidate") if k in kwargs]
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"Unsupported legacy parameters in refactored predictor: {names}")

    return PerplexityConfig(
        perplexity_window=int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        step_size=int(kwargs.get("step_size", 1)),
        smoothing_window=int(kwargs.get("sg_window_length", SG_WINDOW_LENGTH)),
        smoothing_poly_order=int(kwargs.get("sg_poly_order", SG_POLY_ORDER)),
        flank_size=int(kwargs.get("flank_size", FLANK_SIZE)),
        min_region_length=int(kwargs.get("min_region_length", MIN_REGION_LENGTH)),
        max_region_length=int(kwargs.get("max_region_length", MAX_REGION_LENGTH)),
        min_perplexity_depression=float(kwargs.get("min_pds", kwargs.get("min_perplexity_depression", 0.25))),
        min_persistence_bp=int(kwargs.get("min_persistence_bp", kwargs.get("min_region_length", MIN_REGION_LENGTH))),
        merge_distance=int(kwargs.get("merge_gap", MERGE_GAP)),
    )


def _config_to_params(config: PerplexityConfig) -> dict:
    return {
        "perplexity_window": config.perplexity_window,
        "step_size": config.step_size,
        "sg_window_length": config.smoothing_window,
        "sg_poly_order": config.smoothing_poly_order,
        "flank_size": config.flank_size,
        "min_region_length": config.min_region_length,
        "max_region_length": config.max_region_length,
        "min_pds": config.min_perplexity_depression,
        "min_persistence_bp": config.min_persistence_bp,
        "merge_gap": config.merge_distance,
    }


def compute_di_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    cfg = PerplexityConfig(perplexity_window=window)
    return calculate_perplexity_profile(seq, cfg).raw_perplexity


def smooth_perplexity(arr: np.ndarray, window_length: int = SG_WINDOW_LENGTH, poly_order: int = SG_POLY_ORDER) -> np.ndarray:
    return smooth_profile(arr, window_length, poly_order)


def compute_pds(smoothed_di: np.ndarray, flank_size: int = FLANK_SIZE, **kwargs: int) -> np.ndarray:
    unsupported = [k for k in ("spacer_size", "min_candidate", "max_candidate") if k in kwargs]
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"Unsupported legacy parameters in refactored predictor: {names}")
    bg = estimate_local_background(smoothed_di, flank_size)
    return calculate_perplexity_depression(smoothed_di, bg)


def _prediction_to_analysis_result(prediction: PredictionResult, params: dict, sequence: str) -> AnalysisResult:
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
                "Sequence": sequence[region.start:region.end + 1],
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
    normalized = clean_sequence(seq)
    prediction = predict_regulatory_regions(sequence_id=sequence_id, sequence=normalized, config=config)
    return _prediction_to_analysis_result(prediction, params=_config_to_params(config), sequence=normalized)


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
        lines: list[str] = []
        for _, row in df.iterrows():
            lines.append(f">region_{int(row['Rank'])}|{row['Sequence_ID']}:{int(row['Start']) + 1}-{int(row['End']) + 1}")
            lines.append(str(row["Sequence"]))
        return ("\n".join(lines) + ("\n" if lines else "")).encode()

    chunks = []
    for sequence_id, sub_df in df.groupby("Sequence_ID"):
        chunks.append(export_region_fasta(sub_df, sequence_map.get(sequence_id, "")).decode())
    return "".join(chunks).encode()


def export_gff(df: pd.DataFrame, gff3: bool = True) -> bytes:
    return _export_gff(df, gff3=gff3)


def cli() -> None:
    parser = argparse.ArgumentParser(description="PerCALL candidate regulatory region detector")
    parser.add_argument("fasta", help="Input FASTA")
    parser.add_argument("--out", default="percall_regions.csv")
    parser.add_argument("--perplexity-window", type=int, default=PERPLEXITY_WINDOW)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--sg-window", type=int, default=SG_WINDOW_LENGTH)
    parser.add_argument("--sg-order", type=int, default=SG_POLY_ORDER)
    parser.add_argument("--flank-size", type=int, default=FLANK_SIZE)
    parser.add_argument("--min-region", type=int, default=MIN_REGION_LENGTH)
    parser.add_argument("--max-region", type=int, default=MAX_REGION_LENGTH)
    parser.add_argument("--min-pds", type=float, default=0.25)
    parser.add_argument("--min-persistence", type=int, default=MIN_REGION_LENGTH)
    parser.add_argument("--merge-gap", type=int, default=MERGE_GAP)
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        records = parse_fasta(handle.read())

    results = [
        analyze_sequence(
            header,
            sequence,
            perplexity_window=args.perplexity_window,
            step_size=args.step_size,
            sg_window_length=args.sg_window,
            sg_poly_order=args.sg_order,
            flank_size=args.flank_size,
            min_region_length=args.min_region,
            max_region_length=args.max_region,
            min_pds=args.min_pds,
            min_persistence_bp=args.min_persistence,
            merge_gap=args.merge_gap,
        )
        for header, sequence in records
    ]
    df = regions_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} candidate regions to {args.out}")


if __name__ == "__main__":
    cli()
