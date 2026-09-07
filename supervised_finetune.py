from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.supervised.pipeline import HyperParamConfig, SplitConfig, load_human_ecoli_dataset, run_supervised_finetuning


def cli() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for human + E. coli promoter classification")
    parser.add_argument("--cache-dir", default="/tmp/regplex_supervised_data", help="Local directory for downloaded FASTA files")
    parser.add_argument("--output-json", default="/tmp/regplex_supervised_report.json", help="Path to write metrics report")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-species-class", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    sequences, labels, species, _ = load_human_ecoli_dataset(
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        max_samples_per_species_class=args.max_samples_per_species_class,
        seed=args.seed,
    )

    report = run_supervised_finetuning(
        sequences=sequences,
        labels=labels,
        species=species,
        split_cfg=SplitConfig(test_fraction=args.test_fraction, val_fraction=args.val_fraction, seed=args.seed),
        hyper_cfg=HyperParamConfig(),
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    tm = report["test_metrics"]
    print(
        "\nTest metrics: "
        f"precision={tm['precision']:.4f}, recall={tm['recall']:.4f}, "
        f"f1={tm['f1']:.4f}, mcc={tm['mcc']:.4f}"
    )


if __name__ == "__main__":
    cli()
