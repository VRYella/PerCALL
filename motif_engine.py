from __future__ import annotations

from src.motifs import compile_motifs as _compile_motifs
from src.motifs import iupac_to_regex


def compile_motifs(text: str) -> list[tuple[str, object]]:
    return [(motif.name, motif.regex) for motif in _compile_motifs(text)]


def annotate_regions(regions: list[dict], compiled_motifs: list[tuple[str, object]]) -> list[dict]:
    """Annotate detected regions with motif counts and per-pattern hit summaries."""
    for region in regions:
        sequence = region.get("Sequence", "")
        hits: list[str] = []
        total = 0
        for motif, pattern in compiled_motifs:
            count = sum(1 for _ in pattern.finditer(sequence))
            if count:
                hits.append(f"{motif}:{count}")
                total += count
        region["Motif_Count"] = total
        region["Motifs"] = ";".join(hits)
    return regions
