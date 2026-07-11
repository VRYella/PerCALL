from __future__ import annotations

import re

IUPAC_MAP = {
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
    "N": "[ACGT]",
}

_IUPAC_ONLY = set("ACGTRYSWKMBDHVN")


def _is_iupac(pattern: str) -> bool:
    stripped = pattern.strip().upper()
    return bool(stripped) and set(stripped) <= _IUPAC_ONLY


def iupac_to_regex(pattern: str) -> str:
    return "".join(IUPAC_MAP.get(ch, ch) for ch in pattern.strip().upper())


def compile_motifs(text: str) -> list[tuple[str, re.Pattern]]:
    motifs: list[tuple[str, re.Pattern]] = []
    for line in text.splitlines():
        motif = line.strip()
        if not motif:
            continue
        regex = iupac_to_regex(motif) if _is_iupac(motif) else motif
        motifs.append((motif, re.compile(regex, re.IGNORECASE)))
    return motifs


def annotate_regions(regions: list[dict], compiled_motifs: list[tuple[str, re.Pattern]]) -> list[dict]:
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
        region["MotifCount"] = total
        region["Motifs"] = ";".join(hits)
    return regions

