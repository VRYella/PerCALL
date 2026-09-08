from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class CompiledMotif:
    name: str
    pattern: str
    regex: re.Pattern[str]


def _is_iupac(pattern: str) -> bool:
    stripped = pattern.strip().upper()
    return bool(stripped) and set(stripped) <= _IUPAC_ONLY


def iupac_to_regex(pattern: str) -> str:
    return "".join(IUPAC_MAP.get(ch, ch) for ch in pattern.strip().upper())


def _parse_motif_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "\t" in stripped:
        name, pattern = stripped.split("\t", 1)
        return name.strip(), pattern.strip()
    return stripped, stripped


def compile_motifs(text: str) -> list[CompiledMotif]:
    motifs: list[CompiledMotif] = []
    for line in text.splitlines():
        parsed = _parse_motif_line(line)
        if parsed is None:
            continue
        name, pattern = parsed
        regex = iupac_to_regex(pattern) if _is_iupac(pattern) else pattern
        motifs.append(CompiledMotif(name=name, pattern=pattern, regex=re.compile(regex, re.IGNORECASE)))
    return motifs


def load_motifs_from_path(path: str | Path) -> list[CompiledMotif]:
    return compile_motifs(Path(path).read_text(encoding="utf-8"))


def annotate_sequence(sequence: str, compiled_motifs: list[CompiledMotif]) -> tuple[int, str]:
    hits: list[str] = []
    total = 0
    for motif in compiled_motifs:
        count = sum(1 for _ in motif.regex.finditer(sequence))
        if count:
            hits.append(f"{motif.name}:{count}")
            total += count
    return total, ";".join(hits)
