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


def iupac_to_regex(pattern: str) -> str:
    out: list[str] = []
    for ch in pattern.strip().upper():
        out.append(IUPAC_MAP.get(ch, ch))
    return "".join(out)


def compile_motifs(text: str) -> list[tuple[str, re.Pattern]]:
    motifs: list[tuple[str, re.Pattern]] = []
    for line in text.splitlines():
        motif = line.strip()
        if not motif:
            continue
        rx = iupac_to_regex(motif)
        motifs.append((motif, re.compile(rx)))
    return motifs


def annotate_domains(domains: list[dict], compiled_motifs: list[tuple[str, re.Pattern]]) -> list[dict]:
    if not compiled_motifs:
        for d in domains:
            d["Motif_Count"] = 0
            d["Motifs"] = ""
        return domains
    for d in domains:
        seq = d.get("Sequence", "")
        hits = []
        for raw, pat in compiled_motifs:
            count = len(list(pat.finditer(seq)))
            if count:
                hits.append(f"{raw}:{count}")
        d["Motif_Count"] = sum(int(h.split(":", 1)[1]) for h in hits) if hits else 0
        d["Motifs"] = ";".join(hits)
    return domains
