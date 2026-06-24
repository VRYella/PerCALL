from __future__ import annotations

import re

# Built-in non-B DNA structural motif patterns (always applied during annotation).
NON_B_MOTIFS: dict[str, str] = {
    "G_Quadruplex": r"G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}",
    "I_Motif": r"C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}",
    "Z_DNA": r"(?:[GC][AT]){3,}",
    "A_Phased": r"AAAA[ACGT]{4,8}AAAA",
    "H_DNA_Purine": r"[AG]{10,}",
    "H_DNA_Pyrimidine": r"[TC]{10,}",
}

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
    # Always include built-in non-B DNA structural motifs.
    for name, pattern in NON_B_MOTIFS.items():
        motifs.append((name, re.compile(pattern)))
    # Append any user-supplied motifs (IUPAC or raw regex).
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
            count = len(pat.findall(seq))
            if count:
                hits.append(f"{raw}:{count}")
        total = 0
        for h in hits:
            try:
                total += int(h.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                continue
        d["Motif_Count"] = total
        d["Motifs"] = ";".join(hits)
    return domains
