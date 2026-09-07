from __future__ import annotations

DINUCLEOTIDES: tuple[str, ...] = (
    "AA", "AC", "AG", "AT",
    "CA", "CC", "CG", "CT",
    "GA", "GC", "GG", "GT",
    "TA", "TC", "TG", "TT",
)

VALID_BASES: frozenset[str] = frozenset({"A", "C", "G", "T", "N"})
EPSILON: float = 1e-9
