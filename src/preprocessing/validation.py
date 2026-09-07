from __future__ import annotations

import re

from src.utils.constants import VALID_BASES


class SequenceValidationError(ValueError):
    pass


def normalize_and_validate_sequence(sequence: str) -> str:
    """Uppercase, strip whitespace, convert U->T, and validate DNA symbols."""
    compact = re.sub(r"\s+", "", sequence).upper().replace("U", "T")
    if not compact:
        raise SequenceValidationError("Sequence is empty after normalization.")
    invalid = sorted({ch for ch in compact if ch not in VALID_BASES})
    if invalid:
        raise SequenceValidationError(f"Invalid nucleotide characters: {''.join(invalid)}")
    return compact
