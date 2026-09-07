from __future__ import annotations

from src.preprocessing.validation import normalize_and_validate_sequence


def clean_sequence(sequence: str) -> str:
    return normalize_and_validate_sequence(sequence)
