from __future__ import annotations

from src.preprocessing.validation import normalize_and_validate_sequence


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if header is None:
            return
        sequence = normalize_and_validate_sequence("".join(chunks))
        records.append((header, sequence))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                flush()
            header = line[1:].strip() or "sequence"
            chunks = []
        else:
            chunks.append(line)

    if header is not None:
        flush()
    elif chunks:
        records.append(("query", normalize_and_validate_sequence("".join(chunks))))

    return records
