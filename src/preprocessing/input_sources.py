from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from src.preprocessing.fasta import parse_fasta
from src.preprocessing.validation import SequenceValidationError

NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class InputSourceError(ValueError):
    pass


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_allowed_input_path(path: str) -> Path:
    fasta_path = Path(path).expanduser()
    if not fasta_path.is_absolute():
        raise InputSourceError("Disk input must use an absolute path.")
    resolved = fasta_path.resolve()
    allowed_roots = [Path.cwd().resolve(), Path("/tmp").resolve()]
    if not any(_is_within(resolved, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise InputSourceError(f"Disk input must stay within approved roots: {roots}")
    return resolved


def _decode_uploaded_text(uploaded_bytes: bytes) -> str:
    try:
        return uploaded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputSourceError("Uploaded file must be valid UTF-8 FASTA/text.") from exc


def load_records_from_text(text: str) -> list[tuple[str, str]]:
    try:
        records = parse_fasta(text)
    except SequenceValidationError as exc:
        raise InputSourceError(str(exc)) from exc
    if not records:
        raise InputSourceError("No valid sequence records found. Provide FASTA or a plain DNA sequence.")
    return records


def load_records_from_path(path: str) -> list[tuple[str, str]]:
    fasta_path = _resolve_allowed_input_path(path)
    if not fasta_path.exists():
        raise InputSourceError(f"Input file does not exist: {fasta_path}")
    if not fasta_path.is_file():
        raise InputSourceError(f"Input path is not a file: {fasta_path}")
    return load_records_from_text(fasta_path.read_text(encoding="utf-8"))


def fetch_ncbi_fasta(accession: str, timeout: float = 20.0) -> str:
    accession = accession.strip()
    if not accession:
        raise InputSourceError("NCBI accession is empty.")
    query = urlencode({"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"})
    with urlopen(f"{NCBI_EFETCH_URL}?{query}", timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    if not payload.strip() or not payload.lstrip().startswith(">"):
        raise InputSourceError(f"NCBI did not return FASTA for accession '{accession}'.")
    return payload


def load_records_from_accession(accession: str) -> list[tuple[str, str]]:
    return load_records_from_text(fetch_ncbi_fasta(accession))


def load_sequence_records(
    *,
    pasted_text: str = "",
    file_path: str = "",
    accession: str = "",
    uploaded_bytes: bytes | None = None,
) -> tuple[list[tuple[str, str]], str]:
    if pasted_text.strip():
        return load_records_from_text(pasted_text), "Pasted sequence"
    if uploaded_bytes is not None:
        return load_records_from_text(_decode_uploaded_text(uploaded_bytes)), "Uploaded file"
    if file_path.strip():
        return load_records_from_path(file_path), "Disk file"
    if accession.strip():
        return load_records_from_accession(accession), f"NCBI accession: {accession.strip()}"
    raise InputSourceError("Provide pasted sequence text, an uploaded file, a disk path, or an NCBI accession.")
