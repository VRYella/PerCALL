from pathlib import Path

from src.preprocessing.input_sources import InputSourceError, load_records_from_path, load_sequence_records


def test_load_sequence_records_from_text():
    records, source = load_sequence_records(pasted_text=">seq1\nACGT\n")
    assert source == "Pasted sequence"
    assert records == [("seq1", "ACGT")]


def test_load_records_from_path(tmp_path: Path):
    path = tmp_path / "example.fasta"
    path.write_text(">seq2\nAACCGGTT\n", encoding="utf-8")
    records = load_records_from_path(str(path))
    assert records == [("seq2", "AACCGGTT")]


def test_load_sequence_records_requires_one_source():
    try:
        load_sequence_records()
    except InputSourceError as exc:
        assert "indexed local file" in str(exc)
    else:
        raise AssertionError("Expected InputSourceError")
