from src.motifs import annotate_sequence, compile_motifs


def test_compile_motifs_supports_named_and_iupac_lines():
    motifs = compile_motifs("# comment\nTATA_box\tTATAWAAR\nGGGCGG\n")
    assert [motif.name for motif in motifs] == ["TATA_box", "GGGCGG"]
    assert motifs[0].regex.pattern == "TATA[AT]AA[AG]"


def test_annotate_sequence_reports_counts():
    motifs = compile_motifs("TATA_box\tTATA[AT]A[AT][AG]\nGC_box\tGGGCGG\n")
    total, summary = annotate_sequence("TATATAAAGGGCGGTATATAAA", motifs)
    assert total == 3
    assert "TATA_box:2" in summary
    assert "GC_box:1" in summary
