import numpy as np

from src.supervised.pipeline import SplitConfig, _metrics, canonical_signature, split_leakage_safe


def test_canonical_signature_groups_reverse_complements():
    seq = "ATGCCGTA"
    rc = "TACGGCAT"
    assert canonical_signature(seq) == canonical_signature(rc)


def test_leakage_safe_split_has_no_cross_split_signature_overlap():
    sequences = np.array([
        "AAAAAAAAAA",
        "AAAAAAAAAA",
        "TTTTTTTTTT",  # reverse-complement of AAAAAAAAAA
        "CCCCCCCCCC",
        "GGGGGGGGGG",  # reverse-complement of CCCCCCCCCC
        "ACGTACGTAC",
        "GTACGTACGT",  # reverse-complement
        "TGCATGCATG",
        "CATGCATGCA",  # reverse-complement
        "AGAGAGAGAG",
        "CTCTCTCTCT",  # reverse-complement
        "ATATATATAT",
    ], dtype=object)
    labels = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
    species = np.array(["human", "human", "human", "human", "human", "ecoli", "ecoli", "ecoli", "ecoli", "human", "human", "ecoli"], dtype=object)

    split = split_leakage_safe(sequences, labels, species, SplitConfig(test_fraction=0.33, val_fraction=0.17, seed=7))

    signatures = np.array([canonical_signature(s) for s in sequences], dtype=object)
    train_sig = set(signatures[split["train_idx"]].tolist())
    val_sig = set(signatures[split["val_idx"]].tolist())
    test_sig = set(signatures[split["test_idx"]].tolist())

    assert train_sig.isdisjoint(val_sig)
    assert train_sig.isdisjoint(test_sig)
    assert val_sig.isdisjoint(test_sig)


def test_conflicting_duplicate_labels_are_dropped():
    sequences = np.array(["ACGTACGT", "ACGTACGT", "TTTTAAAA", "GGGGCCCC"], dtype=object)
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    species = np.array(["human", "ecoli", "human", "ecoli"], dtype=object)

    split = split_leakage_safe(sequences, labels, species, SplitConfig(seed=1))
    assert split["dropped_conflicting_sequences"]["count"] == 2


def test_metrics_known_values():
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    y_pred = np.array([1, 1, 0, 0, 1, 0], dtype=np.int64)
    m = _metrics(y_true, y_pred)
    assert np.isclose(m["precision"], 2 / 3)
    assert np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["f1"], 2 / 3)
    assert np.isclose(m["mcc"], 1 / 3)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 1, 1)
