from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score

from src.preprocessing.fasta import parse_fasta


DATASET_URLS: dict[str, tuple[str, int, str]] = {
    "human_positive": (
        "https://raw.githubusercontent.com/nmach22/Promoter-Classification/master/dataset/human_non_tata.fa",
        1,
        "human",
    ),
    "human_negative": (
        "https://raw.githubusercontent.com/nmach22/Promoter-Classification/master/dataset/human_nonprom_big.fa",
        0,
        "human",
    ),
    "ecoli_positive": (
        "https://raw.githubusercontent.com/nmach22/Promoter-Classification/master/dataset/Ecoli_prom.fa",
        1,
        "ecoli",
    ),
    "ecoli_negative": (
        "https://raw.githubusercontent.com/nmach22/Promoter-Classification/master/dataset/Ecoli_non_prom.fa",
        0,
        "ecoli",
    ),
}


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float = 0.20
    val_fraction: float = 0.10
    seed: int = 42


@dataclass(frozen=True)
class HyperParamConfig:
    ngram_ranges: tuple[tuple[int, int], ...] = ((4, 7), (5, 6), (5, 8), (3, 8))
    alphas: tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    n_features: int = 2**20
    max_iter: int = 3000


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]


def canonical_signature(seq: str) -> str:
    rc = reverse_complement(seq)
    return seq if seq <= rc else rc


def _download_if_missing(url: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(dst))


def _stratified_group_split(
    signatures: np.ndarray,
    strata: np.ndarray,
    split_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_sig, sig_first_idx = np.unique(signatures, return_index=True)
    group_strata = strata[sig_first_idx]

    holdout_groups: set[str] = set()
    for stratum in np.unique(group_strata):
        idx = np.flatnonzero(group_strata == stratum)
        if idx.size == 0:
            continue
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_holdout = int(round(split_fraction * idx.size))
        if idx.size >= 3:
            n_holdout = min(max(n_holdout, 1), idx.size - 1)
        else:
            n_holdout = 1 if idx.size == 2 else 0
        selected = shuffled[:n_holdout]
        holdout_groups.update(unique_sig[selected].tolist())

    holdout_mask = np.array([sig in holdout_groups for sig in signatures], dtype=bool)
    holdout_idx = np.flatnonzero(holdout_mask)
    remain_idx = np.flatnonzero(~holdout_mask)
    return remain_idx, holdout_idx


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_human_ecoli_dataset(
    cache_dir: str | Path,
    force_download: bool = False,
    max_samples_per_species_class: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = Path(cache_dir)
    rng = np.random.default_rng(seed)

    seqs: list[str] = []
    labels: list[int] = []
    species: list[str] = []
    sources: list[str] = []

    for name, (url, label, specie) in DATASET_URLS.items():
        dst = cache_path / f"{name}.fa"
        _download_if_missing(url, dst, force=force_download)
        text = dst.read_text(encoding="utf-8")
        records = parse_fasta(text)
        current = [sequence for _, sequence in records]
        if max_samples_per_species_class is not None and len(current) > max_samples_per_species_class:
            idx = rng.choice(len(current), size=max_samples_per_species_class, replace=False)
            current = [current[i] for i in sorted(idx)]

        seqs.extend(current)
        labels.extend([label] * len(current))
        species.extend([specie] * len(current))
        sources.extend([name] * len(current))

    return (
        np.array(seqs, dtype=object),
        np.array(labels, dtype=np.int64),
        np.array(species, dtype=object),
        np.array(sources, dtype=object),
    )


def split_leakage_safe(
    sequences: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    config: SplitConfig,
) -> dict[str, np.ndarray | dict[str, int]]:
    signatures = np.array([canonical_signature(s) for s in sequences], dtype=object)
    unique_sig = np.unique(signatures)

    sig_label_sets: dict[str, set[int]] = {sig: set() for sig in unique_sig.tolist()}
    for sig, label in zip(signatures, labels):
        sig_label_sets[sig].add(int(label))

    keep_mask = np.array([len(sig_label_sets[sig]) == 1 for sig in signatures], dtype=bool)
    dropped_conflict = int(np.sum(~keep_mask))

    seq = sequences[keep_mask]
    y = labels[keep_mask]
    sp = species[keep_mask]
    sig = signatures[keep_mask]
    strata = np.array([f"{s}|{int(lbl)}" for s, lbl in zip(sp, y)], dtype=object)

    train_val_idx, test_idx = _stratified_group_split(sig, strata, config.test_fraction, config.seed)

    seq_tv = seq[train_val_idx]
    y_tv = y[train_val_idx]
    sp_tv = sp[train_val_idx]
    sig_tv = sig[train_val_idx]
    strata_tv = np.array([f"{s}|{int(lbl)}" for s, lbl in zip(sp_tv, y_tv)], dtype=object)

    val_fraction_within_trainval = config.val_fraction / max(1.0 - config.test_fraction, 1e-8)
    train_idx_rel, val_idx_rel = _stratified_group_split(sig_tv, strata_tv, val_fraction_within_trainval, config.seed + 1)

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "dropped_conflicting_sequences": {
            "count": dropped_conflict,
        },
    }


def _check_no_signature_overlap(sequences: np.ndarray, split: dict[str, np.ndarray | dict[str, int]]) -> None:
    sig = np.array([canonical_signature(s) for s in sequences], dtype=object)
    train = set(sig[split["train_idx"]].tolist())
    val = set(sig[split["val_idx"]].tolist())
    test = set(sig[split["test_idx"]].tolist())

    if train & val or train & test or val & test:
        raise ValueError("Leakage detected: duplicate/canonical-equivalent sequences overlap across splits")


def _train_once(
    train_seq: np.ndarray,
    train_y: np.ndarray,
    eval_seq: np.ndarray,
    eval_y: np.ndarray,
    ngram_range: tuple[int, int],
    alpha: float,
    n_features: int,
    max_iter: int,
    seed: int,
) -> tuple[dict[str, float | int], HashingVectorizer, SGDClassifier]:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        lowercase=False,
        norm="l2",
    )
    x_train = vectorizer.transform(train_seq.tolist())
    x_eval = vectorizer.transform(eval_seq.tolist())

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
    )
    model.fit(x_train, train_y)
    pred = model.predict(x_eval)
    return _metrics(eval_y, pred), vectorizer, model


def run_supervised_finetuning(
    sequences: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    split_cfg: SplitConfig | None = None,
    hyper_cfg: HyperParamConfig | None = None,
) -> dict:
    split_cfg = split_cfg or SplitConfig()
    hyper_cfg = hyper_cfg or HyperParamConfig()

    split = split_leakage_safe(sequences, labels, species, split_cfg)
    _check_no_signature_overlap(sequences, split)

    tr_idx = split["train_idx"]
    va_idx = split["val_idx"]
    te_idx = split["test_idx"]

    seq_train = sequences[tr_idx]
    y_train = labels[tr_idx]
    seq_val = sequences[va_idx]
    y_val = labels[va_idx]
    seq_test = sequences[te_idx]
    y_test = labels[te_idx]

    best: dict | None = None
    for ngram in hyper_cfg.ngram_ranges:
        for alpha in hyper_cfg.alphas:
            val_metrics, _, _ = _train_once(
                train_seq=seq_train,
                train_y=y_train,
                eval_seq=seq_val,
                eval_y=y_val,
                ngram_range=ngram,
                alpha=alpha,
                n_features=hyper_cfg.n_features,
                max_iter=hyper_cfg.max_iter,
                seed=split_cfg.seed,
            )
            score = (val_metrics["mcc"], val_metrics["f1"], val_metrics["precision"], val_metrics["recall"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "ngram_range": ngram,
                    "alpha": alpha,
                    "val_metrics": val_metrics,
                }

    if best is None:
        raise RuntimeError("No model could be trained")

    seq_train_full = np.concatenate([seq_train, seq_val])
    y_train_full = np.concatenate([y_train, y_val])
    test_metrics, _, _ = _train_once(
        train_seq=seq_train_full,
        train_y=y_train_full,
        eval_seq=seq_test,
        eval_y=y_test,
        ngram_range=best["ngram_range"],
        alpha=best["alpha"],
        n_features=hyper_cfg.n_features,
        max_iter=hyper_cfg.max_iter,
        seed=split_cfg.seed,
    )

    def _subset_counts(idx: np.ndarray) -> dict[str, int]:
        sub_sp = species[idx]
        sub_y = labels[idx]
        out: dict[str, int] = {}
        for sp_name in ("human", "ecoli"):
            for cls, cls_name in ((0, "negative"), (1, "positive")):
                out[f"{sp_name}_{cls_name}"] = int(np.sum((sub_sp == sp_name) & (sub_y == cls)))
        return out

    return {
        "dataset": {
            "total_sequences": int(len(sequences)),
            "train_sequences": int(len(tr_idx)),
            "val_sequences": int(len(va_idx)),
            "test_sequences": int(len(te_idx)),
            "dropped_conflicting_sequences": split["dropped_conflicting_sequences"]["count"],
            "train_breakdown": _subset_counts(tr_idx),
            "val_breakdown": _subset_counts(va_idx),
            "test_breakdown": _subset_counts(te_idx),
        },
        "split": {
            "test_fraction": split_cfg.test_fraction,
            "val_fraction": split_cfg.val_fraction,
            "seed": split_cfg.seed,
        },
        "best_hyperparameters": {
            "ngram_range": list(best["ngram_range"]),
            "alpha": float(best["alpha"]),
            "n_features": int(hyper_cfg.n_features),
            "max_iter": int(hyper_cfg.max_iter),
        },
        "validation_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
    }
