"""Metrics describe domain relevance, never establish doctrinal truth or safety."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)


def classification(y, probabilities, threshold=0.5):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected nonempty aligned labels/probabilities in [0,1]")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "false_accept_rate": float(fp / (fp + tn)) if fp + tn else None,
            "false_reject_rate": float(fn / (fn + tp)) if fn + tp else None,
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
            "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def group_bootstrap_accuracy(y, p, groups, threshold=0.5, seed=42, repeats=2000):
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == g) for g in unique]
    if len(unique) < 2:
        return None
    values = []
    for _ in range(repeats):
        idx = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
        values.append(float(np.mean((p[idx] >= threshold) == y[idx])))
    return {"method": "percentile group bootstrap", "repeats": repeats, "groups": len(unique),
            "low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975))}


def threshold_from_validation(y, p, min_precision=None):
    if min_precision is None:
        return 0.5
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0,1]")
    y, p = np.asarray(y), np.asarray(p)
    feasible = []
    for t in sorted(set([0.0, 0.5, 1.0] + p.tolist())):
        pred = p >= t
        if pred.any() and precision_score(y, pred, zero_division=0) >= min_precision:
            feasible.append((float(recall_score(y, pred, zero_division=0)), float(t)))
    # No observed feasible operating point: reject all, and make this explicit.
    return max(feasible)[1] if feasible else float(np.nextafter(1.0, 2.0))
