"""Metrics and threshold selection.

PR-AUC is the primary metric because the fraud rate is ~0.58%: a model that
always answers "legit" already scores 99.4% accuracy, so accuracy measures
nothing here. PR-AUC focuses on the fraud class and has a known baseline (the
fraud rate), which is what makes it comparable across periods with different
fraud rates.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_f1_threshold(y_true, probabilities) -> tuple[float, float, float]:
    """Threshold on the PR curve that maximises F1, plus precision and recall."""
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probabilities)
    f1_curve = (2 * precision_curve[:-1] * recall_curve[:-1] /
                (precision_curve[:-1] + recall_curve[:-1] + 1e-12))
    best = int(f1_curve.argmax())
    return float(thresholds[best]), float(precision_curve[best]), float(recall_curve[best])


def threshold_for_recall(y_true, probabilities, min_recall: float) -> float:
    """Smallest threshold (highest alert rate) that still reaches min_recall.

    Fraud review queues are capacity-limited: catching fraud matters more than
    flagging every possible case, so recall is often a hard constraint rather
    than something to optimise freely.
    """
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probabilities)
    ok = np.where(recall_curve[:-1] >= min_recall)[0]
    if len(ok) == 0:
        return float(thresholds[0])
    return float(thresholds[ok[-1]])


def evaluate(name: str, y_true, probabilities, threshold: float | None = None,
             seconds: float = 0.0) -> dict:
    """Full metric row for one model on one dataset."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    pr_auc = float(average_precision_score(y_true, probabilities))
    roc_auc = float(roc_auc_score(y_true, probabilities))

    if threshold is None:
        threshold, precision, recall = best_f1_threshold(y_true, probabilities)
    else:
        predicted = (probabilities >= threshold).astype(int)
        precision = float(precision_score(y_true, predicted, zero_division=0))
        recall = float(recall_score(y_true, predicted, zero_division=0))

    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return {
        "model": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged": int(predicted.sum()),
        "seconds": round(float(seconds), 1),
    }


def lift_over_random(y_true, probabilities) -> float:
    """How much better than guessing, expressed as precision uplift."""
    return float(average_precision_score(y_true, probabilities) / np.mean(y_true))
