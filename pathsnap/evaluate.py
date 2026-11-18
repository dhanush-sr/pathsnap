"""Shared metrics for candidate classifiers."""
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score


def top1_accuracy(df, proba):
    """Observation-level Top-1: the highest-scoring candidate must be the reference arc.
    Observations whose reference arc is not among the candidates count as errors."""
    d = df[["trip", "obs", "label"]].assign(p=proba)
    best = d.sort_values("p", ascending=False).drop_duplicates(["trip", "obs"])
    return float(best["label"].mean())


def candidate_metrics(df, proba, thr=0.5):
    y = df["label"].values
    yhat = (proba >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, average="macro", zero_division=0),
        "recall": recall_score(y, yhat, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, yhat),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "top1_accuracy": top1_accuracy(df, proba),
    }
