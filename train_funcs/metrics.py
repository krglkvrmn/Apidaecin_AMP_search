from functools import partial

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef
)

__all__ = ["f1_score", "precision_score", "recall_score", "accuracy_score", "matthews_corrcoef"]

precision_score = partial(precision_score, zero_division=0)
recall_score = partial(recall_score, zero_division=0)
f1_score = partial(f1_score, zero_division=0)

precision_score.__name__ = "precision_score"
recall_score.__name__ = "recall_score"
f1_score.__name__ = "f1_score"
