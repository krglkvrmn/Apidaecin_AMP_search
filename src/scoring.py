__all__ = ["length_score", "consecutive_score", "fraction_score", "calculate_score_threshold"]

from functools import wraps

import pandas as pd
from Bio.Seq import Seq
from sklearn.cluster import KMeans

from src._typing import PredictionMaskAny
from src.utils import numerize_mask


def scoring_preprocessing(scoring_func):
    @wraps(scoring_func)
    def wrapper(sequence: Seq | str, mask: PredictionMaskAny):
        mask = numerize_mask(mask)
        if len(sequence) != len(mask):
            raise ValueError(f"Length of sequence ({len(sequence)}) does not match length of prediction mask ({len(mask)})")
        return scoring_func(sequence=sequence, mask=mask)
    return wrapper


@scoring_preprocessing
def length_score(sequence: str, mask: PredictionMaskAny) -> float:
    return mask.sum()


@scoring_preprocessing
def fraction_score(sequence: str, mask: PredictionMaskAny) -> float:
    return mask.sum() / len(mask)


@scoring_preprocessing
def consecutive_score(sequence: str, mask: PredictionMaskAny) -> float:
    score = 0.0
    for idx, symb_pred in enumerate(mask[1:-1], start=1):
        symbol_score = symb_pred * (mask[idx - 1] + mask[idx + 1])
        score += symbol_score
    return score


def calculate_score_threshold(scores: pd.Series) -> float:
    if scores.empty:
        return 0.0
    clustering = KMeans(n_clusters=2)
    clustering.fit(pd.DataFrame(scores))
    cluster_border = clustering.cluster_centers_.mean()
    return cluster_border
