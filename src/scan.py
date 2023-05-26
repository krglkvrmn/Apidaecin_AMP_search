from typing import Collection, Sequence

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskBool
from src.core import Controller
from src.logs import logger
from src.utils import encode_mask


def scan_single_sequence(controller: Controller, sequence: str, stride: int = 20) -> PredictionMaskBool:
    prediction_mask = controller.predict_sequence(sequence, stride=stride, as_numpy=False)
    if prediction_mask.any() and stride > 1:
        # Refine region of interest
        return controller.predict_sequence(sequence, stride=1, as_numpy=True)
    else:
        return PredictionMaskBool(np.zeros(len(sequence), dtype=np.bool8))


def scan_records(controller: Controller, records: Collection[SeqRecord], stride: int = 20,
                 save_predictions_without_hits: bool = False, logging_interval: int = 1000) -> pd.DataFrame:
    predictions = []
    for idx, record in enumerate(records, start=1):
        sequence = str(record.seq)
        prediction_mask = scan_single_sequence(controller=controller, sequence=sequence, stride=stride)
        assert len(sequence) == len(prediction_mask), f"Lengths of the sequence and prediction do not match for record {record.id}"
        if prediction_mask.any():
            pos_count = prediction_mask.sum()
            logger.log("PREDICTION", f"{record.id}: {pos_count}/{len(sequence)}")
            predictions.append((record.id, record.description, pos_count, sequence, encode_mask(prediction_mask)))
        elif save_predictions_without_hits:
            predictions.append((record.id, record.description, 0, sequence, encode_mask(prediction_mask)))

        if logging_interval and idx % logging_interval == 0:
            logger.info(f"Processed {idx}/{len(records)} sequences")
    predictions_df = pd.DataFrame(
        predictions, columns=["record_id", "record_description", "pos_count", "sequence", "prediction_mask"]
    ).sort_values("pos_count", ascending=False)
    return predictions_df
