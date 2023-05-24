from typing import Collection

import pandas as pd
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskBool
from src.core import Controller
from src.logs import logger
from src.utils import stringify_mask


def scan_single_sequence(controller: Controller, sequence: str, stride: int = 20) -> PredictionMaskBool:
    prediction_mask = controller.predict_mask(sequence, stride=stride)
    pos_count = prediction_mask.sum()
    if pos_count > 0 and stride > 1:
        # Refine region of interest
        prediction_mask = controller.predict_mask(sequence, stride=1)
    else:
        prediction_mask = [0] * len(sequence)
    return prediction_mask


def scan_records(controller: Controller, records: Collection[SeqRecord],
                 stride: int = 20, logging_interval: int = 1000) -> pd.DataFrame:
    predictions = []
    for idx, record in enumerate(records, start=1):
        sequence = str(record.seq)
        prediction_mask = scan_single_sequence(controller=controller, sequence=sequence, stride=stride)
        assert len(sequence) == len(prediction_mask), f"Lengths of the sequence and prediction do not match for record {record.id}"
        pos_count = sum(prediction_mask)
        if pos_count > 0:
            logger.log("PREDICTION", f"{record.id}: {pos_count}/{len(sequence)}")
        predictions.append((record.id, record.description, pos_count, sequence, stringify_mask(prediction_mask)))
        if logging_interval and idx % logging_interval == 0:
            logger.info(f"Processed {idx}/{len(records)} sequences")
    predictions_df = pd.DataFrame(
        predictions, columns=["record_id", "record_description", "pos_count", "sequence", "prediction_mask"]
    ).sort_values("pos_count", ascending=False)
    return predictions_df
