from typing import Collection, Sequence, Iterable

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskBool
from src.core import Controller
from src.logs import logger
from src.utils import encode_mask


class ScanScheduler:
    def __init__(self, controller: Controller, patches_limit: int = 0, scan_stride: int = 1, n_jobs: int = 1):
        self.controller = controller
        self.n_jobs = n_jobs
        self.scan_stride = scan_stride
        self.size = 0
        self.limit = patches_limit
        self.scheduled_records = {}
        self.successful_records = {}
        self.predictions = []

    def __schedule_record_scan(self, record: SeqRecord, stride: int):
        self.size += (len(record) + self.controller.hp.patch_size - self.controller.hp.patch_size % 2) / stride
        self.scheduled_records[record.id] = record

    def schedule_sparse_scan(self, record: SeqRecord):
        self.__schedule_record_scan(record, stride=self.scan_stride)

    def schedule_refine_scan(self, record: SeqRecord):
        self.__schedule_record_scan(record, stride=1)

    def sparse_scan(self):
        predictions = self.controller.predict_sequences([str(record.seq) for record in self.scheduled_records.values()],
                                                        stride=self.scan_stride, n_jobs=self.n_jobs, limit=self.limit, as_numpy=False)
        for prediction, record in zip(predictions, self.scheduled_records.values()):
            if prediction.any():
                logger.log("PREDICTION", f"{record.id}: Hit detected")
                self.successful_records[record.id] = record
        self.scheduled_records.clear()
        self.size = 0

    def refine_scan(self):
        predictions = self.controller.predict_sequences(
            [str(record.seq) for record in self.scheduled_records.values()],
            stride=1, n_jobs=self.n_jobs, as_numpy=True, limit=self.limit
        )
        for prediction, record in zip(predictions, self.scheduled_records.values()):
            pos_count = prediction.sum()
            logger.log("PREDICTION", f"{record.id}: {pos_count}/{len(record)}")
            self.predictions.append(
                (record.id, record.description, pos_count, str(record.seq), encode_mask(prediction))
            )
        self.scheduled_records.clear()
        self.size = 0

    def __can_sparse_schedule(self, record: SeqRecord) -> bool:
        extra_length = self.controller.hp.patch_size - self.controller.hp.patch_size % 2
        return self.size + (len(record) + extra_length) / self.scan_stride > self.limit

    def __can_refine_schedule(self, record: SeqRecord) -> bool:
        extra_length = self.controller.hp.patch_size - self.controller.hp.patch_size % 2
        return self.size + len(record) + extra_length > self.limit

    def run(self, records: Sequence[SeqRecord], logging_interval: int = 1000):
        for idx, record in enumerate(records, start=1):
            if self.__can_sparse_schedule(record):
                self.sparse_scan()
            self.schedule_sparse_scan(record)
            if logging_interval and idx % logging_interval == 0:
                logger.info(f"Processed {idx}/{len(records)} sequences")
        if len(self.scheduled_records) > 0:
            self.sparse_scan()

        logger.info(f"Refining {len(self.successful_records)} sequences")
        for record in self.successful_records.values():
            if self.__can_refine_schedule(record):
                self.refine_scan()
            self.schedule_refine_scan(record)
        if len(self.scheduled_records) > 0:
            self.refine_scan()
        return pd.DataFrame(
            self.predictions, columns=["record_id", "record_description", "pos_count", "sequence", "prediction_mask"]
        ).sort_values("pos_count", ascending=False)


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
