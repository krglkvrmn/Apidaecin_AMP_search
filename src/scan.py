import math
from typing import Collection, Sequence, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskBool
from src.core import Controller
from src.logs import logger
from src.utils import encode_mask


class RecordsScanner:
    def __init__(self, controller: Controller, batch_size_limit: Optional[int] = None, scan_stride: int = 1):
        self.__controller = controller
        self.__scan_stride = scan_stride
        self.__current_batch_size = 0
        self.__batch_size_limit = batch_size_limit
        self._scheduled_records = {}
        self._successful_records = {}
        self._predictions = []

    @property
    def _batch_extra_length(self):
        return self.__controller.hp.patch_size - self.__controller.hp.patch_size % 2

    def _sparse_scan(self):
        predictions = self.__controller.predict_sequences(
            [str(record.seq) for record in self._scheduled_records.values()], stride=self.__scan_stride,
            batch_size=self.__batch_size_limit, as_numpy=False)
        for prediction, record in zip(predictions, self._scheduled_records.values()):
            if prediction.any():
                logger.log("PREDICTION", f"{record.id}: Hit detected")
                self._successful_records[record.id] = record
        self._clear_schedule()

    def _refine_scan(self):
        predictions = self.__controller.predict_sequences(
            [str(record.seq) for record in self._scheduled_records.values()], stride=1,
            batch_size=self.__batch_size_limit, as_numpy=True)
        for prediction, record in zip(predictions, self._scheduled_records.values()):
            pos_count = prediction.sum()
            logger.log("PREDICTION", f"{record.id}: {pos_count}/{len(record)}")
            self._predictions.append(
                (record.id, record.description, pos_count, str(record.seq), encode_mask(prediction))
            )
        self._clear_schedule()

    def _clear_schedule(self):
        self._scheduled_records.clear()
        self.__current_batch_size = 0

    def _schedule_record_scan(self, record: SeqRecord, stride: int):
        self.__current_batch_size += (len(record) + self._batch_extra_length) / stride
        self._scheduled_records[record.id] = record

    def __can_schedule(self, record: SeqRecord, scan_type: Literal["sparse", "refine"]) -> bool:
        match scan_type:
            case "sparse": stride = self.__scan_stride
            case "refine": stride = 1
            case _: raise ValueError(f"Invalid scan_type: {scan_type}")

        new_batch_size = self.__current_batch_size + (len(record) + self._batch_extra_length) / stride
        limit_exceeded = new_batch_size > self.__batch_size_limit
        if limit_exceeded and len(self._scheduled_records) == 0:
            raise ValueError(f"Cannot fit record {record.id} with {math.floor(new_batch_size)} patches "
                             f"to batch size {self.__batch_size_limit}")
        return not limit_exceeded

    def scan_scheduled_records(self, scan_type: Literal["sparse", "refine"]):
        match scan_type:
            case "sparse": self._sparse_scan()
            case "refine": self._refine_scan()
            case _: raise ValueError(f"Invalid scan_type: {scan_type}")

    def schedule(self, record: SeqRecord, scan_type: Literal["sparse", "refine"]):
        match scan_type:
            case "sparse": self._schedule_record_scan(record, stride=self.__scan_stride)
            case "refine": self._schedule_record_scan(record, stride=1)
            case _: raise ValueError(f"Invalid scan_type: {scan_type}")

    def _scan_records(self, records: Sequence[SeqRecord],
                      scan_type: Literal["sparse", "refine"],
                      logging_interval: int):
        for idx, record in enumerate(records, start=1):
            if not self.__can_schedule(record, scan_type=scan_type):
                self.scan_scheduled_records(scan_type=scan_type)
            self.schedule(record, scan_type=scan_type)
            if logging_interval and idx % logging_interval == 0:
                logger.info(f"Processed {idx}/{len(records)} sequences")
        if len(self._scheduled_records) > 0:
            self.scan_scheduled_records(scan_type=scan_type)

    def run(self, records: Sequence[SeqRecord], logging_interval: int = 1000) -> pd.DataFrame:
        if self.__batch_size_limit is None:
            self.__batch_size_limit = len(max(records, key=len)) + self._batch_extra_length

        logger.info(f"Running sparse scanning of {len(records)} sequences "
                    f"with batch_size={self.__batch_size_limit} and stride={self.__scan_stride}")
        self._scan_records(records, scan_type="sparse", logging_interval=logging_interval)

        logger.info(f"Refining {len(self._successful_records)} sequences containing hits "
                    f"with batch_size={self.__batch_size_limit}")
        self._scan_records(list(self._successful_records.values()), scan_type="refine", logging_interval=0)

        return pd.DataFrame(
            self._predictions, columns=["record_id", "record_description", "pos_count", "sequence", "prediction_mask"]
        ).sort_values("pos_count", ascending=False)
