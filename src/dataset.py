import random
import warnings
from typing import Callable, List

from loguru import logger

from src.processing import get_single_seq_patches


warnings.filterwarnings("ignore")


class SequencePatchDataset:
    """
    Dataset for training models, that are dealing with patches
    :param sequences: Sequences to produce dataset from
    :type sequences: List[str]
    :param labels: Class labels of given sequences. Must have the same length
    :type labels: List[int]
    :param patch_len: Length fragments produces from sequences
    :type patch_len: int
    :param stride: Step taken between produced fragments
    :type stride: int
    :param pos_proba: Fraction of positive class objects in batch
    :type pos_proba: float
    :param db_proba: Fraction of database peptides in batch
    :type db_proba: float
    :param enable_db_labels: Whether to yield database peptides as separate class
    :type enable_db_labels: bool
    :param augmentator: Object that performs augmentation on single sequence
    :type augmentator: Callable
    """

    def __init__(self, sequences: List[str], labels: List[int], patch_len: int = 10, stride: int = 1,
                 pos_proba: float = 0.1, db_proba: float = 0.1, enable_db_labels: bool = False,
                 augmentator: Callable = lambda x: x):
        self.augmentator = augmentator
        self.enable_db_labels = enable_db_labels
        self.data = []
        self.labels = []
        for seq, lab in zip(sequences, labels):
            if lab in {1, 2}:
                patches = get_single_seq_patches(seq, patch_len=patch_len, stride=stride)
            else:
                patches = random.choices(get_single_seq_patches(seq, patch_len=patch_len, stride=stride), k=2)
            labels = [lab] * len(patches)
            self.data += patches
            self.labels += labels
        self.n_pos = sum([label == 1 for label in self.labels])
        self.n_dbneg = sum([label == 2 for label in self.labels])
        self.n_neg = len(self.labels) - self.n_pos - self.n_dbneg
        pos_part = self.n_pos / len(self.labels)
        dbneg_part = self.n_dbneg / len(self.labels)

        self.pos_weight = pos_proba / pos_part
        self.db_weight = db_proba / dbneg_part
        if self.db_weight == float("inf"):
            self.db_weight = 0
        self.neg_weight = (1 - pos_proba - db_proba) / (self.n_neg / len(self.labels))
        label_weight_mapper = {0: self.neg_weight, 1: self.pos_weight, 2: self.db_weight}
        self.weights = [label_weight_mapper[lab] for lab in self.labels]

        logger.debug(f"Created dataset. pos_proba={pos_proba}, db_proba={db_proba}, len_data={len(self.data)}, len_labels={len(self.labels)}, pos_count={self.n_pos}, db_count={self.n_dbneg}, neg_count={self.n_neg}, pos_weight={self.pos_weight}, db_weight={self.db_weight}, neg_weight={self.neg_weight}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch, label = self.data[idx], self.labels[idx]
        if label == 1:
            patch = self.augmentator(patch)
        elif label == 2:
            patch = self.augmentator(patch)
            if not self.enable_db_labels:
                label = 0
        return patch, label
