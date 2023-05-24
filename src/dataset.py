import random
import warnings
from typing import Callable, Sequence

import torch.utils.data

from src.logs import logger
from src.processing import get_single_seq_patches

warnings.filterwarnings("ignore")


class SequencePatchDataset(torch.utils.data.Dataset):
    """
    Dataset for training models, that are dealing with patches
    :param sequences: Sequences to produce dataset from
    :type sequences: Sequence[str]
    :param labels: Class labels of given sequences. Must have the same length
    :type labels: Sequence[int]
    :param patch_len: Length fragments produces from sequences
    :type patch_len: int
    :param stride: Step taken between produced fragments
    :type stride: int
    :param pos_proba: Fraction of positive class objects in batch
    :type pos_proba: float
    :param antipos_proba: Fraction of anti-positive class
    :type antipos_proba: float
    :param enable_antipos_labels: Whether to yield antipos peptides as a separate class
    :type enable_antipos_labels: bool
    :param augmentator: Object that performs augmentation on single sequence
    :type augmentator: Callable
    """

    def __init__(self, sequences: Sequence[str], labels: Sequence[int], patch_len: int = 10, stride: int = 1,
                 pos_proba: float = 0.1, antipos_proba: float = 0.1, enable_antipos_labels: bool = False,
                 augmentator: Callable = lambda x: x):
        self.augmentator = augmentator
        self.enable_antipos_labels = enable_antipos_labels
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
        self.n_antipos = sum([label == 2 for label in self.labels])
        self.n_neg = len(self.labels) - self.n_pos - self.n_antipos
        pos_part = self.n_pos / len(self.labels)
        antipos_part = self.n_antipos / len(self.labels)

        self.pos_weight = pos_proba / pos_part
        self.antipos_weight = antipos_proba / antipos_part
        if self.antipos_weight == float("inf"):
            self.antipos_weight = 0
        self.neg_weight = (1 - pos_proba - antipos_proba) / (self.n_neg / len(self.labels))
        label_weight_mapper = {0: self.neg_weight, 1: self.pos_weight, 2: self.antipos_weight}
        self.weights = [label_weight_mapper[lab] for lab in self.labels]

        logger.debug(f"Created dataset. pos_proba={pos_proba}, antipos_proba={antipos_proba}, "
                     f"len_data={len(self.data)}, len_labels={len(self.labels)}, pos_count={self.n_pos}, "
                     f"antipos_count={self.n_antipos}, neg_count={self.n_neg}, pos_weight={self.pos_weight}, "
                     f"antipos_weight={self.antipos_weight}, neg_weight={self.neg_weight}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch, label = self.data[idx], self.labels[idx]
        if label == 1:
            patch = self.augmentator(patch)
        elif label == 2:
            patch = self.augmentator(patch)
            if not self.enable_antipos_labels:
                label = 0
        return patch, label
