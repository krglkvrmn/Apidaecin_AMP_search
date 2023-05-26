import itertools
from typing import List

from Bio.Align import substitution_matrices
import numpy as np
import torch
from torch import nn


class OneHotEncoder:
    """
    Class performing one-hot-encoding

    :param alphabet: Which alphabet to use. Options are ['dna' | 'prot']
    :type alphabet: str
    :param device: Device where resulting tensors are stored. Options are ['cuda' | 'cpu']
    :type device: str
    """
    ALPHABETS = {
        "dna": "ATGC",
        "prot": "ACDEFGHIKLMNPQRSTVWYXBZJUO*"
    }

    def __init__(self, alphabet: str = "prot", device: str = "cpu"):
        self.device = device
        self.alphabet = self.ALPHABETS.get(alphabet)
        if not self.alphabet:
            self.alphabet = alphabet
        self.letter_index_map = dict(zip(self.alphabet, range(len(self.alphabet))))

    def get_batch_embedding(self, sequences: list) -> torch.FloatTensor:
        """
        One-hot-encode a batch of sequences and create single tensor
        :param sequences: Sequences of equal size, usually representing single batch
        :type sequences: List[str]
        :return: Tensor of encoded sequences
        :rtype: torch.LongTensor
        """
        num_classes = len(self.letter_index_map)
        proposed_len = len(sequences[0])
        sequences_combined = itertools.chain(*sequences)
        sequences_combined_aa_labels = [self.letter_index_map[aa] for aa in sequences_combined]
        one_hot_vector = nn.functional.one_hot(
            torch.LongTensor(sequences_combined_aa_labels), num_classes=num_classes
        ).view(len(sequences), -1, num_classes).float().to(self.device)
        return one_hot_vector


class SequenceAugmentator:
    """
    Applies substitution matrix based augmentations with specified parameters
    :param matrix_name: Substitution matrix name. Options are ['BLOSUMXX' | 'PAMXXX']
    :type matrix_name: str
    :param replacement_proba_factor: Factor that controls overall substitution frequency
    :type replacement_proba_factor: int
    """

    def __init__(self, matrix_name: str = "BLOSUM62", replacement_proba_factor: int = 1):
        self.replacement_proba_factor = replacement_proba_factor
        self.matrix = self._load_matrix(matrix_name)
        self.matrix_array = np.array(self.matrix)
        self.alphabet = np.array(list(self.matrix.alphabet))
        self.letter_index_map = dict(zip(self.matrix.alphabet, range(len(self.alphabet))))

    def __call__(self, sequence: str) -> str:
        return self.apply_augmentation(sequence)

    def apply_augmentation(self, sequence: str) -> str:
        """
        Applies augmentations to given sequence
        :param sequence: Sequence
        :type sequence: str
        :return: Sequence with or without substitutions
        :rtype: str
        """
        sequence = np.array(list(sequence))
        new_aa_vector = np.random.randint(0, len(self.alphabet), size=len(sequence))
        seq_aa_indices = [self.letter_index_map[aa] for aa in sequence]
        replacement_probas = self.matrix_array[seq_aa_indices, new_aa_vector] * self.replacement_proba_factor
        probabilities = np.random.rand(len(sequence))
        change_map = probabilities < replacement_probas
        sequence[change_map] = self.alphabet[new_aa_vector[change_map]]
        return "".join(sequence)

    @staticmethod
    def _load_matrix(matrix_name):
        matrix = substitution_matrices.load(matrix_name)
        matrix = substitution_matrices.Array(alphabet=matrix.alphabet,
                                             data=nn.Softmax(dim=1)(torch.from_numpy(matrix)).numpy())
        return matrix


def get_single_seq_patches(seq: str, patch_len: int = 20, stride: int = 1) -> List[str]:
    """
    Split sequence into fragments with overlaps
    :param seq: Sequence to split
    :type seq: str
    :param patch_len: Length of produced fragments
    :type patch_len: int
    :param stride: Step taken between two produced fragments
    :type stride: int
    :return: A list of fragments of equal length
    :rtype: List[str]
    """
    patches = []
    try:
        for idx in range(0, len(seq), stride):
            patch = seq[idx:idx + patch_len]
            if len(patch) < patch_len:
                modified_patch = seq[-patch_len:]
                if modified_patch != patches[-1]:
                    patches.append(modified_patch)
                break
            patches.append(patch)
    except IndexError:  # len(seq) < patch_len
        padding_size = (patch_len - len(seq)) // 2
        padded_seq = "X" * padding_size + seq + "X" * padding_size
        padded_seq += "X" * (len(padded_seq) == patch_len - 1)
        patches.append(padded_seq)
    return patches


def get_batch_seq_patches(sequence_batch: List[str], patch_len: int = 20, stride: int = 1) -> List[str]:
    """
    Split sequences in batch into fragments with overlaps
    :param sequence_batch: A list of sequences to split (batch)
    :type sequence_batch: List[str]
    :param patch_len: Length of produced fragments
    :type patch_len: int
    :param stride: Step taken between two produced fragments
    :type stride: int
    :return: A list of fragments of equal length
    :rtype: List[str]
    """
    patches = []
    for sequence in sequence_batch:
        patches += get_single_seq_patches(sequence, patch_len=patch_len, stride=stride)
    return patches
