from dataclasses import dataclass

from torch import nn

from train_funcs.processing import OneHotEncoder
from train_funcs.metrics import (
    f1_score,
    precision_score,
    matthews_corrcoef,
    recall_score,
    accuracy_score
)


@dataclass
class ModelParameters:
    n_classes: int = 2
    embedding_size: int = 20
    conv_channels: int = 256
    conv_kernel_size: int = 5
    dropout_rate: float = 0.2

    blstm_input_size: int = 256
    blstm_output_size: int = 512
    lstm_output_size: int = 512

    activation: str = "relu"


@dataclass
class Hyperparameters:
    batch_size: int = 1000
    patch_size: int = 35
    patch_stride: int = 1
    substitution_matrix: str = "BLOSUM45"
    replacement_proba_factor: int = 250
    pos_proba: float = 0.1
    db_proba: float = 0.1

    model_parameters: ModelParameters = ModelParameters()
    encoder: OneHotEncoder = OneHotEncoder("prot", "cpu")

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: str = "adam"
    scheduler: nn.Module = None
    lr: float = 1e-4

    metric_fns: tuple = (matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score)
