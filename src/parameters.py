import configparser
from dataclasses import dataclass
from typing import Union, Dict

from torch import nn

from src.processing import OneHotEncoder
from src.metrics import (
    f1_score,
    precision_score,
    matthews_corrcoef,
    recall_score,
    accuracy_score
)


def parse_config_as_dict(path: str) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
    Parse parameters config located in path
    :param path: Path with config
    :type path: str
    :return: Dictionary containing 3 dictionaries: model specific parameters, training parameters, general parameters
    :rtype: Dict[str, Dict[str, Union[str, int, float]]]
    """
    config = configparser.ConfigParser()
    config.read(path)

    model_parameters = dict(
        n_classes=int(config.get("MODEL_PARAMETERS", "n_classes")),
        embedding_size=int(config.get("MODEL_PARAMETERS", "embedding_size")),
        conv_channels=int(config.get("MODEL_PARAMETERS", "conv_channels")),
        dropout_rate=float(config.get("MODEL_PARAMETERS", "dropout_rate")),
        blstm_output_size=int(config.get("MODEL_PARAMETERS", "blstm_output_size")),
        lstm_output_size=int(config.get("MODEL_PARAMETERS", "lstm_output_size")),
        activation=config.get("MODEL_PARAMETERS", "activation")
    )
    training_parameters = dict(
        batch_size=int(config.get("TRAINING", "batch_size")),
        patch_size=int(config.get("TRAINING", "patch_size")),
        patch_stride=int(config.get("TRAINING", "patch_stride")),
        substitution_matrix=config.get("TRAINING", "substitution_matrix"),
        replacement_proba_factor=int(config.get("TRAINING", "replacement_proba_factor")),
        pos_proba=float(config.get("TRAINING", "pos_proba")),
        db_proba=float(config.get("TRAINING", "db_proba")),
        optimizer=config.get("TRAINING", "optimizer"),
        lr=float(config.get("TRAINING", "lr"))
    )
    general_parameters = dict(
        device=config.get("GENERAL", "device")
    )
    parameters = dict(
        mp=model_parameters,
        hp=training_parameters,
        gp=general_parameters
    )
    return parameters


@dataclass
class ModelParameters:
    n_classes: int = 2
    embedding_size: int = 20
    conv_channels: int = 256
    conv_kernel_size: int = 5
    dropout_rate: float = 0.2

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
