import configparser
from dataclasses import dataclass
from typing import Union, Dict, Literal

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
        device=config.get("GENERAL", "device"),
        batch_size=int(config.get("TRAINING", "batch_size")),
        patch_size=int(config.get("TRAINING", "patch_size")),
        patch_stride=int(config.get("TRAINING", "patch_stride")),
        substitution_matrix=config.get("TRAINING", "substitution_matrix"),
        replacement_proba_factor=int(config.get("TRAINING", "replacement_proba_factor")),
        pos_proba=float(config.get("TRAINING", "pos_proba")),
        antipos_proba=float(config.get("TRAINING", "antipos_proba")),
        optimizer=config.get("TRAINING", "optimizer"),
        use_scheduler=config.get("TRAINING", "use_scheduler").lower() == "true",
        scheduler_type=config.get("TRAINING", "scheduler_type"),
        scheduler_factor=float(config.get("TRAINING", "scheduler_factor")),
        scheduler_interval=float(config.get("TRAINING", "scheduler_interval")),
        scheduler_patience=int(config.get("TRAINING", "scheduler_patience")),
        lr=float(config.get("TRAINING", "lr"))
    )
    parameters = dict(
        mp=model_parameters,
        hp=training_parameters,
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
    device: str = "cpu"
    batch_size: int = 1000
    patch_size: int = 35
    patch_stride: int = 1
    substitution_matrix: str = "BLOSUM45"
    replacement_proba_factor: int = 250
    pos_proba: float = 0.1
    antipos_proba: float = 0.1

    model_parameters: ModelParameters = ModelParameters()
    encoder: OneHotEncoder = OneHotEncoder("prot", "cpu")

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: str = "adam"
    use_scheduler: bool = False
    scheduler_type: Literal["reduce_on_plateau", "step"] = "reduce_on_plateau"
    scheduler_interval: int = 10
    scheduler_factor: float = 0.1
    scheduler_patience: int = 10
    lr: float = 1e-4

    metric_fns: tuple = (matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score)
