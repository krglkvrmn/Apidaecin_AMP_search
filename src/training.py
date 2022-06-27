import os.path
import pickle
import re
from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import SequencePatchDataset
from src.models import SimpleCNN, HybridModel
from src.parameters import Hyperparameters, ModelParameters
from src.processing import SequenceAugmentator, get_single_seq_patches


class Trainer:
    """
    Class controls training, logging and caching results
    :param model_class: Class of model to set up and train
    :type model_class: torch.nn.Module
    :param X_train: Sequences to form train dataset
    :type X_train: Optional[List[str]]
    :param X_val: Sequences to form validation dataset
    :type X_val: Optional[List[str]]
    :param y_train: Labels to form train dataset
    :type y_train: Optional[List[int]]
    :param y_val: Labels to form validation dataset
    :type y_val: Optional[List[str]]
    :param hyperparameters: Object containing adjustable parameters as attributes
    :type hyperparameters: Hyperparameters
    :param setup: Whether to create datasets, data loaders, model and set optimizer on initialization
    :type setup: bool
    :param device: Device used for training. Options are ['cpu' | 'cuda']
    :type device: str
    """
    _activations_map = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leakyrelu": nn.LeakyReLU()
    }

    _optimizers_map = {
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
    }

    def __init__(self, model_class: torch.nn.Module,
                 X_train: Optional[List[str]], X_val: Optional[List[str]],
                 y_train: Optional[List[int]], y_val: Optional[List[str]],
                 hyperparameters: Hyperparameters, setup: bool = False, device: str = "cpu"):
        self.model = None
        self.optimizer = None
        self.device = device
        self.model_class = model_class
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.hp = hyperparameters

        if setup:
            self.setup()

        self.criterion = self.hp.criterion
        self.scheduler = self.hp.scheduler
        self.encoder = self.hp.encoder
        self.augmentator = SequenceAugmentator(matrix_name=self.hp.substitution_matrix,
                                               replacement_proba_factor=self.hp.replacement_proba_factor)

        self.epoch = 1

        self.embedding_cache = {}

    def setup(self):
        """
        Create datasets, data loaders, model and set optimizer
        """
        self._make_datasets()
        self._make_dataloaders()
        self.create_model()
        self.set_optimizer()

    def reset(self):
        """
        Reset model state
        """
        delattr(self, "augmentator")
        delattr(self, "train_dataset")
        delattr(self, "test_dataset")
        delattr(self, "train_loader")
        delattr(self, "val_loader")
        delattr(self, "model")
        delattr(self, "optimizer")
        self.epoch = 0

    def _make_datasets(self, enable_db_labels: bool = False):
        """
        Create train dataset and validation dataset if data was provided
        :param enable_db_labels: Whether to yield database peptides as separate class
        :type enable_db_labels: bool
        """
        self.augmentator = SequenceAugmentator(self.hp.substitution_matrix,
                                               replacement_proba_factor=self.hp.replacement_proba_factor)
        self.train_dataset = SequencePatchDataset(self.X_train, self.y_train,
                                                  patch_len=self.hp.patch_size,
                                                  stride=self.hp.patch_stride,
                                                  pos_proba=self.hp.pos_proba,
                                                  db_proba=self.hp.db_proba,
                                                  augmentator=self.augmentator,
                                                  enable_db_labels=enable_db_labels)
        if self.X_val is not None:
            self.test_dataset = SequencePatchDataset(self.X_val, self.y_val, patch_len=self.hp.patch_size)

    def _make_dataloaders(self):
        """
        Create train dataloader and validation dataloader if data was provided
        """
        train_sampler = WeightedRandomSampler(self.train_dataset.weights, sum(self.train_dataset.labels).item())

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.hp.batch_size, sampler=train_sampler)
        if self.X_val is not None:
            self.val_loader = DataLoader(self.test_dataset, batch_size=self.hp.batch_size)

    def create_model(self, model_parameters: ModelParameters = None) -> torch.nn.Module:
        """
        Create model object with specified adjustable parameters
        :param model_parameters: Object containing adjustable parameters for model as attributes
        :type model_parameters: ModelParameters
        :return: Model object
        :rtype: torch.nn.Module
        """
        if model_parameters is None:
            model_parameters = self.hp.model_parameters

        mp_copy = deepcopy(model_parameters)
        mp_copy.activation = self._activations_map[model_parameters.activation]
        if self.model_class is SimpleCNN:
            self.model = self.model_class(patch_size=self.hp.patch_size, **vars(mp_copy)).to(self.device)
        else:
            self.model = self.model_class(**vars(mp_copy)).to(self.device)
        return self.model

    def set_optimizer(self):
        """
        Set optimizer with predefined parameters
        """
        self.optimizer = self._optimizers_map[self.hp.optimizer](self.model.parameters(), lr=self.hp.lr)

    def train_epoch(self) -> float:
        """
        Train model on single epoch

        :return: Average loss value of all batches
        :rtype: float
        """
        losses = []
        exc_count = 0
        self.model.train()
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            try:
                self.optimizer.zero_grad()
                data = self.encoder.get_batch_embedding(data)
                labels = labels.to(self.device)

                preds = self.model(data)
                loss = self.criterion(preds, labels)
                loss.backward()

                self.optimizer.step()

                losses.append(loss.item())
            except Exception as exc:
                logger.error(f"Training on batch {batch_idx} failed ({exc}).")
                logger.info(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Predictions shape: {preds.shape}")
                exc_count += 1
        self.epoch += 1
        return sum(losses) / (len(losses) - exc_count)

    def train(self, n_epochs: int = 10, writer: Optional[SummaryWriter] = None, valid: Union[bool, int] = True,
              vis_seqs: Optional[List[str]] = None, cache_embeddings: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Train model on specified number of epochs and log results
        :param n_epochs: Number of epochs to train model on
        :type n_epochs: int
        :param writer: Tensorboard summary writer. Disable tensorboard logging if None
        :type writer: Optional[SummaryWriter]
        :param valid: Perform validation every specified number of epochs. Disable validation if False
        E.g. valid = 2 means validation occurs every 2 epochs
        :type valid: Union[bool, int]
        :param vis_seqs: Sequences for prediction visualization in Tensorboard. Could be of any length. Disable visualization if None
        :type vis_seqs: Optional[List[str]]
        :param cache_embeddings: Store latent data representation generated by model on validation for every validation epoch
        :type cache_embeddings: bool
        :return: Average loss value for validation dataset and dict that maps metric function name to obtained on validation set itself
        :rtype: Tuple[float, Dict[str, float]]
        """
        logger.debug(f"Training started. {self.hp}")
        start_time = datetime.now()
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch()

            val_status = valid and epoch % valid == 0
            if val_status:
                if valid >= 5:
                    logger.info(f"Epoch: {epoch}\tRunning validation...")
                val_loss, val_metrics = self.validate(cache_embeddings=cache_embeddings)

            if self.scheduler and val_status:
                self.scheduler.step(val_loss)

            logger_base_string = f"Epoch: {epoch}\tTrain loss: {train_loss:.6f}"
            logger_val_string = f"Validation loss: {val_loss:.6f}" if val_status else ""
            logger_metrics_string = "\t".join([f"{metric}: {value:.4f}"
                                               for metric, value in val_metrics.items()]) if val_status else ""
            logger_complete_string = "\t".join([logger_base_string,
                                                logger_val_string,
                                                logger_metrics_string])

            logger.info(logger_complete_string)

            if writer:
                scalars_dict_preset = {"train": train_loss}
                if val_status:
                    scalars_dict_preset.update({"validation": val_loss})
                    writer.add_scalars("Metric", val_metrics, epoch)
                writer.add_scalars("Loss", scalars_dict_preset, epoch)

                if vis_seqs is not None:
                    self.model.eval()
                    for seq_id, sequence in enumerate(vis_seqs):
                        mask = self.predict_mask(sequence)
                        repr_ = sequence + "\n" + "".join([str(aa_mask) for aa_mask in mask])
                        writer.add_text(f"Sequence {seq_id + 1}", repr_, epoch)
                writer.flush()
        total_time = datetime.now() - start_time
        logger.info(f"Training finished in {total_time}")
        if val_status:
            return val_loss, val_metrics

    def validate(self, cache_embeddings: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Validate model using validation dataset
        :param cache_embeddings: Whether to save latent data representation generated by model on validation
        :type cache_embeddings: bool
        :return: Average loss value for validation dataset and dict that maps metric function name to obtained on validation set itself
        Example {'precision_score': 0.98, 'f1_score': 0.9}
        :rtype: Tuple[float, Dict[str, float]]
        """
        self.model.eval()
        with torch.no_grad():
            metrics = {}
            predictions = []
            labels = []
            if cache_embeddings:
                all_embeddings = []
            for batch_idx, (data, labs) in enumerate(self.val_loader):
                data = self.encoder.get_batch_embedding(data)
                labs = labs.to(self.device)
                if cache_embeddings:
                    preds, embedding = self.model.forward(data, return_embedding=True)
                    all_embeddings.append(embedding)
                else:
                    preds = self.model(data)
                predictions.append(preds.cpu())
                labels.append(labs.cpu())

            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            loss = self.criterion(predictions, labels).item()

            predictions = np.argmax(predictions.numpy(), axis=1)
            labels = labels.numpy()

            if cache_embeddings:
                self.embedding_cache[self.epoch] = (torch.cat(all_embeddings).numpy(), labels)

            for metric_fn in self.hp.metric_fns:
                metrics[metric_fn.__name__] = metric_fn(labels, predictions)

            return loss, metrics

    def predict(self, batch: List[str]) -> torch.LongTensor:
        """
        Predict classes of given sequences
        :param batch: A list of sequences of equal length
        :type batch: List[str]
        :return: Tensor with predicted labels for each sequence
        :rtype: torch.LongTensor
        """
        self.model.eval()
        with torch.no_grad():
            batch = self.encoder.get_batch_embedding(batch)
            predictions = self.model(batch)
            predictions = predictions.argmax(dim=1)
        return predictions

    def predict_mask(self, sequence: str, patch_size: Optional[int] = None, stride: int = 1) -> np.ndarray:
        """
        Predict label for each letter in sequence
        :param sequence: Sequence
        :type sequence: str
        :param patch_size: Length of classification fragments. If None, use defined by Trainer
        :type patch_size: Optional[int]
        :param stride: Step taken between classification fragments. Must be set to 1 for precise classification of individual letters
        :type stride: int
        :return: Predicted label for each letter in sequence
        :rtype: np.ndarray
        """
        if patch_size is None:
            patch_size = self.hp.patch_size
        padding = "X" * (patch_size // 2)
        sequence = padding + sequence + padding
        patches = get_single_seq_patches(sequence, patch_len=patch_size, stride=stride)
        predictions = self.predict(patches)
        return predictions.cpu().numpy()

    @staticmethod
    def __check_save_path(save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        weights_dir = os.path.join(save_dir, "weights")
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        params_dir = os.path.join(save_dir, "params")
        if not os.path.exists(params_dir):
            os.mkdir(params_dir)
        return weights_dir, params_dir

    def save_model(self, save_dir: str, model_name: str, version: int = 1):
        """
        Save trained model weights and hyperparameters
        :param save_dir: Directory to save results
        :type save_dir: str
        :param model_name: Name of model, also prefix for filename
        :type model_name: str
        :param version: Version of model, greater - newer
        :type version: int
        """
        weights_path, params_path = self.__check_save_path(save_dir)
        weights_save_path = os.path.join(weights_path, f"{model_name}_v{version}.pt")
        params_save_path = os.path.join(params_path, f"{model_name}_v{version}.pk")
        if os.path.exists(weights_save_path):
            self.save_model(save_dir, model_name, version=version+1)
        else:
            torch.save(self.model.state_dict(), weights_save_path)
            with open(params_save_path, "wb") as file:
                pickle.dump(self.hp, file=file)
            logger.info(f"Model weights saved to {weights_save_path}")
            logger.info(f"Model parameters saved to {params_save_path}")

    def load_weights(self, weights_path: str):
        """
        Load weights to model from file
        :param weights_path: Path with saved pytorch state dict
        :type weights_path: object
        """
        if self.model:
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)
        else:
            logger.error("Model is not set up")

    @staticmethod
    def __load_pickle_hp(params_path: str):
        with open(params_path, "rb") as file:
            hp = pickle.load(file)
            return hp

    @staticmethod
    def __find_latest_version(weight_path: str, model_name: str):
        return max(map(lambda file: int(re.search(r'{}_v(\d+).*'.format(model_name), file).group(1)), os.listdir(weight_path)))

    @classmethod
    def make_predictor(cls, models_path: str, model_name: str) -> "Trainer":
        """
        Create instance of `Trainer` class used only for prediction from pretrained models
        :param models_path: Path to directory with saved models (must contain `weights` and `params` subdirectories)
        :type models_path: str
        :param model_name: Model name (base name aka class name or base name + version). E.g. HybridModel or SimpleCnn_v1
        :type model_name: str
        :return: Trainer object having pretrained model
        :rtype: Trainer
        """
        logger.info(f"Loading {model_name} from {models_path}")
        weights_path = os.path.join(models_path, "weights")
        params_path = os.path.join(models_path, "params")
        if os.path.exists(weights_path) and os.path.exists(params_path):
            if model_name.startswith("HybridModel"):
                model_class = HybridModel
            elif model_name.startswith("SimpleCNN"):
                model_class = SimpleCNN
            else:
                logger.error(f"Invalid model {model_name}")
                raise Exception

            if model_name in {"HybridModel", "SimpleCNN"}:
                last_version = cls.__find_latest_version(weights_path, model_name)
                model_name = f"{model_name}_v{last_version}"
                hp = cls.__load_pickle_hp(os.path.join(params_path, f"{model_name}.pk"))
            else:
                hp = cls.__load_pickle_hp(os.path.join(params_path, f"{model_name}.pk"))
            predictor = cls(model_class, None, None, None, None, hp)
            predictor.create_model()
            predictor.load_weights(os.path.join(weights_path, f"{model_name}.pt"))
            logger.info(f"Model {model_name} successfully loaded")
            return predictor

        else:
            logger.error(f"Invalid path. {weights_path} or {params_path} does not exist")
            raise FileNotFoundError
