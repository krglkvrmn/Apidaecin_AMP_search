from copy import deepcopy
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_funcs.dataset import SequencePatchDataset
from train_funcs.parameters import Hyperparameters, ModelParameters
from train_funcs.processing import SequenceAugmentator, get_single_seq_patches


class Trainer:
    """
    Class controls training, logging and caching results
    :param model_class: Class of model to set up and train
    :type model_class: torch.nn.Module
    :param X_train: Sequences to form train dataset
    :type X_train: List[str]
    :param X_val: Sequences to form validation dataset
    :type X_val: List[str]
    :param y_train: Labels to form train dataset
    :type y_train: List[str]
    :param y_val: Labels to form validation dataset
    :type y_val: List[str]
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
                 X_train: List[str], X_val: List[str], y_train: List[str], y_val: List[str],
                 hyperparameters: Hyperparameters, setup: bool = False, device: str = "cpu"):
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
