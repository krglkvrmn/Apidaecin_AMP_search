from typing import Tuple, Union

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    Simple multilayer convolutional network
    :param n_classes: Number of output classes
    :type n_classes: int
    :param patch_size: Length single sequence vector
    :type patch_len: int
    :param embedding_size: Output size of embedding layer
    :type embedding_size: int
    :param hidden_fc_dim: Number of neurons in hidden fully-connected layer
    :type hidden_fc_dim: int
    :param kwargs: Not needed, for compatibility with training interface only
    :type kwargs: dict
    """

    def __init__(self, n_classes: int = 2, patch_size: int = 50, embedding_size: int = 20, hidden_fc_dim: int = 100,
                 **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_fc_dim = hidden_fc_dim
        self.patch_size = patch_size

        self.embedding = nn.Linear(27, embedding_size)

        self.conv = nn.Sequential(
            nn.Conv1d(embedding_size, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=4),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=4),
            nn.Flatten(),
        )

        fc_input_size = self._calc_fc_input_size()

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, self.hidden_fc_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_fc_dim, n_classes)
        )

    def _calc_fc_input_size(self):
        dummy_input = torch.rand((1, self.patch_size, 27))
        dummy_input = self.embedding(dummy_input)
        dummy_input = torch.transpose(dummy_input, 2, 1)
        dummy_input = self.conv(dummy_input)
        return dummy_input.size(1)

    def forward(self, x: torch.FloatTensor,
                return_embedding: bool = False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Pass data through network
        :param x: Input tensor of shape (batch_size, patch_size, n_amino_acids)
        :type x: torch.FloatTensor
        :param return_embedding: Whether return latent data representation
        :type return_embedding: bool
        :return: Tensor with predicted labels of shape (batch_size, n_classes) with optional
                 embedding tensor of shape (batch_size, latent_size)
        :rtype: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]
        """
        x = self.embedding(x)
        x = torch.transpose(x, 2, 1)
        x = self.conv(x)
        if return_embedding:
            return self.fc(x), x
        return self.fc(x)


class HybridModel(nn.Module):
    """
    Network with hybrid architecture containing convolutional and recurrent layers
    :param n_classes: Number of output classes
    :type n_classes: int
    :param embedding_size: Output size of embedding layer
    :type embedding_size: int
    :param conv_channels: Number of output channels of convolutional layer
    :type conv_channels: int
    :param conv_kernel_size: Number of output channels of convolutional layer
    :type conv_kernel_size: int
    :param pool_kernel_size: Kernel size and stride of MaxPool1d layer
    :type pool_kernel_size: int
    :param dropout_rate: Dropout rate
    :type dropout_rate: float
    :param blstm_input_size: Input size of bidirectional LSTM layer. Must be equal `conv_channels`
    :type blstm_input_size: int
    :param blstm_output_size: Output size of bidirectional LSTM layer
    :type blstm_output_size: int
    :param lstm_output_size: Output size of LSTM layer
    :type lstm_output_size: int
    :param activation: Activation function
    :type activation: torch.nn.Module

    """

    def __init__(self, n_classes: int, embedding_size: int = 20, conv_channels: int = 256,
                 conv_kernel_size: int = 5, pool_kernel_size: int = 3, dropout_rate: float = 0.2,
                 blstm_input_size: int = 256, blstm_output_size: int = 512, lstm_output_size: int = 512,
                 activation: torch.nn.Module = nn.ReLU()):
        super().__init__()
        self.n_classes = n_classes

        self.embedding = nn.Linear(27, embedding_size)
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=conv_channels,
                               kernel_size=conv_kernel_size,
                               padding=conv_kernel_size // 2)  # out_channels=64 - mcc does not exceed 0.9

        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size,
                                 stride=pool_kernel_size)
        self.do = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(input_size=blstm_input_size, hidden_size=blstm_output_size,
                              batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=blstm_output_size * 2, hidden_size=lstm_output_size, batch_first=True)
        self.fc = nn.Linear(lstm_output_size, n_classes)

        self.activation = activation

    def forward(self, x: torch.FloatTensor,
                return_embedding: bool = False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Pass data through network
        :param x: Input tensor of shape (batch_size, patch_size, n_amino_acids)
        :type x: torch.FloatTensor
        :param return_embedding: Whether return latent data representation
        :type return_embedding: bool
        :return: Tensor with predicted labels of shape (batch_size, n_classes) with optional
                 embedding tensor of shape (batch_size, lstm_output_size)
        :rtype: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]
        """
        x = self.embedding(x)
        x = torch.transpose(x, 2, 1)
        x = self.conv1(x)
        x = self.activation(x)

        x = self.pool(x)
        x = self.do(x)
        x = torch.transpose(x, 2, 1)
        x, _ = self.bilstm(x)
        x = self.do(x)
        _, (h_n, _) = self.lstm(x)

        if return_embedding:
            return self.fc(h_n.squeeze()), h_n.squeeze()

        return self.fc(h_n.squeeze())
