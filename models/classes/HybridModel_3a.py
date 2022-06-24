from dataclasses import dataclass
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef


precision_score.__name__ = "precision_score"
recall_score.__name__ = "recall_score"
f1_score.__name__ = "f1_score"


class OneHotEncoder:
    ALPHABETS = {
            "dna": "ATGC",
            "prot": "ACDEFGHIKLMNPQRSTVWYXBZJUO*"
    }

    def __init__(self, alphabet="prot", device="cpu"):
        self.device = device
        self.alphabet = self.ALPHABETS.get(alphabet)
        if not self.alphabet:
            self.alphabet = alphabet
        self.letter_index_map = dict(zip(self.alphabet, range(len(self.alphabet))))

    def get_batch_embedding(self, sequences):
        proposed_len = len(sequences[0])
        assert all(len(seq) == proposed_len for seq in sequences)
        sequences_combined = "".join(sequences)
        sequences_combined_aa_labels = [self.letter_index_map[aa] for aa in sequences_combined]
        one_hot_vector = nn.functional.one_hot(torch.LongTensor(sequences_combined_aa_labels),
                                 num_classes=len(self.letter_index_map)).view(len(sequences), -1, len(self.letter_index_map)).float()
        return one_hot_vector
    
    
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

    
class HybridModel(nn.Module):
    def __init__(self, n_classes, embedding_size=20, conv_channels=256,
                 conv_kernel_size=5, pool_kernel_size=3, dropout_rate=0.2,
                 blstm_input_size=256, blstm_output_size=512, lstm_output_size=512,
                 activation=nn.ReLU()):
        super().__init__()
        self.n_classes = n_classes

        self.embedding = nn.Linear(27, embedding_size)
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=conv_channels,
                               kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)   # out_channels=64 - mcc does not exceed 0.9
 
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size,
                                 stride=pool_kernel_size)
        self.do = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(input_size=blstm_input_size, hidden_size=blstm_output_size,
                              batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=blstm_output_size*2, hidden_size=lstm_output_size, batch_first=True)
        self.fc = nn.Linear(lstm_output_size, n_classes)

        self.activation = activation

    def forward(self, x, return_embedding=False):
        x = self.embedding(x)
        x = torch.transpose(x, 2, 1)
        x = self.conv1(x)
        x = self.activation(x)

        x = self.pool(x)
        x = self.do(x)
        x = torch.transpose(x, 2, 1)
        x, (h_n, c_n) = self.bilstm(x)
        x = self.do(x)
        x, (h_n, c_n) = self.lstm(x)
        
        if return_embedding:
            return h_n.squeeze()
        
        x = self.fc(h_n.squeeze())
        return x

DEVICE = "cpu"

mp = ModelParameters(
    n_classes = 2,
    embedding_size = 39,
    conv_channels = 256,
    conv_kernel_size = 12,
    dropout_rate = 0.2,

    blstm_input_size = 256,
    blstm_output_size = 512,
    lstm_output_size = 512,

    activation = "relu"
)

hp = Hyperparameters(
        batch_size = 1000,
        patch_size = 33,
        patch_stride = 1,
        substitution_matrix = "BLOSUM45",
        replacement_proba_factor = 250,
        pos_proba = 0.1,
        db_proba = 0.1,

        model_parameters = mp,
        encoder = OneHotEncoder(alphabet="prot", device=DEVICE),

        criterion = nn.CrossEntropyLoss(),
        optimizer = "adam",
        scheduler = None,
        lr = 0.0047474,

        metric_fns = ()
)
    
if __name__ == "__main__":
    pass