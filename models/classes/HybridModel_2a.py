# Train loss: 0.001855
# Validation loss: 0.001782
# matthews_corrcoef: 0.9900
# accuracy_score: 0.9996
# precision_score: 0.9945
# recall_score: 0.9859
# f1_score: 0.9902

import torch
from torch import nn


BATCH_SIZE = 1000
PATCH_SIZE = 35
EMBEDDING_SIZE = 20
LR = 1e-4
DEVICE = "cuda"


class HybridModel(nn.Module):
    def __init__(self, n_classes, patch_size, embedding_size):
        super().__init__()
        self.n_classes = n_classes
        self.patch_size = patch_size


        self.embedding = nn.Linear(20, embedding_size)
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=256, kernel_size=5, padding=2)   # out_channels=64 - mcc does not exceed 0.9
 
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.do = nn.Dropout(0.2)
        self.bilstm = nn.LSTM(input_size=256, hidden_size=512, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, batch_first=True)
        self.fc = nn.Linear(512, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.pool(x)
        x = self.do(x)
        x = torch.transpose(x, 2, 1)
        x, (h_n, c_n) = self.bilstm(x)
        x = self.do(x)
        x, (h_n, c_n) = self.lstm(x)

        x = self.fc(h_n.squeeze())
        return x
    
    
if __name__ == "__main__":
    pos_labels = [1] * len(pro_apidaecins_sequences)
    neg_labels = [0] * len(proteins_sequences) + [2] * len(apd_sequences)
    X_train, X_test, y_train, y_test = train_test_split(pro_apidaecins_sequences + proteins_sequences + apd_sequences , pos_labels + neg_labels,
                                                        random_state=42, shuffle=True, test_size=0.2)
    augmentator = SequenceAugmentator("BLOSUM45", replacement_proba_factor=200)
    train_dataset = SequencePatchDataset(X_train, y_train, patch_len=PATCH_SIZE, pos_weight=0.05, db_weight=0.1, augmentator=augmentator)
    test_dataset = SequencePatchDataset(X_test, y_test, patch_len=PATCH_SIZE)
    batch_size = 1000
    train_sampler = WeightedRandomSampler(train_dataset.weights, sum(train_dataset.labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = HybridModel(2, patch_size=patch_size, embedding_size=20)