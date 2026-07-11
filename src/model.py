import torch
import torch.nn as nn
from torchvision import models

class MultiModalModelV2(nn.Module):
    def __init__(self, num_tickers=16, embedding_dim=16):
        super().__init__()
        self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
        base = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.cnn_fc = nn.Linear(512, 128)
        self.lstm = nn.LSTM(input_size=8, hidden_size=128,
                           num_layers=2, batch_first=True, dropout=0.3)
        self.lstm_fc = nn.Linear(128, 128)
        self.classifier = nn.Sequential(
            nn.Linear(272, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, img, ts, ticker_id):
        cnn_out = self.cnn(img)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = torch.relu(self.cnn_fc(cnn_out))
        lstm_out, _ = self.lstm(ts)
        lstm_out = torch.relu(self.lstm_fc(lstm_out[:, -1, :]))
        ticker_emb = self.ticker_embedding(ticker_id)
        combined = torch.cat([cnn_out, lstm_out, ticker_emb], dim=1)
        return self.classifier(combined)