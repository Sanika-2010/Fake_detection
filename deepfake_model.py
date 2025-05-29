import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)  # Change from (512, 512) to (512, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn_weights = self.softmax(self.attn(x))  # Shape: [batch, seq_len, 1]
        attn_weights = attn_weights.squeeze(-1)  # Shape: [batch, seq_len]
        attn_out = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # Weighted sum
        return attn_out

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1792, hidden_size=512, num_layers=2, batch_first=True)
        self.attention = Attention(512)  # Keep shape [512, 1]
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)  
        return self.fc(attn_out)
