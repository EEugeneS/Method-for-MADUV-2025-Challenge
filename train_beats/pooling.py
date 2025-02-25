import torch, torch.nn as nn, torch.nn.functional as F
from basicBlocks import *

class AvgPool(nn.Module):
    
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)


class AttentivePooling(nn.Module):
    def __init__(self, input_dim, attention_dim=128, kernel_size=1):
        super(AttentivePooling, self).__init__()
        self.attention_conv = nn.Conv1d(input_dim, attention_dim, kernel_size=kernel_size)
        self.attention_score = nn.Conv1d(attention_dim, 1, kernel_size=kernel_size)

    def forward(self, x):
        # input x, (B, T, D)
        x = x.permute(0, 2, 1)
        attention = torch.tanh(self.attention_conv(x))  # (batch_size, attention_dim, seq_len)
        scores = self.attention_score(attention)  # (batch_size, 1, seq_len)
        scores = scores.squeeze(1)  # (batch_size, seq_len)

        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        weighted_representation = torch.sum(
            x * attention_weights.unsqueeze(1), dim=2
        )
        return weighted_representation


