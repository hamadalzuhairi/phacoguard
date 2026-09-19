"""GRU over cached frame features for surgical phase recognition."""
import torch, torch.nn as nn

class PhaseGRU(nn.Module):
    def __init__(self, feat_dim=2048, hidden=256, n_phases=7):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_phases)
    def forward(self, x):            # x: [B, T, feat_dim]
        h, _ = self.gru(x)
        return self.head(h[:, -1])   # predict phase at the last timestep
