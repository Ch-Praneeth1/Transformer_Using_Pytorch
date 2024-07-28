import torch 
import torch.nn as nn
import math
from LayerNormalization import layerNormalization as LayerNormalization

class residualConnection(nn.Module):

    def __init__(Self, dropout: float) -> None:
        super().__init__()
        Self.dropout = nn.Dropout(dropout)
        Self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))