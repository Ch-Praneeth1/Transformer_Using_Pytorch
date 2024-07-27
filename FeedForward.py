import torch 
import torch.nn as nn


class feedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, droupout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.droupot = nn.Dropout(droupout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.droupot(torch.relu(self.linear_1(x))))
    