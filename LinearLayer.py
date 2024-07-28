import torch 
import torch.nn as nn 
import math 


class linearLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, voacb_size)
        return torch.log_softmax(self.linear(x), dim = -1)
