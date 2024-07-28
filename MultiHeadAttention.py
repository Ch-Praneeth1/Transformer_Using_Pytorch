import torch 
import torch.nn as nn
import math 


class multiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = h
        assert d_model % h == 0, "d_model is not divisible by h"    # d_model should be divided by h to divide the d_model into h-no.of heads equally

        self.d_k = d_model // h   # d_k is the dimession of the each head 
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv
        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query_prime, key_prime, value_prime, mask, dropout: nn.Dropout):
        d_k = query_prime.shape[-1]

        # (batch, h, seq_len, d_k)  -->  (batch, h, seq_len, seq_len)
        attention_scores = (query_prime @ key_prime.transponse(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value_prime) , attention_scores


    def forward(self, q, k, v, mask):
        query_prime = self.w_q(q)    # (Batch, Seq_len, d_model) --> (batch, seq_len, d_model)
        key_prime = self.w_k(k)      # (Batch, Seq_len, d_model) --> (batch, seq_len, d_model)
        value_prime = self.w_v(v)    # (Batch, Seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query_prime = query_prime.view(query_prime.shape[0], query_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        key_prime = key_prime.view(key_prime.shape[0], key_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        value_prime = value_prime.view(value_prime.shape[0], value_prime.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = multiHeadAttention.attention(query_prime, key_prime, value_prime, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k)  --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_le, d_model)
        return self.w_o(x)
