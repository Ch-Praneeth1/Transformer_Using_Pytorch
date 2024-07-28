import torch 
import torch.nn as nn
from MultiHeadAttention import multiHeadAttention
from FeedForward import feedForward
from residualConnection import residualConnection
from LayerNormalization import layerNormalization

class encoderBlock(nn.Module):

    def __init__(self, self_attention_block: multiHeadAttention, feed_forward_block: feedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connectiosn = nn.ModuleList([residualConnection(dropout) for _ in range(2)])


    def forward(self, x, src_mask):
        x = self.residual_connectiosn[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connectiosn[1](x, self.feed_forward_block)
        return x
    

class encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    

