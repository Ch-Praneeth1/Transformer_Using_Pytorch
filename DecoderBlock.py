import torch 
import torch.nn as nn
from MultiHeadAttention import multiHeadAttention
from FeedForward import feedForward
from residualConnection import residualConnection
from LayerNormalization import layerNormalization


class decoderBlock(nn.Module):

    def __init__(self, self_attention_block: multiHeadAttention, cross_attention_block: multiHeadAttention, feed_forward_block: feedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block= self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([residualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    

class decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
