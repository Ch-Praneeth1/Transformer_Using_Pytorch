import torch 
import torch.nn as nn
from EncoderBlock import encoder
from DecoderBlock import decoder
from InputEmbeddings import inputEmbeddings
from PositionalEncoding import positionalEncoding
from LinearLayer import linearLayer



class transformer(nn.Module):

    def __init__(self, encoder: encoder, decoder: decoder, src_embed: inputEmbeddings, target_embed: inputEmbeddings, src_pos: positionalEncoding, target_pos: positionalEncoding, linear_layer: linearLayer) -> None :
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.linear_layer = linear_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_ouput, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_ouput, src_mask, target_mask)
    
    def linear_layer(self, x):
        return self.linear_layer(x)
    

    