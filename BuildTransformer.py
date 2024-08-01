import torch 
import torch.nn as nn
from Transformer import transformer as Transformer
from InputEmbeddings import inputEmbeddings
from PositionalEncoding import positionalEncoding
from MultiHeadAttention import multiHeadAttention
from FeedForward import feedForward
from EncoderBlock import encoderBlock
from DecoderBlock import decoderBlock
from DecoderBlock import decoder as Decoder 
from EncoderBlock import encoder as Encoder 
from LinearLayer import linearLayer 


def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> transformer:

    # create a embedding layer 
    src_embed = inputEmbeddings(d_model, src_vocab_size)
    target_embed = inputEmbeddings(d_model, target_vocab_size)

    # create the positional encoding layers
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    target_pos = positionalEncoding(d_model, target_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = multiHeadAttention(d_model, h, dropout)
        feed_forward_block = feedForward(d_model, d_ff, dropout)
        encoder_block = encoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks 
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = multiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = multiHeadAttention(d_model, h, dropout)
        feed_forward_block = feedForward(d_model, d_ff, dropout)
        decoder_block = decoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the linear layer
    lineary_layer = linearLayer(d_model, target_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, lineary_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)

    return transformer
