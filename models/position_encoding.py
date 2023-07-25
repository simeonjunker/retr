# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Position embedding for target in decoder
    taken from here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # shape (seq_len, 1, embedding_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim, seq_len]
        """
        batch_size, _, seq_len = x.shape
        # get positional encoding
        encoding = self.pe[:seq_len]
        # reshape & add batch dimension
        encoding = encoding.permute(1, 2, 0).repeat(batch_size, 1, 1)

        return encoding


class PositionalEmbedding(nn.Module):

    def __init__(self,
                 embedding_dim,
                 dropout=0.1,
                 max_position_embeddings=5000):
        super().__init__()

        self.pos_embed = nn.Embedding(num_embeddings=max_position_embeddings,
                                      embedding_dim=embedding_dim)
        self.LayerNorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, _, seq_len = x.shape
        device = x.device
        # get positional embedding
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_embeds = self.pos_embed(position_ids)
        position_embeds = self.LayerNorm(position_embeds)
        # reshape  & add batch dimension
        position_embeds = position_embeds.unsqueeze(0).permute(0, 2, 1)
        position_embeds = position_embeds.repeat(batch_size, 1, 1)

        return self.dropout(position_embeds)


def build_position_encoding(config):
    if config.position_embedding in ('v2', 'sine'):
        print('Using sine/cosine positional encodings')
        position_embedding = PositionalEncoding(config.hidden_dim,
                                                max_len=1024)
    elif config.position_embedding in ('v3', 'learned'):
        print('Using learned positional encodings')
        position_embedding = PositionalEmbedding(config.hidden_dim,
                                                 max_position_embeddings=1024,
                                                 dropout=0.1)
    else:
        raise ValueError(f"not supported {config.position_embedding}")

    return position_embedding