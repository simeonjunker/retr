import torch
import torch.nn.functional as F
from torch import nn
from .utils import with_pos_embed

def feed_forward(dim_input, dim_feedforward):
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class AttResidualBase(nn.Module):
    def __init__(self, sublayer, dimension, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension)


class SelfAttResidual(AttResidualBase):
    def forward(
            self, 
            qkv,
            qkv_pos,
            key_padding_mask, attn_mask
        ):

        # pre norm
        norm_qkv = self.norm(qkv)

        # positional encoding, query/key/value tensors
        query = key = with_pos_embed(norm_qkv, qkv_pos)
        value = norm_qkv

        # self attention
        att_out, att_weights = self.sublayer(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )

        # residual + dropout
        res_out = qkv + self.dropout(att_out)
        
        return res_out, att_weights


class CrossAttResidual(AttResidualBase):
    def forward(
            self, 
            q, kv,
            q_pos, k_pos,
            key_padding_mask, attn_mask
        ):

        # pre norm
        norm_q = self.norm(q)

        # positional encoding, query/key/value tensors
        query = with_pos_embed(norm_q, q_pos)
        key = with_pos_embed(kv, k_pos)
        value = kv

        # cross attention
        att_out, att_weights = self.sublayer(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )

        # residual + dropout
        res_out = q + self.dropout(att_out)
        
        return res_out, att_weights


class FFResidual(nn.Module):
    def __init__(self, sublayer, dimension, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension)

    def forward(
            self, x
        ):

        # pre norm
        x_norm = self.norm(x)

        # feedforward
        ff_out = self.sublayer(x_norm)

        # residual + dropout
        res_out = x + self.dropout(ff_out)

        return res_out
   

class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings