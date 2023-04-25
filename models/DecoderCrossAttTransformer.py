# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .position_encoding import build_position_encoding


class DecoderCrossAttTransformer(nn.Module):

    # ENCODER
    # [Target] -> SelfAtt#1 -> [SAtt_T]                 | Encode Target
    # [Context] -> SelfAtt#2 -> [SAtt_C]                | Encode Context

    # DECODER
    # [Cap] -> SelfAtt#3 -> [SAtt_Cap]                  | Encode Cap
    # [SAtt_Cap] / [SAtt_T] -> CrossAtt#1 -> [SAtt_Cap] | Merge Cap / Target
    # [SAtt_Cap] / [SAtt_C] -> CrossAtt#2 -> [Out]      | Merge Cap / Context

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.positional_encoding = build_position_encoding(config)

        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_t, mask_t, src_c, mask_c, tgt, tgt_mask):     

        bs, c, hw = src_t.shape

        # pos embed + permute NxCxHW to HWxNxC for target
        pos_t = self.positional_encoding(src_t)
        src_t = src_t.permute(2, 0, 1)
        pos_t = pos_t.permute(2, 0, 1)

        # pos embed + permute NxCxHW to HWxNxC for context
        pos_c = self.positional_encoding(src_c)
        src_c = src_c.permute(2, 0, 1)
        pos_c = pos_c.permute(2, 0, 1)

        # embed + permute target expression
        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        # Dual Encoder (separate for target / context)
        memory_t, memory_c = self.encoder(
            src_t, src_c,
            src_t_key_padding_mask=mask_t, src_c_key_padding_mask=mask_c,
            pos_t=pos_t, pos_c=pos_c
        )
        # Decoder (with target / context merge via cross-attention)
        hs = self.decoder(
            tgt, 
            memory_t, 
            memory_c, 
            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device),
            tgt_key_padding_mask=tgt_mask,
            t_memory_key_padding_mask=mask_t, 
            c_memory_key_padding_mask=mask_c, 
            t_pos=pos_t, 
            c_pos=pos_c, 
            query_pos=query_embed
        )

        return hs


class TransformerEncoder(nn.Module):

    # ENCODER
    # [Target] -> SelfAtt#1 -> [SAtt_T]             | Encode Target
    # [Context] -> SelfAtt#2 -> [SAtt_C]            | Encode Context

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.t_norm = norm
        self.c_norm = copy.deepcopy(self.t_norm)

    def forward(self, 
                src_t, src_c,
                src_t_mask: Optional[Tensor] = None, src_c_mask: Optional[Tensor] = None,
                src_t_key_padding_mask: Optional[Tensor] = None, src_c_key_padding_mask: Optional[Tensor] = None,
                pos_t: Optional[Tensor] = None, pos_c: Optional[Tensor] = None
                ):
        
        out_t, out_c = src_t, src_c

        for layer in self.layers:
            out_t, out_c = layer(
                out_t, out_c,
                src_t_mask=src_t_mask, src_c_mask=src_c_mask, 
                src_t_key_padding_mask=src_t_key_padding_mask, src_c_key_padding_mask=src_c_key_padding_mask, 
                pos_t=pos_t, pos_c=pos_c
            )

        if self.t_norm is not None:
            out_t = self.t_norm(out_t)
        if self.c_norm is not None:
            out_c = self.c_norm(out_c)

        return out_t, out_c


class TransformerDecoder(nn.Module):

    # DECODER
    # [Cap] -> SelfAtt#3 -> [SAtt_Cap]                  | Encode Cap
    # [SAtt_Cap] / [SAtt_T] -> CrossAtt#1 -> [SAtt_Cap] | Merge Cap / Target
    # [SAtt_Cap] / [SAtt_C] -> CrossAtt#2 -> [Out]      | Merge Cap / Context

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, 
                tgt, 
                t_memory,
                c_memory,
                tgt_mask: Optional[Tensor] = None,
                t_memory_mask: Optional[Tensor] = None,
                c_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                t_memory_key_padding_mask: Optional[Tensor] = None,
                c_memory_key_padding_mask: Optional[Tensor] = None,
                t_pos: Optional[Tensor] = None,
                c_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, 
                           t_memory, 
                           c_memory, 
                           tgt_mask=tgt_mask,
                           t_memory_mask=t_memory_mask,
                           c_memory_mask=c_memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           t_memory_key_padding_mask=t_memory_key_padding_mask,
                           c_memory_key_padding_mask=c_memory_key_padding_mask,
                           t_pos=t_pos, 
                           c_pos=c_pos,                            
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    # ENCODER
    # [Target] -> SelfAtt#1 -> [SAtt_T]             | Encode Target
    # [Context] -> SelfAtt#2 -> [SAtt_C]            | Encode Context

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Target Self-Att
        self.t_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.t_dropout = nn.Dropout(dropout)
        self.t_norm_1 = nn.LayerNorm(d_model)
        self.t_norm_2 = nn.LayerNorm(d_model)

        # Context Self-Att
        self.c_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.c_dropout = nn.Dropout(dropout)
        self.c_norm_1 = nn.LayerNorm(d_model)
        self.c_norm_2 = nn.LayerNorm(d_model)

        # Target Feedforward
        self.t_linear1 = nn.Linear(d_model, dim_feedforward)
        self.t_dropout1 = nn.Dropout(dropout)
        self.t_linear2 = nn.Linear(dim_feedforward, d_model)
        self.t_dropout2 = nn.Dropout(dropout)

        # Context Feedforward
        self.c_linear1 = nn.Linear(d_model, dim_feedforward)
        self.c_dropout1 = nn.Dropout(dropout)
        self.c_linear2 = nn.Linear(dim_feedforward, d_model)
        self.c_dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src_t, src_c,
                    src_t_mask: Optional[Tensor] = None,
                    src_c_mask: Optional[Tensor] = None,
                    src_c_key_padding_mask: Optional[Tensor] = None,
                    src_t_key_padding_mask: Optional[Tensor] = None,
                    pos_c: Optional[Tensor] = None,
                    pos_t: Optional[Tensor] = None):
    
        assert self.normalize_before

        # encode target via self-attention
        src_t2 = self.t_norm_1(src_t)
        q = k = self.with_pos_embed(src_t2, pos_t)
        src_t2 = self.t_self_attn(q, k, value=src_t2, attn_mask=src_t_mask,
                              key_padding_mask=src_t_key_padding_mask)[0]
        src_t = src_t + self.t_dropout(src_t2)
        src_t2 = self.t_norm_2(src_t)

        # target linear layer
        src_t2 = self.t_linear2(self.t_dropout1(self.activation(self.t_linear1(src_t2))))
        src_t = src_t + self.t_dropout2(src_t2)

        # encode context via self-attention
        src_c2 = self.c_norm_1(src_c)
        q = k = self.with_pos_embed(src_c2, pos_c)
        src_c2 = self.c_self_attn(q, k, value=src_c2, attn_mask=src_c_mask,
                              key_padding_mask=src_c_key_padding_mask)[0]
        src_c = src_c + self.c_dropout(src_c2)
        src_c2 = self.c_norm_2(src_c)

        # context linear layer
        src_c2 = self.c_linear2(self.c_dropout1(self.activation(self.c_linear1(src_c2))))
        src_c = src_c + self.c_dropout2(src_c2)

        return src_t, src_c


class TransformerDecoderLayer(nn.Module):

    # DECODER
    # [Cap] -> SelfAtt#3 -> [SAtt_Cap]                  | Encode Cap
    # [SAtt_Cap] / [SAtt_T] -> CrossAtt#1 -> [SAtt_Cap] | Merge Cap / Target
    # [SAtt_Cap] / [SAtt_C] -> CrossAtt#2 -> [Out]      | Merge Cap / Context

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Expression Target Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Referential Target Cross-Attention
        self.t_multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.t_norm = nn.LayerNorm(d_model)
        self.t_dropout = nn.Dropout(dropout)

        # Referential Context Cross-Attention
        self.c_multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.c_norm = nn.LayerNorm(d_model)
        self.c_dropout = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, 
                    tgt, 
                    t_memory,
                    c_memory,
                    tgt_mask: Optional[Tensor] = None,
                    t_memory_mask: Optional[Tensor] = None,
                    c_memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    t_memory_key_padding_mask: Optional[Tensor] = None,
                    c_memory_key_padding_mask: Optional[Tensor] = None,
                    t_pos: Optional[Tensor] = None,
                    c_pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        
        assert self.normalize_before

        # self-attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # target/expression cross-attention
        tgt2 = self.t_norm(tgt)
        tgt2 = self.t_multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(t_memory, t_pos),
                                   value=t_memory, attn_mask=t_memory_mask,
                                   key_padding_mask=t_memory_key_padding_mask)[0]
        tgt = tgt + self.t_dropout(tgt2)

        # context/expression cross-attention
        tgt2 = self.c_norm(tgt)
        tgt2 = self.c_multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(c_memory, c_pos),
                                   value=c_memory, attn_mask=c_memory_mask,
                                   key_padding_mask=c_memory_key_padding_mask)[0]
        tgt = tgt + self.c_dropout(tgt2)

        # linear
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_transformer(config):
    return DecoderCrossAttTransformer(
        config,
        d_model=config.hidden_dim,
        dropout=config.dropout,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        normalize_before=config.pre_norm,
        return_intermediate_dec=False,
    )
