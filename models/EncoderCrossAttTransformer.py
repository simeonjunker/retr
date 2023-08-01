# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List

import torch
from torch import nn, Tensor
from .position_encoding import build_position_encoding
from .utils import _get_clones, generate_square_subsequent_mask
from .transformer_modules import feed_forward, SelfAttResidual, CrossAttResidual, FFResidual, DecoderEmbeddings


class EncoderCrossAttTransformer(nn.Module):

    # ENCODER
    # [Target] -> SelfAtt#1 -> [SAtt_T]             | Encode Target
    # [Context] -> SelfAtt#2 -> [SAtt_C]            | Encode Context
    # [SAtt_T] / [SAtt_C] -> CrossAtt#1 -> [Src]    | Merge Target / Context

    # DECODER
    # [Cap] -> SelfAtt#3 -> [SAtt_Cap]              | Encode Cap
    # [SAtt_Cap] / [Src] -> CrossAtt#2 -> [Out]     | Merge TargetContext / Cap

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

        # Encoder (with target / context merge)
        memory = self.encoder(
            src_t, src_c,
            src_t_key_padding_mask=mask_t, src_c_key_padding_mask=mask_c,
            pos_t=pos_t, pos_c=pos_c
        )
        # Decoder
        hs = self.decoder(
            tgt, memory, 
            memory_key_padding_mask=mask_t, tgt_key_padding_mask=tgt_mask,
            pos=pos_t, query_pos=query_embed,
            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        )

        return hs


class TransformerEncoder(nn.Module):

    # ENCODER
    # [Target] -> SelfAtt#1 -> [SAtt_T]             | Encode Target
    # [Context] -> SelfAtt#2 -> [SAtt_C]            | Encode Context
    # [SAtt_T] / [SAtt_C] -> CrossAtt#1 -> [Src]    | Merge Target / Context

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                src_t, src_c,
                src_t_mask: Optional[Tensor] = None, src_c_mask: Optional[Tensor] = None,
                src_t_key_padding_mask: Optional[Tensor] = None, src_c_key_padding_mask: Optional[Tensor] = None,
                pos_t: Optional[Tensor] = None, pos_c: Optional[Tensor] = None
                ):
        
        output = src_t

        for layer in self.layers:
            output = layer(
                output, src_c,
                src_t_mask=src_t_mask, src_c_mask=src_c_mask, 
                src_t_key_padding_mask=src_t_key_padding_mask, src_c_key_padding_mask=src_c_key_padding_mask, 
                pos_t=pos_t, pos_c=pos_c
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    # DECODER
    # [Cap] -> SelfAtt#3 -> [SAtt_Cap]              | Encode Cap
    # [SAtt_Cap] / [Src (Memory)] -> CrossAtt#2 -> [Out]     | Merge TargetContext / Cap

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
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
    # [Target / Src] -> SelfAtt#1 -> [SAtt_T]       | Encode Target
    # [Context] -> SelfAtt#2 -> [SAtt_C]            | Encode Context
    # [SAtt_T] / [SAtt_C] -> CrossAtt#1 -> [Src]    | Merge Target / Context

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Target Self-Att
        self.t_self_attn = SelfAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout
        )

        # Context Self-Att
        self.c_self_attn = SelfAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout
        )

        # Target/Context Cross-Att
        self.tc_cross_attn = CrossAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout
        )
        
        # Feedforward
        self.ff = FFResidual(
            feed_forward(dim_input=d_model, dim_feedforward=dim_feedforward),
            dimension=d_model,
            dropout=dropout)

    def forward(self, src_t, src_c,
                    src_t_mask: Optional[Tensor] = None,  # is None during inference
                    src_c_mask: Optional[Tensor] = None,  # is None during inference
                    src_c_key_padding_mask: Optional[Tensor] = None,
                    src_t_key_padding_mask: Optional[Tensor] = None,
                    pos_c: Optional[Tensor] = None,
                    pos_t: Optional[Tensor] = None):
    
        # TARGET
        # self attention
        src_t, _ = self.t_self_attn(
            qkv=src_t,
            qkv_pos=pos_t,
            attn_mask=None,
            key_padding_mask=src_t_key_padding_mask
        )


        # CONTEXT
        # self attention
        src_c, _ = self.c_self_attn(
            qkv=src_c,
            qkv_pos=pos_c,
            attn_mask=None,
            key_padding_mask=src_c_key_padding_mask
        )

        # MERGE TARGET/CONTEXT
        # cross attention
        # TODO return attention here
        src_t, _ = self.tc_cross_attn(
            q=src_t,
            kv=src_c,
            q_pos=src_t,
            k_pos=src_c,
            attn_mask=None,
            key_padding_mask=src_c_key_padding_mask
        )

        # FEED FORWARD
        src_t = self.ff(
            src_t
        )

        return src_t


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Expression Self-Attention
        self.tgt_self_attn = SelfAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout)

        # Expression/Target Cross-Attention
        self.tgt_t_cross_attn = CrossAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout)

        # Feed Forward
        self.ff = FFResidual(
            feed_forward(dim_input=d_model, dim_feedforward=dim_feedforward),
            dimension=d_model,
            dropout=dropout)
        

    def forward(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,  # is None during inference
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        
        # EXPRESSION SELF ATT
        tgt, _ = self.tgt_self_attn(
            qkv=tgt,
            qkv_pos=query_pos,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )

        # EXPRESSION / TARGET CROSS ATT
        tgt, _ = self.tgt_t_cross_attn(
            q=tgt,
            kv=memory,
            q_pos=query_pos,
            k_pos=pos,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )
        
        # FEED FORWARD
        tgt = self.ff(
            tgt
        )

        return tgt


def build_transformer(config):
    return EncoderCrossAttTransformer(
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
