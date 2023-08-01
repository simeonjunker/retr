# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .position_encoding import build_position_encoding
from .utils import _get_clones, generate_square_subsequent_mask
from .transformer_modules import feed_forward, SelfAttResidual, CrossAttResidual, FFResidual, DecoderEmbeddings


class ConcatTransformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False
                 ):
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
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_t, mask_t, src_c, mask_c, tgt, tgt_mask):

        # merge information
        if src_c is not None:
            # concatenate
            src = torch.concat([src_t, src_c], 2)
            mask = torch.concat([mask_t, mask_c], 1)
        else:
            src, mask = src_t, mask_t

        pos_embed = self.positional_encoding(src)

        # permute NxCxHW to HWxNxC
        bs, c, hw = src.shape

        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)

        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        memory, encoder_atts = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        out, decoder_atts = self.decoder(tgt, memory, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))

        atts = {**encoder_atts, **decoder_atts}

        return out, atts


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        output = src

        all_layer_atts = defaultdict(list)

        for layer in self.layers:
            output, att_dict = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            for att_label, att_values in att_dict.items():
                all_layer_atts[att_label].append(att_values)

        stacked_atts = dict()
        for att_label, att_value_list in all_layer_atts.items():
            stacked_atts[att_label] = torch.stack(att_value_list)

        if self.norm is not None:
            output = self.norm(output)

        return output, stacked_atts


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        output = tgt

        all_layer_atts = defaultdict(list)

        for layer in self.layers:
            output, att_dict = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            
            for att_label, att_values in att_dict.items():
                all_layer_atts[att_label].append(att_values)


        stacked_atts = dict()
        for att_label, att_value_list in all_layer_atts.items():
            stacked_atts[att_label] = torch.stack(att_value_list)
            
        if self.norm is not None:
            output = self.norm(output)

        return output, stacked_atts


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Self-Att
        self.self_attn = SelfAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout
        )
        
        # Feedforward
        self.ff = FFResidual(
            feed_forward(dim_input=d_model, dim_feedforward=dim_feedforward),
            dimension=d_model,
            dropout=dropout)

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,  # is None during inference
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        att_values = dict()

        # self attention
        att_label = 'enc_tc_self_att'
        src, att_value = self.self_attn(
            qkv=src,
            qkv_pos=pos,
            attn_mask=None,
            key_padding_mask=src_key_padding_mask
        )
        att_values[att_label] = att_value

        # feedforward
        src = self.ff(
            src
        )

        return src, att_values

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Expression Self-Attention
        self.tgt_self_attn = SelfAttResidual(
            nn.MultiheadAttention(d_model, nhead, dropout=dropout),
            dimension=d_model,
            dropout=dropout)

        # Expression/Memory Cross-Attention
        self.tgt_src_cross_attn = CrossAttResidual(
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
        
        att_values = dict()

        # EXPRESSION SELF ATT
        att_label = 'dec_exp_self_att'
        tgt, att_value = self.tgt_self_attn(
            qkv=tgt,
            qkv_pos=query_pos,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        att_values[att_label] = att_value

        # EXPRESSION / TARGET CROSS ATT
        att_label = 'dec_exp_tc_cross_att'
        tgt, att_value = self.tgt_src_cross_attn(
            q=tgt,
            kv=memory,
            q_pos=query_pos,
            k_pos=pos,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )
        att_values[att_label] = att_value

        # FEED FORWARD
        tgt = self.ff(
            tgt
        )

        return tgt, att_values

def build_transformer(config):
    return ConcatTransformer(
        config,
        d_model=config.hidden_dim,
        dropout=config.dropout,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        normalize_before=config.pre_norm,
    )
