###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Implementation of Transformer decoder and decoder layer with self attention.

Implementation of the decoder layer as in [1] and [2] for sequence to 
sequence modeling.

[1] https://arxiv.org/pdf/1706.03762.pdf
[2] https://arxiv.org/pdf/2005.12872.pdf
"""


import numpy as np
import sys
import os

import torch
import torch.nn as nn

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
import utils.utils as utils


class DecoderLayer(nn.Module):
  """Implements the transformer decoder Layer."""

  def __init__(self,
               model_dim=256,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               init_fn=utils.normal_init_,
               pre_normalization=False,
               use_query_embedding=False):
    """Decoder layer initialization.

    Args:
      model_dim:
      num_heads:
      dim_ffn:
      dropout:
    """
    super(DecoderLayer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._pre_normalization = pre_normalization
    self._use_query_embedding = use_query_embedding

    self._self_attn = nn.MultiheadAttention(
        model_dim, num_heads, dropout=dropout
    )
    self._multihead_attn = nn.MultiheadAttention(
        model_dim, num_heads, dropout=dropout
    )

    # the so-called point-wise network
    self._linear1 = nn.Linear(model_dim, dim_ffn)
    self._linear2 = nn.Linear(dim_ffn, model_dim)
    self._relu = nn.ReLU()

    self._norm1 = nn.LayerNorm(model_dim)
    self._norm2 = nn.LayerNorm(model_dim)
    self._norm3 = nn.LayerNorm(model_dim)
    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)
    self._dropout3 = nn.Dropout(dropout)
    self._dropout4 = nn.Dropout(dropout)

    utils.weight_init(self._linear1, init_fn_=init_fn)
    utils.weight_init(self._linear2, init_fn_=init_fn)

    self._forward_fn = self.forward_pre if pre_normalization else self.forward_post

  def forward(self, 
              target_seq, 
              memory,
              pos_encodings, 
              query_embedding=None,
              mask_look_ahead=None, 
              mask_target_padding=None):
    """Forward pass of the layer.

    Args:
      target_seq: [target_seq_length, batch_size, model_dim]
      memory: [source_seq_length, batch_size, model_dim]
      mask_look_ahead: []
      mask_target_padding:
    """
    return self._forward_fn(
        target_seq,
        memory,
        pos_encodings,
        query_embedding=query_embedding,
        mask_look_ahead=mask_look_ahead,
        mask_target_padding=mask_target_padding
    )

  def handle_query_embedding(self, sequence, embedding):
    """Handle """
    if self._use_query_embedding:
      return sequence + embedding
    return sequence

  def forward_post(self, 
              target_seq, 
              memory,
              pos_encodings, 
              query_embedding=None,
              mask_look_ahead=None, 
              mask_target_padding=None):
    """Forward pass of the layer with post normalization.

    Args:
      target_seq: [target_seq_length, batch_size, model_dim]
      memory: [source_seq_length, batch_size, model_dim]
      mask_look_ahead: []
      mask_target_padding:
    """
    # 1) Compute self attention with current sequence of inferred tokens
    # query is the same as key for self attention
    # [batch_size, seq_length, model_dim]
    if self._use_query_embedding:
      q = k = v = target_seq + query_embedding
    else:
      q = k = v =  target_seq + pos_encodings

    self_attn, self_attn_weights = self._self_attn(
        query=q, key=k, value=v, #target_seq,
        attn_mask=mask_look_ahead,
        key_padding_mask=mask_target_padding
    )
    self_attn = self._dropout1(self_attn)
    out_self_attn = self._norm1(self_attn + target_seq)

    # 2) Attend the encoder's memory given the comptued self attention
    # [batch_size, seq_length, model_dim]
    attn, attn_weights = self._multihead_attn(
        query=self.handle_query_embedding(out_self_attn, query_embedding), 
        key=self.handle_query_embedding(memory, pos_encodings), 
        value=memory)
    attn = self._dropout2(attn)
    out_attn = self._norm2(attn + out_self_attn)

    # 3) Compute pointwise embeding by expanding and projecting + dropout
    ffn_output = self._linear1(out_attn)
    ffn_output = self._relu(ffn_output)
    ffn_output = self._dropout4(ffn_output)
    ffn_output = self._linear2(ffn_output)

    # 4) Compute residual connection as final output
    ffn_output = self._dropout3(ffn_output)
    outputs = self._norm3(ffn_output + out_attn)

    return outputs, self_attn_weights, attn_weights

  def forward_pre(self, 
              target_seq_, 
              memory,
              pos_encodings, 
              query_embedding=None,
              mask_look_ahead=None, 
              mask_target_padding=None):
    """Forward pass of the layer with pre normalization.

    Args:
      target_seq: [target_seq_length, batch_size, model_dim]
      memory: [source_seq_length, batch_size, model_dim]
      mask_look_ahead: []
      mask_target_padding:
    """
    target_seq = self._norm1(target_seq_)
    # 1) Compute self attention with current sequence of inferred tokens
    # query is the same as key for self attention
    # [batch_size, seq_length, model_dim]
    if self._use_query_embedding:
      # in case of using only the query embedding follow DETR [2] which drops
      # values to zero and uses only the query embeddings
      q = k = target_seq + query_embedding
      v = target_seq
    else:
      q = k = v =  target_seq + pos_encodings

    self_attn, self_attn_weights = self._self_attn(
        query=q, key=k, value=v,
        attn_mask=mask_look_ahead, 
        key_padding_mask=mask_target_padding
    )
    self_attn = self._dropout1(self_attn)
    out_self_attn = self._norm2(self_attn + target_seq_)

    # 2) Attend the encoder's memory given the comptued self attention
    # [batch_size, seq_length, model_dim]
    attn, attn_weights = self._multihead_attn(
        query=self.handle_query_embedding(out_self_attn, query_embedding), 
        key=self.handle_query_embedding(memory, pos_encodings),
        value=memory)
    attn = self._dropout2(attn)
    out_attn = self._norm3(attn + out_self_attn)

    # 3) Compute pointwise embeding by expanding and projecting + dropout
    ffn_output = self._linear1(out_attn)
    ffn_output = self._relu(ffn_output)
    ffn_output = self._dropout4(ffn_output)
    ffn_output = self._linear2(ffn_output)

    # 4) Compute residual connection as final output
    ffn_output = self._dropout3(ffn_output)

    return ffn_output, self_attn_weights, attn_weights


class TransformerDecoder(nn.Module):
  """Transformer decoder module."""

  def __init__(self,
               num_layers=6,
               model_dim=256,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               init_fn=utils.normal_init_,
               pre_normalization=False,
               use_query_embedding=False):
    super(TransformerDecoder, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._num_layers = num_layers
    self._use_query_embedding = use_query_embedding
    self._pre_normalization = pre_normalization

    self._decoder_stack = self.init_decoder_stack(init_fn)

  def init_decoder_stack(self, init_fn):
    stack = nn.ModuleList()
    for s in range(self._num_layers):
      layer = DecoderLayer(
          model_dim=self._model_dim,
          num_heads=self._num_heads,
          dim_ffn=self._dim_ffn,
          dropout=self._dropout,
          init_fn=init_fn,
          pre_normalization=self._pre_normalization,
          use_query_embedding=self._use_query_embedding
      )
      stack.append(layer)
    return stack

  def forward(self, 
              target_seq,
              memory,
              pos_encodings,
              query_embedding=None,
              mask_target_padding=None,
              mask_look_ahead=None,
              get_attn_weights=False):
    """Computes forward pass of decoder.

    Args:
      target_seq: [target_sequence_length, batch_size, model_dim].
      memory: [source_sequence_length, batch_size, model_dim].
      pos_encodings: [target_seq_length, model_dim].
      mask_look_ahead: [target_seq_length, model_dim].

    Returns:
      A tensor with the decoded attention with shape [target_sequence_length,
      batch_size, model_dim].
    """
    seq_length = target_seq.size()[0]
    output_list = []
    attn_weights_list = [] if get_attn_weights else None
    outputs = torch.zeros_like(target_seq) if self._use_query_embedding else target_seq

    for l in range(self._num_layers):
      outputs, self_attn_weights, attn_weights = self._decoder_stack[l](
          outputs, memory,
          pos_encodings=pos_encodings,
          query_embedding=query_embedding,
          mask_target_padding=mask_target_padding,
          mask_look_ahead=mask_look_ahead
      )
      if get_attn_weights:
        attn_weights_list.append(attn_weights)
      output_list.append(outputs)

    return output_list, attn_weights_list


if __name__ == '__main__':
  thispath = os.path.dirname(os.path.abspath(__file__))
  sys.path.insert(0, thispath+"/../")
  import utils.utils as utils

  seq_length = 55
  batch_size = 8
  model_dim = 256
  tgt_seq = torch.FloatTensor(seq_length, batch_size, model_dim).fill_(1)
  memory = torch.FloatTensor(seq_length, batch_size, model_dim).uniform_(0, 1)

  mask_look_ahead = utils.create_look_ahead_mask(seq_length)
  mask_look_ahead = torch.from_numpy(mask_look_ahead)

  encodings = torch.FloatTensor(seq_length, 1, model_dim).uniform_(0,1)

  decoder = TransformerDecoder() 
  outputs = decoder(tgt_seq, memory, encodings, mask_look_ahead=mask_look_ahead)
 
  print(outputs.size())


