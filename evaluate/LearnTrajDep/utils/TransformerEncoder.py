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

"""Implementation of Transformer encoder and encoder layer with self attention.

Implementation of the encoder layer as in [1] and [2] for sequence to 
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


class EncoderLayer(nn.Module):
  """Implements the transformer encoder Layer."""

  def __init__(self,
               model_dim=256,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               init_fn=utils.normal_init_,
               pre_normalization=False):
    """Encoder layer initialization.

    Args:
      model_dim:
      num_heads:
      dim_ffn:
      dropout:
    """
    super(EncoderLayer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._pre_normalization = pre_normalization

    self._self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout)
    self._relu = nn.ReLU()
    self._dropout_layer = nn.Dropout(self._dropout)

    self._linear1 = nn.Linear(model_dim, self._dim_ffn)
    self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
    self._norm1 = nn.LayerNorm(model_dim, eps=1e-5)
    self._norm2 = nn.LayerNorm(model_dim, eps=1e-5)

    utils.weight_init(self._linear1, init_fn_=init_fn)
    utils.weight_init(self._linear2, init_fn_=init_fn)

  def forward(self, source_seq, pos_encodings):
    """Computes forward pass according.

    Args:
      source_seq: [sequence_length, batch_size, model_dim].
      pos_encodings: [sequence_length, model_dim].

    Returns:
      Tensor of shape [sequence_length, batch_size, model_dim].
    """
    if self._pre_normalization:
      return self.forward_pre(source_seq, pos_encodings)

    return self.forward_post(source_seq, pos_encodings)

  def forward_post(self, source_seq, pos_encodings):
    """Computes decoder layer forward pass with pre normalization.

    Args:
      source_seq: [sequence_length, batch_size, model_dim].
      pos_encodings: [sequence_length, model_dim].

    Returns:
      Tensor of shape [sequence_length, batch_size, model_dim].
    """
    # add positional encodings to the input sequence
    # for self attention query is the same as key
    query = source_seq + pos_encodings
    key = query
    value = source_seq

    attn_output, attn_weights = self._self_attn(
        query, 
        key, 
        value, 
        need_weights=True
    )

    norm_attn = self._dropout_layer(attn_output) + source_seq
    norm_attn = self._norm1(norm_attn)

    output = self._linear1(norm_attn)
    output = self._relu(output)
    output = self._dropout_layer(output)
    output = self._linear2(output)
    output = self._dropout_layer(output) + norm_attn
    output = self._norm2(output)

    return output, attn_weights

  def forward_pre(self, source_seq_, pos_encodings):
    """Computes decoder layer forward pass with pre normalization.

    Args:
      source_seq: [sequence_length, batch_size, model_dim].
      pos_encodings: [sequence_length, model_dim].

    Returns:
      Tensor of shape [sequence_length, batch_size, model_dim].
    """
    # add positional encodings to the input sequence
    # for self attention query is the same as key
    source_seq = self._norm1(source_seq_)
    query = source_seq + pos_encodings
    key = query
    value = source_seq

    attn_output, attn_weights = self._self_attn(
        query, 
        key, 
        value, 
        need_weights=True
    )

    norm_attn_ = self._dropout_layer(attn_output) + source_seq_
    norm_attn = self._norm2(norm_attn_)

    output = self._linear1(norm_attn)
    output = self._relu(output)
    output = self._dropout_layer(output)
    output = self._linear2(output)
    output = self._dropout_layer(output) + norm_attn_

    return output, attn_weights


class TransformerEncoder(nn.Module):
  def __init__(self,
               num_layers=6,
               model_dim=256,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               init_fn=utils.normal_init_,
               pre_normalization=False):
    super(TransformerEncoder, self).__init__()
    """Transforme encoder initialization."""
    self._num_layers = num_layers
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    # self._norm = norm
    self._pre_normalization = pre_normalization

    self._encoder_stack = self.init_encoder_stack(init_fn)

  def init_encoder_stack(self, init_fn):
    """Create the stack of encoder layers."""
    stack = nn.ModuleList()
    for s in range(self._num_layers):
      layer = EncoderLayer(
          model_dim=self._model_dim,
          num_heads=self._num_heads,
          dim_ffn=self._dim_ffn,
          dropout=self._dropout,
          init_fn=init_fn,
          pre_normalization=self._pre_normalization
      )
      stack.append(layer)
    return stack

  def forward(self, input_sequence, pos_encodings):
    """Computes decoder forward pass.

    Args:
      source_seq: [sequence_length, batch_size, model_dim].
      pos_encodings: [sequence_length, model_dim].

    Returns:
      Tensor of shape [sequence_length, batch_size, model_dim].
    """
    outputs = input_sequence

    for l in range(self._num_layers):
      outputs, attn_weights = self._encoder_stack[l](outputs, pos_encodings)

#    if self._norm:
#      outputs = self._norm(outputs)

    return outputs, attn_weights

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  seq_length = 50

  pos_encodings = torch.FloatTensor(seq_length, 1, 256).uniform_(0,1)
  seq = torch.FloatTensor(seq_length, 8, 256).fill_(1.0)

  pos_encodings = pos_encodings.to(device)
  seq = seq.to(device)

  encoder = TransformerEncoder(num_layers=6)
  encoder.to(device)
  encoder.eval()

  print(encoder(seq, pos_encodings).size())


