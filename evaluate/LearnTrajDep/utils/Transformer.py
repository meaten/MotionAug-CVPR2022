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


"""Implementation of the Transformer for sequence-to-sequence decoding.

Implementation of the transformer for sequence to sequence prediction as in
[1] and [2].

[1] https://arxiv.org/pdf/1706.03762.pdf
[2] https://arxiv.org/pdf/2005.12872.pdf
"""


import numpy as np
import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils
import utils.PositionEncodings as PositionEncodings
import utils.TransformerEncoder as Encoder
import utils.TransformerDecoder as Decoder


class Transformer(nn.Module):
  def __init__(self,
              num_encoder_layers=6,
              num_decoder_layers=6,
              model_dim=256,
              num_heads=8,
              dim_ffn=2048,
              dropout=0.1,
              init_fn=utils.normal_init_,
              use_query_embedding=False,
              pre_normalization=False,
              query_selection=False,
              target_seq_len=25):
    """Implements the Transformer model for sequence-to-sequence modeling."""
    super(Transformer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._use_query_embedding = use_query_embedding
    self._query_selection = query_selection
    self._tgt_seq_len = target_seq_len

    self._encoder = Encoder.TransformerEncoder(
        num_layers=num_encoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        pre_normalization=pre_normalization
    )
    self._decoder = Decoder.TransformerDecoder(
        num_layers=num_decoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        use_query_embedding=use_query_embedding,
        pre_normalization=pre_normalization
    )

    if self._query_selection:
      self._position_predictor = nn.Linear(self._model_dim, self._tgt_seq_len)

  def process_index_selection(self, self_attn, one_to_one_selection=False):
    """Selection of query elments using position predictor from encoder memory.

    After prediction a maximum assignement problem is solved to get indices for
    each element in the query sequence.

    Args:
      self_attn: Encoder memory with shape [src_len, batch_size, model_dim]

    Returns:
      A tuple with two list of i and j matrix entries of m
    """
    batch_size = self_attn.size()[1]
    # batch_size x src_seq_len x model_dim
    in_pos = torch.transpose(self_attn, 0, 1)
    # predict the matrix of similitudes
    # batch_size x src_seq_len x tgt_seq_len 
    prob_matrix = self._position_predictor(in_pos)
    # apply softmax on the similutes to get probabilities on positions
    # batch_size x src_seq_len x tgt_seq_len

    if one_to_one_selection:
      soft_matrix = F.softmax(prob_matrix, dim=2)
      # predict assignments in a one to one fashion maximizing the sum of probs
      indices = [linear_sum_assignment(soft_matrix[i].cpu().detach(), maximize=True) 
                 for i in range(batch_size)
      ]
    else:
      # perform softmax by rows to have many targets to one input assignements
      soft_matrix = F.softmax(prob_matrix, dim=1)
      indices_rows = torch.argmax(soft_matrix, 1)
      indices = [(indices_rows[i], list(range(prob_matrix.size()[2]))) 
          for i in range(batch_size)
      ]

    return indices, soft_matrix 

  def forward(self,
              source_seq,
              target_seq,
              encoder_position_encodings=None,
              decoder_position_encodings=None,
              query_embedding=None,
              mask_target_padding=None,
              mask_look_ahead=None,
              get_attn_weights=False,
              query_selection_fn=None):
    if self._use_query_embedding:
      bs = source_seq.size()[1]
      query_embedding = query_embedding.unsqueeze(1).repeat(1, bs, 1)
      decoder_position_encodings = encoder_position_encodings
    
    memory, enc_weights = self._encoder(source_seq, encoder_position_encodings)

    tgt_plain = None
    # perform selection from input sequence
    if self._query_selection:
      indices, prob_matrix = self.process_index_selection(memory)
      tgt_plain, target_seq = query_selection_fn(indices)
    
    out_attn, out_weights = self._decoder(
        target_seq,
        memory,
        decoder_position_encodings,
        query_embedding=query_embedding,
        mask_target_padding=mask_target_padding,
        mask_look_ahead=mask_look_ahead,
        get_attn_weights=get_attn_weights
    )

    out_weights_ = None
    enc_weights_ = None
    prob_matrix_ = None
    if get_attn_weights:
      out_weights_, enc_weights_ = out_weights, enc_weights

    if self._query_selection:
      prob_matrix_ =  prob_matrix

    return out_attn, memory, out_weights_, enc_weights_, (tgt_plain, prob_matrix_)
