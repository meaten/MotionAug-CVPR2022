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

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils
import utils.PositionEncodings as PositionEncodings
import utils.TransformerEncoder as Encoder
import utils.TransformerDecoder as Decoder
from utils.Transformer import Transformer


_SOURCE_LENGTH = 110
_TARGET_LENGTH = 55
_POSE_DIM = 54
_PAD_LENGTH = _SOURCE_LENGTH


class PoseTransformer(nn.Module):
  """Implements the sequence-to-sequence Transformer .model for pose prediction."""
  def __init__(self,
               pose_dim=_POSE_DIM,
               model_dim=256,
               num_encoder_layers=6,
               num_decoder_layers=6,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               target_seq_length=_TARGET_LENGTH,
               source_seq_length=_SOURCE_LENGTH,
               input_dim=None,
               init_fn=utils.xavier_init_,
               non_autoregressive=False,
               use_query_embedding=False,
               pre_normalization=False,
               predict_activity=False,
               use_memory=False,
               num_activities=None,
               pose_embedding=None,
               pose_decoder=None,
               copy_method='uniform_scan',
               query_selection=False,
               pos_encoding_params=(10000, 1)):
    """Initialization of pose transformers."""
    super(PoseTransformer, self).__init__()
    self._target_seq_length = target_seq_length
    self._source_seq_length = source_seq_length
    self._pose_dim = pose_dim
    self._input_dim = pose_dim if input_dim is None else input_dim
    self._model_dim = model_dim
    self._use_query_embedding = use_query_embedding
    self._predict_activity = predict_activity
    self._num_activities = num_activities
    self._num_decoder_layers = num_decoder_layers
    self._mlp_dim = model_dim 
    self._non_autoregressive = non_autoregressive
    self._pose_embedding = pose_embedding
    self._pose_decoder = pose_decoder
    self._query_selection = query_selection
    thisname = self.__class__.__name__
    self._copy_method = copy_method
    self._pos_encoding_params = pos_encoding_params

    self._transformer = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        use_query_embedding=use_query_embedding,
        pre_normalization=pre_normalization,
        query_selection=query_selection,
        target_seq_len=target_seq_length
    )

    t_params = filter(lambda p: p.requires_grad, self._transformer.parameters())
    nparams = sum([np.prod(p.size()) for p in t_params])
    print('[INFO] ({}) Transformer has {} parameters!'.format(thisname, nparams))

    self._pos_encoder = PositionEncodings.PositionEncodings1D(
        num_pos_feats=self._model_dim,
        temperature=self._pos_encoding_params[0],
        alpha=self._pos_encoding_params[1]
    )
    self._pos_decoder = PositionEncodings.PositionEncodings1D(
        num_pos_feats=self._model_dim,
        temperature=self._pos_encoding_params[0],
        alpha=self._pos_encoding_params[1]
    )
    # self.init_pose_encoder_decoders(init_fn)
    self._use_class_token = False
    self.init_position_encodings()
    self.init_query_embedding()

    if self._use_class_token:
      self.init_class_token()

    if self._predict_activity:
      self._action_head_size = self._model_dim if self._use_class_token \
          else self._model_dim*(self._source_seq_length-1)
      self._action_head = nn.Sequential(
          nn.Linear(self._action_head_size, self._num_activities),
      )

  def init_query_embedding(self):
    """Initialization of query sequence embedding."""
    self._query_embed = nn.Embedding(self._target_seq_length, self._model_dim)
    print('[INFO] ({}) Init query embedding!'.format(self.__class__.__name__))
    nn.init.xavier_uniform_(self._query_embed.weight.data)
    # self._query_embed.weight.data.normal_(0.0, 0.004)

  def init_class_token(self):
    token = torch.FloatTensor(1, self._model_dim)
    print('[INFO] ({}) Init class token!'.format(self.__class__.__name__))
    self._class_token = nn.Parameter(token, requires_grad=True)
    nn.init.xavier_uniform_(self._class_token.data)

  def init_position_encodings(self):
    src_len = self._source_seq_length
    # when using a token we need an extra element in the sequence
    if self._use_class_token:
      src_len = src_len + 1
    encoder_pos_encodings = self._pos_encoder(src_len).view(
            src_len, 1, self._model_dim)
    decoder_pos_encodings = self._pos_decoder(self._target_seq_length).view(
            self._target_seq_length, 1, self._model_dim)
    mask_look_ahead = torch.from_numpy(
        utils.create_look_ahead_mask(
            self._target_seq_length, self._non_autoregressive))
    self._encoder_pos_encodings = nn.Parameter(
        encoder_pos_encodings, requires_grad=False)
    self._decoder_pos_encodings = nn.Parameter(
        decoder_pos_encodings, requires_grad=False)
    self._mask_look_ahead = nn.Parameter(
        mask_look_ahead, requires_grad=False)

  def forward(self, 
              input_pose_seq,
              target_pose_seq=None,
              mask_target_padding=None,
              get_attn_weights=False):
    """Performs the forward pass of the pose transformers.

    Args:
      input_pose_seq: Shape [batch_size, src_sequence_length, dim_pose].
      target_pose_seq: Shape [batch_size, tgt_sequence_length, dim_pose].

    Returns:
      A tensor of the predicted sequence with shape [batch_size, 
      tgt_sequence_length, dim_pose].
    """
    if self.training:
      return self.forward_training(
          input_pose_seq, target_pose_seq, mask_target_padding, get_attn_weights)

    # eval forward for non auto regressive type of model
    if self._non_autoregressive:
      return self.forward_training(
          input_pose_seq, target_pose_seq, mask_target_padding, get_attn_weights)

    return self.forward_autoregressive(
        input_pose_seq, target_pose_seq, mask_target_padding, get_attn_weights)


  def handle_class_token(self, input_pose_seq):
    """
    Args:
      input_pose_seq: [src_len, batch_size, model_dim]
    """
    # concatenate extra token for activity prediction as an extra
    # element of the input sequence
    # specialized token is not a skeleton
    _, B, _ = input_pose_seq.size()
    token = self._class_token.squeeze().repeat(1, B, 1)
    input_pose_seq = torch.cat([token, input_pose_seq], axis=0)

    return input_pose_seq

  def handle_copy_query(self, indices, input_pose_seq_):
    """Handles the way queries are generated copying items from the inputs.

    Args:
      indices: A list of tuples len `batch_size`. Each tuple contains has the
        form (input_list, target_list) where input_list contains indices of
        elements in the input to be copy to elements in the target specified by
        target_list.
      input_pose_seq_: Source skeleton sequence [batch_size, src_len, pose_dim].

    Returns:
      A tuple with first elements the decoder input skeletons with shape
      [tgt_len, batch_size, skeleton_dim], and the skeleton embeddings of the 
      input sequence with shape [tgt_len, batch_size, pose_dim].
    """
    batch_size = input_pose_seq_.size()[0]
    decoder_inputs = torch.FloatTensor(
        batch_size,
        self._target_seq_length,
        self._pose_dim
    ).to(self._decoder_pos_encodings.device)
    for i in range(batch_size):
      for j in range(self._target_seq_length):
        src_idx, tgt_idx = indices[i][0][j], indices[i][1][j]
        decoder_inputs[i, tgt_idx] = input_pose_seq_[i, src_idx]
    dec_inputs_encode = self._pose_embedding(decoder_inputs)

    return torch.transpose(decoder_inputs, 0, 1), \
        torch.transpose(dec_inputs_encode, 0, 1)

  def forward_training(self,
                       input_pose_seq_,
                       target_pose_seq_,
                       mask_target_padding,
                       get_attn_weights=False):
    """Compute forward pass for training and non recursive inference.
    Args:
       input_pose_seq_: Source sequence [batch_size, src_len, skeleton_dim].
       target_pose_seq_: Query target sequence [batch_size, tgt_len, skeleton_dim].
       mask_target_padding: Mask for target masking with ones where elements 
          belong to the padding elements of shape [batch_size, tgt_len, skeleton_dim].
       get_attn_weights: Boolean to indicate if attention weights should be returned.
    Returns:
    """
    # 1) Encode the sequence with given pose encoder
    # [batch_size, sequence_length, model_dim]
    input_pose_seq = input_pose_seq_
    target_pose_seq = target_pose_seq_
    if self._pose_embedding is not None:
      input_pose_seq = self._pose_embedding(input_pose_seq)
      target_pose_seq = self._pose_embedding(target_pose_seq)

    # 2) compute the look-ahead mask and the positional encodings
    # [sequence_length, batch_size, model_dim]
    input_pose_seq = torch.transpose(input_pose_seq, 0, 1)
    target_pose_seq = torch.transpose(target_pose_seq, 0, 1)

    def query_copy_fn(indices):
      return self.handle_copy_query(indices, input_pose_seq_)

    # concatenate extra token for activity prediction as an extr element of the 
    # input sequence, i.e. specialized token is not a skeleton
    if self._use_class_token:
      input_pose_seq = self.handle_class_token(input_pose_seq)

    # 3) compute the attention weights using the transformer
    # [target_sequence_length, batch_size, model_dim]
    attn_output, memory, attn_weights, enc_weights, mat = self._transformer(
        input_pose_seq,
        target_pose_seq,
        query_embedding=self._query_embed.weight,
        encoder_position_encodings=self._encoder_pos_encodings,
        decoder_position_encodings=self._decoder_pos_encodings,
        mask_look_ahead=self._mask_look_ahead,
        mask_target_padding=mask_target_padding,
        get_attn_weights=get_attn_weights,
        query_selection_fn=query_copy_fn
    )

    end = self._input_dim if self._input_dim == self._pose_dim else self._pose_dim
    out_sequence = []
    target_pose_seq_ = mat[0] if self._query_selection else \
        torch.transpose(target_pose_seq_, 0, 1)

    # 4) decode sequence with pose decoder. The decoding process is time
    # independent. It means non-autoregressive or parallel decoding.
    # [batch_size, target_sequence_length, pose_dim]
    for l in range(self._num_decoder_layers):
      # [target_seq_length*batch_size, pose_dim]
      out_sequence_ = self._pose_decoder(
          attn_output[l].view(-1, self._model_dim))
      # [target_seq_length, batch_size, pose_dim]
      out_sequence_ = out_sequence_.view(
          self._target_seq_length, -1, self._pose_dim)
      # apply residual connection between target query and predicted pose
      # [tgt_seq_len, batch_size, pose_dim]
      out_sequence_ = out_sequence_ + target_pose_seq_[:, :, 0:end]
      # [batch_size, tgt_seq_len, pose_dim]
      out_sequence_ = torch.transpose(out_sequence_, 0, 1)
      out_sequence.append(out_sequence_)

    if self._predict_activity:
      out_class = self.predict_activity(attn_output, memory)
      return out_sequence, out_class, attn_weights, enc_weights, mat

    return out_sequence, attn_weights, enc_weights, mat

  def predict_activity(self, attn_output, memory):
    """Performs activity prediction either from memory or class token.

    attn_output: Encoder memory. Shape [src_seq_len, batch_size, model_dim].
    """
    # [batch_size, src_len, model_dim]
    in_act = torch.transpose(memory, 0, 1)

    # use a single specialized token for predicting activity
    # the specialized token is in the first element of the sequence
    if self._use_class_token:
      # [batch_size, model_dim]
      token = in_act[:, 0]
      actions = self._action_head(token)
      return [actions]      

    # use all the input sequence attention to predict activity
    # [batch_size, src_len*model_dim]
    in_act = torch.reshape(in_act, (-1, self._action_head_size))
    actions = self._action_head(in_act)
    return [actions]

    #out_class = []
    #for l in range(self._num_decoder_layers):
    #  in_act = torch.transpose(attn_output[l], 0, 1)
    #  in_act = torch.reshape(in_act, (-1, self._action_head_size))
    #  actions = self._action_head(in_act)
    #  out_class.append(actions)
    #return out_class


  def forward_autoregressive(self,
                      input_pose_seq,
                      target_pose_seq=None,
                      mask_target_padding=None,
                      get_attn_weights=False):
    """Compute forward pass for auto-regressive inferece in test time."""
    thisdevice = self._encoder_pos_encodings.device
    # the first query pose is the first in the target
    prev_target = input_pose_seq[:, 0, :]
    # 1) Enconde using the pose embeding
    if self._pose_embedding is not None:
      input_pose_seq = self._pose_embedding(input_pose_seq)
      target_pose_seq = self._pose_embedding(target_pose_seq)
    # [batch_size, 1, model_dim]
    target_seq = input_pose_seq[:, 0:1, :]

    # 2) compute the look-ahead mask and the positional encodings
    # [sequence_length, batch_size, model_dim]
    input_pose_seq = torch.transpose(input_pose_seq, 0, 1)
    target_seq = torch.transpose(target_seq, 0, 1)

    # concatenate extra token for activity prediction as an extra
    if self._use_class_token:
      input_pose_seq = self.handle_class_token(input_pose_seq)

    # 3) use auto recursion to compute the predicted set of tokens    
    memory, enc_attn_weights = self._transformer._encoder(
        input_pose_seq, self._encoder_pos_encodings)

    # get only the first In teory it should only be one target pose at testing
    batch_size = memory.size()[1]
    out_pred_seq = torch.FloatTensor(
        batch_size, self._target_seq_length, self._pose_dim).to(thisdevice)

    for t in range(self._target_seq_length):
      position_encodings = self._pos_decoder(t+1).view(
          t+1, 1, self._model_dim).to(thisdevice)
      mask_look_ahead = torch.from_numpy(
          utils.create_look_ahead_mask(t+1)).to(thisdevice)

      # a list of length n_decoder_layers with elements of 
      # shape [t, batch_size, model_dim]
      out_attn, out_weights = self._transformer._decoder(
          target_seq,
          memory,
          position_encodings,
          mask_look_ahead=mask_look_ahead
      )
      # get only the last predicted token decode it to get the pose and 
      # then encode the pose. shape [1*batch_size, pose_dim]
      # for 8 seeds of evaluation (batch_size)
      pred_pose = self._pose_decoder(
          out_attn[-1][t:(t+1), :, :].view(-1, self._model_dim))
      # apply residual between last target pose and recently generated pose
      if self._pose_dim == self._input_dim:
        pred_pose = pred_pose + prev_target
      else:
        prev_target[:, 0:self._pose_dim] = pred_pose + prev_target[:,0:self._pose_dim]
        pred_pose = prev_target
      prev_target = pred_pose
      out_pred_seq[:, t, :] = pred_pose.view(-1, self._input_dim)[:, 0:self._pose_dim]
      if self._pose_embedding is not None:
        pose_code = self._pose_embedding(pred_pose.unsqueeze(0))
        # [1, batch_size, model_dim]
        pose_code = pose_code.view(-1, batch_size, self._model_dim)
      # [t+1, batch_size, model_dim]
      target_seq = torch.cat([target_seq, pose_code], axis=0)

    target_seq = target_seq[:, 1:, :]
    # 1) the last attention output contains all the necessary sequence; or
    # 2) Use all the memory to predict
    if self._predict_activity:
      actions = self.predict_activity(out_attn, memory)

    if self._predict_activity:
      return [out_pred_seq], [actions[-1]], None, None

    return [out_pred_seq]


def model_factory(params, pose_embedding_fn, pose_decoder_fn):
  init_fn = utils.normal_init_ \
      if params['init_fn'] == 'normal_init' else utils.xavier_init_
  return PoseTransformer(
      pose_dim=params['pose_dim'],
      input_dim=params['input_dim'],
      model_dim=params['model_dim'],
      num_encoder_layers=params['num_encoder_layers'],
      num_decoder_layers=params['num_decoder_layers'],
      num_heads=params['num_heads'],
      dim_ffn=params['dim_ffn'],
      dropout=params['dropout'],
      target_seq_length=params['target_seq_len'],
      source_seq_length=params['source_seq_len'],
      init_fn=init_fn,
      non_autoregressive=params['non_autoregressive'],
      use_query_embedding=params['use_query_embedding'],
      pre_normalization=params['pre_normalization'],
      predict_activity=params['predict_activity'],
      num_activities=params['num_activities'],
      use_memory=params['use_memory'],
      pose_embedding=pose_embedding_fn(params),
      pose_decoder=pose_decoder_fn(params),
      query_selection=params['query_selection'],
      pos_encoding_params=(params['pos_enc_beta'], params['pos_enc_alpha'])
  )


if __name__ == '__main__':
  transformer = PoseTransformer(model_dim=_POSE_DIM, num_heads=6)
  transformer.eval()
  batch_size = 8
  model_dim = 256
  tgt_seq = torch.FloatTensor(batch_size, _TARGET_LENGTH, _POSE_DIM).fill_(1)
  src_seq = torch.FloatTensor(batch_size, _SOURCE_LENGTH-1, _POSE_DIM).fill_(1)

  outputs = transformer(src_seq, tgt_seq)
  print(outputs[-1].size())

