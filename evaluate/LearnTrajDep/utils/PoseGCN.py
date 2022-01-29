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
"""Graph Convolutional Neural Network implementation.

Code adapted from [1].

[1] https://github.com/wei-mao-2019/HisRepItself
[2] https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
"""


import os
import sys
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils


class GraphConvolution(nn.Module):
  """Implements graph convolutions."""

  def __init__(self, in_features, out_features, output_nodes=48, bias=False):
    """Constructor.

    The graph convolutions can be defined as \sigma(AxHxW), where A is the 
    adjacency matrix, H is the feature representation from previous layer
    and W is the wegith of the current layer. The dimensions of such martices
    A\in R^{NxN}, H\in R^{NxM} and W\in R^{MxO} where
      - N is the number of nodes
      - M is the number of input features per node
      - O is the number of output features per node

    Args:
      in_features: Number of input features per node.
      out_features: Number of output features per node.
      output_nodes: Number of nodes in the graph.
    """
    super(GraphConvolution, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self._output_nodes = output_nodes
    # W\in R^{MxO}
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    # A\in R^{NxN}
    self.att = Parameter(torch.FloatTensor(output_nodes, output_nodes))
    if bias:
        self.bias = Parameter(torch.FloatTensor(out_features))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    self.att.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)

  def forward(self, x):
    """Forward pass.

    Args:
      x: [batch_size, n_nodes, input_features]
    Returns:
      Feature representation computed from inputs. 
      Shape is [batch_size, n_nodes, output_features].
    """
    # [batch_size, input_dim, output_features]
    # HxW = {NxM}x{MxO} = {NxO}
    support = torch.matmul(x, self.weight)
    # [batch_size, n_nodes, output_features]
    # = {NxN}x{NxO} = {NxO}
    output = torch.matmul(self.att, support)

    if self.bias is not None:
        return output + self.bias
    else:
        return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
             + str(self.in_features) + ' -> ' \
             + str(self.out_features) + ')'


class GC_Block(nn.Module):
  """Residual block with graph convolutions.

  The implementation uses the same number of input features for outputs.
  """

  def __init__(self, in_features, p_dropout, output_nodes=48, bias=False):
    """Constructor.

    Args:
      in_features: Number of input and output features.
      p_dropout: Dropout used in the layers.
      output_nodes: Number  of output nodes in the graph.
    """
    super(GC_Block, self).__init__()
    self.in_features = in_features
    self.out_features = in_features

    self.gc1 = GraphConvolution(
        in_features, in_features,
        output_nodes=output_nodes, 
        bias=bias
    )
    self.bn1 = nn.BatchNorm1d(output_nodes * in_features)
    self.gc2 = GraphConvolution(
        in_features, in_features, 
        output_nodes=output_nodes, 
        bias=bias
    )
    self.bn2 = nn.BatchNorm1d(output_nodes * in_features)

    self.do = nn.Dropout(p_dropout)
    self.act_f = nn.Tanh()

  def forward(self, x):
    """Forward pass of the residual module"""
    y = self.gc1(x)
    b, n, f = y.shape
    y = self.bn1(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)

    y = self.gc2(y)
    b, n, f = y.shape
    y = self.bn2(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)

    return y + x

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
             + str(self.out_features) + ')'


class PoseGCN(nn.Module):
  def __init__(self,
               input_features=128,
               output_features=3,
               model_dim=128,
               output_nodes=21,
               p_dropout=0.1,
               num_stage=1):
    """Constructor.

    Args:
      input_feature: num of input feature of the graph nodes.
      model_dim: num of hidden features of the generated embeddings.
      p_dropout: dropout probability
      num_stage: number of residual blocks in the network.
      output_nodes: number of nodes in graph.
    """
    super(PoseGCN, self).__init__()
    self.num_stage = num_stage
    self._n_nodes = output_nodes
    self._model_dim = model_dim
    self._output_features = output_features
    self._hidden_dim = 512

    self._front = nn.Sequential(
        nn.Linear(model_dim, output_nodes*self._hidden_dim),
        nn.Dropout(p_dropout)
    )
    utils.weight_init(self._front, init_fn_=utils.xavier_init_)                            

    self.gc1 = GraphConvolution(
        self._hidden_dim, 
        self._hidden_dim, 
        output_nodes=output_nodes
    )
    self.bn1 = nn.BatchNorm1d(output_nodes * self._hidden_dim)

    self.gcbs = []
    for i in range(num_stage):
      self.gcbs.append(GC_Block(
          self._hidden_dim, 
          p_dropout=p_dropout, 
          output_nodes=output_nodes)
      )

    self.gcbs = nn.ModuleList(self.gcbs)

    self.gc7 = GraphConvolution(
        self._hidden_dim, 
        output_features,
        output_nodes=output_nodes
    )
    self.do = nn.Dropout(p_dropout)
    self.act_f = nn.Tanh()

    gcn_params = filter(lambda p: p.requires_grad, self.parameters())
    nparams = sum([np.prod(p.size()) for p in gcn_params])
    print('[INFO] ({}) GCN has {} params!'.format(self.__class__.__name__, nparams))


  def preprocess(self, x):
    if len(x.size()) < 3:
      _, D = x.size()
      # seq_len, batch_size, input_dim
      x = x.view(self._seq_len, -1, D)
      # [batch_size, seq_len, input_dim]
      x = torch.transpose(x, 0, 1) 
      # [batch_size, input_dim, seq_len]
      x = torch.transpose(x, 1, 2)
      return x

    return x

  def postprocess(self, y):
    """Flattents the input tensor.
    Args:
      y: Input tensor of shape [batch_size, n_nodes, output_features].
    """
    y = y.view(-1, self._n_nodes*self._output_features)
    return y

  def forward(self, x):
    """Forward pass of network.

    Args:
      x: [batch_size, model_dim]. 
    """
    # [batch_size, model_dim*n_nodes]
    x = self._front(x)
    x = x.view(-1, self._n_nodes, self._hidden_dim)

    # [batch_size, n_joints, model_dim]
    y = self.gc1(x)
    b, n, f = y.shape
    y = self.bn1(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)

    for i in range(self.num_stage):
        y = self.gcbs[i](y)

    # [batch_size, n_joints, output_features]
    y = self.gc7(y)
    # y = y + x

    # [seq_len*batch_size, input_dim]
    y = self.postprocess(y)

    return y


class SimpleEncoder(nn.Module):
  def __init__(self,
               n_nodes=63,
               input_features=1,
               model_dim=128,
               p_dropout=0.1):
    """Constructor.

    Args:
      input_dim: Dimension of the input vector. This will be equivalent to
        the number of nodes in the graph, each node with 1 feature each.
      model_dim: Dimension of the output vector to produce.
      p_dropout: Dropout to be applied for regularization.
    """
    super(SimpleEncoder, self).__init__() 
    #The graph convolutions can be defined as \sigma(AxHxW), where A is the 
    #A\in R^{NxN} x H\in R^{NxM} x  W\in R ^{MxO}
    self._input_features = input_features
    self._output_nodes = n_nodes
    self._hidden_dim = 512
    self._model_dim = model_dim
    self._num_stage = 1

    print('[INFO] ({}) Hidden dimension: {}!'.format(
        self.__class__.__name__, self._hidden_dim))
    self.gc1 = GraphConvolution(
        in_features=self._input_features, 
        out_features=self._hidden_dim, 
        output_nodes=self._output_nodes
    )
    self.bn1 = nn.BatchNorm1d(self._output_nodes*self._hidden_dim)
    self.gc2 = GraphConvolution(
        in_features=self._hidden_dim, 
        out_features=model_dim,
        output_nodes=self._output_nodes
    )

    self.gcbs = []
    for i in range(self._num_stage):
      self.gcbs.append(GC_Block(
          self._hidden_dim, 
          p_dropout=p_dropout, 
          output_nodes=self._output_nodes)
      )
    self.gcbs = nn.ModuleList(self.gcbs)

    self.do = nn.Dropout(p_dropout)
    self.act_f = nn.Tanh()

    self._back = nn.Sequential(
        nn.Linear(model_dim*self._output_nodes, model_dim),
        nn.Dropout(p_dropout)
    )
    utils.weight_init(self._back, init_fn_=utils.xavier_init_)                            

    gcn_params = filter(lambda p: p.requires_grad, self.parameters())
    nparams = sum([np.prod(p.size()) for p in gcn_params])
    print('[INFO] ({}) GCN has {} params!'.format(self.__class__.__name__, nparams))

  def forward(self, x):
    """Forward pass of network.

    Args:
      x: [batch_size, n_poses, pose_dim/input_dim]. 
    """
    B, S, D = x.size()
    # [batch_size, n_joints, model_dim]
    y = self.gc1(x.view(-1, self._output_nodes, self._input_features))
    b, n, f = y.shape
    y = self.bn1(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)

    for i in range(self._num_stage):
        y = self.gcbs[i](y)

    # [batch_size, n_joints, model_dim]
    y = self.gc2(y)

    # [batch_size, model_dim]
    y = self._back(y.view(-1, self._model_dim*self._output_nodes))

    # [batch_size, n_poses, model_dim]
    y = y.view(B, S, self._model_dim)

    return y


def test_decoder():
  seq_len = 25
  input_size = 63
  model_dim = 128
  dropout = 0.3
  n_stages = 2
  output_nodes = 21 

  joint_dof = 1
  n_joints = model_dim
  layer = GraphConvolution(
      in_features=joint_dof, 
      out_features=model_dim, 
      output_nodes=n_joints
  )

  X = torch.FloatTensor(10, n_joints, joint_dof)
  print(layer(X).size())

  gcn = PoseGCN(
      input_features=model_dim,
      output_features=3,
      model_dim=model_dim,
      output_nodes=output_nodes, 
      p_dropout=0.1, 
      num_stage=2
  )

  X = torch.FloatTensor(10*seq_len, model_dim)
  print(gcn(X).size())


def test_encoder():
  input_size = 63
  model_dim = 128
  dropout = 0.3
  n_stages = 2
  output_nodes = 21 
  dof = 9

  encoder = SimpleEncoder(
      n_nodes=output_nodes,  
      model_dim=model_dim,
      input_features=dof,
      p_dropout=0.1
  )
  X = torch.FloatTensor(10, 25, output_nodes*dof)

  print(encoder(X).size())


if __name__ == '__main__':
  #test_decoder()
  test_encoder()




