#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[1])
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(ckpt_path, file_name[0])
        torch.save(state, file_path)

def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
  classname = layer.__class__.__name__
  if classname.find('Linear')!=-1:
    print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
    nn.init.xavier_uniform_(layer.weight.data)
    # nninit.xavier_normal(layer.bias.data)
    if norm_bias:
      layer.bias.data.normal_(0, 0.05)
    else:
      layer.bias.data.zero_()

def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
  """Intialization of layers with normal distribution with mean and bias"""
  classname = layer.__class__.__name__
  # Only use the convolutional layers of the module
  #if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
  if classname.find('Linear') != -1:
    print('[INFO] (normal_init) Initializing layer {}'.format(classname))
    layer.weight.data.normal_(mean_, sd_)
    if norm_bias:
      layer.bias.data.normal_(bias, 0.05)
    else:
      layer.bias.data.fill_(bias)

def weight_init(
    module, 
    mean_=0, 
    sd_=0.004, 
    bias=0.0, 
    norm_bias=False, 
    init_fn_=normal_init_):
  """Initialization of layers with normal distribution"""
  moduleclass = module.__class__.__name__
  try:
    for layer in module:
      if layer.__class__.__name__ == 'Sequential':
        for l in layer:
          init_fn_(l, mean_, sd_, bias, norm_bias)
      else:
        init_fn_(layer, mean_, sd_, bias, norm_bias)
  except TypeError:
    init_fn_(module, mean_, sd_, bias, norm_bias)

def create_look_ahead_mask(seq_length, is_nonautoregressive=False):
  """Generates a binary mask to prevent to use future context in a sequence."""
  if is_nonautoregressive:
    return np.zeros((seq_length, seq_length), dtype=np.float32)
  x = np.ones((seq_length, seq_length), dtype=np.float32)
  mask = np.triu(x, 1).astype(np.float32)
  return mask  # (seq_len, seq_len)

