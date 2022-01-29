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

"""Implementation of the 2D positional encodings used in [1].

Position encodings gives a signature to each pixel in the image by a set 
of sine frequecies computed with a 2D sine function.

[1] https://arxiv.org/abs/2005.12872
[2] https://arxiv.org/pdf/1706.03762.pdf
"""


import numpy as np
import math
import torch
from torch import nn


class PositionEncodings2D(object):
  """Implementation of 2d masked position encodings as a NN layer.

  This is a more general version of the position embedding, very similar 
  to the one used by the Attention is all you need paper, but generalized 
  to work on images as used in [1].
  """
  def __init__(
      self, 
      num_pos_feats=64, 
      temperature=10000, 
      normalize=False, 
      scale=None):
    """Constructs position embeding layer.

    Args:
      num_pos_feats: An integer for the depth of the encoding signature per 
        pixel for each axis `x` and `y`.
      temperature: Value of the exponential temperature.
      normalize: Bool indicating if the encodings shuld be normalized by number
        of pixels in each image row.
      scale: Use for scaling factor. Normally None is used for  2*pi scaling.
    """
    super().__init__()
    self._num_pos_feats = num_pos_feats
    self._temperature = temperature
    self._normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self._scale = scale

  def __call__(self, mask):
    """Generates the positional encoding given image boolean mask.

    Args:
      mask: Boolean tensor of shape [batch_size, width, height] with ones 
        in pixels that belong to the padding and zero in valid pixels.

    Returns:
      Sine position encodings. Shape [batch_size, num_pos_feats*2, width, height]
    """
    # the positional encodings are generated for valid pixels hence 
    # we need to take the negation of the boolean mask
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if self._normalize:
      eps = 1e-6
      y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self._scale
      x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self._scale

    dim_t = torch.arange(
        self._num_pos_feats, dtype=torch.float32)
    dim_t = self._temperature ** (2 * (dim_t // 2) / self._num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                         pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), 
                         pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos


class PositionEncodings1D(object):
  """Positional encodings for `1D` sequences.

  Implements the following equations:

  PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
  PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_model})

  Where d_model is the number of positional features. Also known as the
  depth of the positional encodings. These are the positional encodings
  proposed in [2].
  """

  def __init__(self, num_pos_feats=512, temperature=10000, alpha=1):
    self._num_pos_feats = num_pos_feats
    self._temperature = temperature
    self._alpha = alpha

  def __call__(self, seq_length):
    angle_rads = self.get_angles(
        np.arange(seq_length)[:, np.newaxis],
        np.arange(self._num_pos_feats)[np.newaxis, :]
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = pos_encoding.astype(np.float32)

    return torch.from_numpy(pos_encoding)

  def get_angles(self, pos, i):
    angle_rates = 1 / np.power(
        self._temperature, (2 * (i//2)) / np.float32(self._num_pos_feats))
    return self._alpha*pos * angle_rates


def visualize_2d_encodings():
  import cv2
  import numpy as np
  import matplotlib.pyplot as pplt

  # Create a mask where pixels are all valid
  mask = torch.BoolTensor(1, 32, 32).fill_(False)
  # position encodigns with a signature of depth per pixel
  # the efective pixel signature is num_pos_feats*2 (128 for each axis)
  pos_encodings_gen = PositionEncodings2D(num_pos_feats=128, normalize=True)

  encodings = pos_encodings_gen(mask).numpy()
  print('Shape of encodings', encodings.shape)
  # visualize the first frequency channel for x and y
  y_encodings = encodings[0,0, :, :]
  x_encodings = encodings[0,128, : ,:]

  pplt.matshow(x_encodings, cmap=pplt.get_cmap('jet'))
  pplt.matshow(y_encodings, cmap=pplt.get_cmap('jet'))
  pplt.show()


def visualize_1d_encodings():
  import matplotlib.pyplot as plt
  pos_encoder_gen = PositionEncodings1D()
  
  pos_encoding = pos_encoder_gen(50).numpy()
  print(pos_encoding.shape)

  plt.pcolormesh(pos_encoding[0], cmap='RdBu')
  plt.xlabel('Depth')
  plt.xlim((0, 512))
  plt.ylabel('position in sequence')
  plt.colorbar()
  plt.show()

if __name__ == "__main__":
  visualize_2d_encodings()
#  visualize_1d_encodings()

