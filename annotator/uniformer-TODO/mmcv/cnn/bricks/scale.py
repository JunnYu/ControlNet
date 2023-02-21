# Copyright (c) OpenMMLab. All rights reserved.
import paddle
import paddle.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(paddle.to_tensor(scale, dtype=paddle.float))

    def forward(self, x):
        return x * self.scale
