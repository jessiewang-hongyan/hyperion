"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

import torch.nn as nn

from ..torch_model import TorchModel

class NetArch(TorchModel):


    def in_context(self):
        return 0

    def in_dim(self):
        return len(self.in_shape())

    def out_dim(self):
        return len(self.in_shape())

    def in_shape(self):
        raise NotImplementedError()

    def out_shape(self, in_shape=None):
        raise NotImplementedError()

