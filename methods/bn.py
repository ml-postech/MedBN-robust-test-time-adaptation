# The code is adapted from https://github.com/bethgelab/robustness.git
# Below is the original license:
# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

""" Batch norm variants"""

import torch
from torch import nn
from torch.nn import functional as F

from conf import cfg

class RBN(nn.Module):
    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = RBN(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(RBN.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior, layer_index):
        replace_mods = RBN.find_bns(model, prior)
        # print(replace_mods)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for index, (parent, name, child) in enumerate(replace_mods):
            if index >= len(replace_mods) - layer_index:
                child.prior = 1.0
            setattr(parent, name, child)
        return model

    def __init__(
        self,
        layer,
        prior,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    ):
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.weight = layer.weight
        self.bias = layer.bias
        self.eps = layer.eps
        self.num_features = layer.num_features
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.prior = prior

        self.bn_stat = cfg.MODEL.BN_STAT

    def forward(self, input):
        prior = self.prior
        if self.bn_stat == "mean":
            b_var, b_mean = torch.var_mean(
                input, dim=[0, 2, 3], unbiased=False, keepdim=False
            )  # (C,)
        elif self.bn_stat == "median":
            b_mean = self.find_median(input)
            b_var = self.find_med_var(input, b_mean)

        mean = (1 - prior) * b_mean + prior * self.layer.running_mean
        var = (1 - prior) * b_var + prior * self.layer.running_var

        input = (input - mean[None, :, None, None]) / (
            torch.sqrt(var[None, :, None, None] + self.eps)
        )

        if self.affine:
            input = (
                input * self.weight[None, :, None, None]
                + self.bias[None, :, None, None]
            )
        return input

    def find_median(self, input_data):
        shape = input_data.shape
        input2 = input_data.transpose(1, 0)
        input3 = input2.reshape(shape[1], -1)
        median = input3.median(1)[0]
        return median

    def find_med_var(self, input, median):
        err = input - median[None, :, None, None]
        return err.pow(2).sum(dim=0).sum(dim=1).sum(dim=1) / input.shape[0]

    def find_mad(self, input_data, median):
        input_norm = torch.abs(input_data - median[None, :, None, None])
        mad = self.find_median(input_norm)
        return mad
