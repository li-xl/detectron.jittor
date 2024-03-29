# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import jittor as jt 
from jittor import nn,Module,init
from detectron.config import cfg
from detectron.layers import Conv2d
from detectron.modeling.poolers import Pooler


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn

def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
    return nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def make_conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_gn=False,
    use_relu=False,
    kaiming_init=True
):
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True
    )
    if kaiming_init:
        init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        init.gauss_(conv.weight, std=0.01)
    if not use_gn:
        init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU())
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv

def make_fc(dim_in, hidden_dim, use_gn=False):

    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    init.kaiming_uniform_(fc.weight, a=1)
    init.constant_(fc.bias, 0)
    return fc


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )

        init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.Relu())
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


def make_conv(
    in_channels,
    out_channels,
    kernel_size,
    dilation=1,
    stride=1,
    use_gn=False,
    use_relu=False,
    kaiming_init=True
):
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation * (kernel_size - 1) // 2,
        dilation=dilation,
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        init.gauss_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU())
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_conv1x1(
    in_channels, 
    out_channels, 
    dilation=1, 
    stride=1, 
    use_gn=False,
    use_relu=False,
    kaiming_init=True
):
    conv = Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        stride=stride, 
        padding=0, 
        dilation=dilation, 
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        init.gauss_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU())
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv

class SpatialAttention(Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        try:
            avg_out = jt.mean(x, dim=1, keepdims=True)
            max_out = jt.max(x, dim=1, keepdims=True)
            scale = jt.contrib.concat([avg_out, max_out], dim=1)
            scale = self.conv(scale)
            out = x * self.sigmoid(scale)
        except Exception as e:
            print(e)
            out = x

        return out


class Hsigmoid(Module):
    def __init__(self):
        super(Hsigmoid, self).__init__()

    def execute(self, x):
        return nn.relu6(x + 3) / 6.


class SEModule(Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv(channel, channel // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv(channel // reduction, channel , kernel_size=1,
                             padding=0)
        self.hsigmoid = Hsigmoid()

    def execute(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hsigmoid(x)
        return input * x