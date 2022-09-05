# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

import ipdb


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(search, kernel):
    """ depthwise cross correlation
    Args:
        search: (b, c, 29, 29)
        kernel: (b, c, 5, 5)
    Return:
        out: (b, c, 25, 25)
    """
    batch = kernel.size(0)
    channel = kernel.size(1)    # channel = 256
    # search: (1, batch*256, 29, 29)
    search = search.view(1, batch*channel, search.size(2), search.size(3))
    # kernel: (batch*256, 1, 5, 5)
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    # out: (1, batch*256, 25, 25)
    out = F.conv2d(search, kernel, groups=batch*channel)
    # out: (batch, 256, 25, 25)
    out = out.view(batch, channel, out.size(2), out.size(3))

    return out
