import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        self.arch = 'normal'
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        torch.nn.init.eye_(self.w)
        self.w.requires_grad = True

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):

        if self.dropout is not None:
            inputs = self.dropout(inputs)
        outputs = torch.mm(adj.cuda(), torch.mm(inputs, self.w))
        if self.relu is not None:
            outputs = self.relu(outputs)
        # residual结构下， 每层输出都乘上输入
        if self.arch == 'residual':
            outputs = outputs * inputs
        return outputs


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_layers):
        super().__init__()
        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.arch = conv.arch
        self.add_module('conv_last', conv)
        layers.append(conv)

        self.layers = layers


    def forward(self, x, edges):
        adj = edges.cuda()
        if self.arch == 'residual':
            for conv in self.layers:
                x = conv(x, adj)
            out = x
        else:
            input_data = x
            for conv in self.layers:
                x = conv(x, adj)
            out = x * input_data
        out = F.normalize(out)
        out = F.relu(out)
        return out

