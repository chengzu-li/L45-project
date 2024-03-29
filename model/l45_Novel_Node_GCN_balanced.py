import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx


from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_scatter


from torch import Tensor

from torch_geometric.nn import MessagePassing, GCNConv, DenseGCNConv, GINConv, GraphConv
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, convert

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    get_type_hints,
)


class Node_GCNConv_balanced(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='max'):
        """
        This is adopted from the GCN implemented by MessagePassing network in torch_geometric
        Args:
            in_channels:
            out_channels:
            aggr:
        """
        super(Node_GCNConv_balanced, self).__init__()

        self.aggr = aggr
        self.combine_aggr = nn.Linear(out_channels*4, out_channels, bias=False)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.beta = nn.Parameter(torch.Tensor([1]))

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.

        norm = (self.intersection/(self.intersection + self.alpha * self.diff_ij + self.beta * self.diff_ji))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out
    def aggregate(self, inputs, index):
        if self.aggr == 'max':
            return torch_scatter.scatter_max(inputs, index, dim=self.node_dim)[0]
        elif self.aggr == 'mean':
            return torch_scatter.scatter_mean(inputs, index, dim=self.node_dim)
        elif self.aggr == 'add':
            return torch_scatter.scatter_add(inputs, index, dim=self.node_dim)
        elif self.aggr == 'min':
            return torch_scatter.scatter_min(inputs, index, dim=self.node_dim)[0]
        elif self.aggr == 'div':
            return torch_scatter.scatter_div(inputs, index, dim=self.node_dim)
        elif self.aggr == 'mul':
            return torch_scatter.scatter_mul(inputs, index, dim=self.node_dim)
        elif self.aggr == 'std':
            return torch_scatter.scatter_std(inputs, index, dim=self.node_dim)
        elif self.aggr == 'multi':
            mean = torch_scatter.scatter_mean(inputs, index, dim=self.node_dim)
            add = torch_scatter.scatter_add(inputs, index, dim=self.node_dim)
            # std = torch_scatter.scatter_std(inputs, index, dim=self.node_dim)
            min = torch_scatter.scatter_min(inputs, index, dim=self.node_dim)[0]
            max = torch_scatter.scatter_max(inputs, index, dim=self.node_dim)[0]
            return self.combine_aggr(torch.cat((mean, add, min, max), dim=-1))
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class Novel_Node_GCN_balanced(nn.Module):
    def __init__(self, args, num_layers, num_in_features, num_hidden_features, num_classes, name, aggr="max"):
        """

        Args:
            num_in_features:
            num_hidden_features:
            num_classes:
            name:
            aggr: Possible choices for aggr: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
        """
        super(Novel_Node_GCN_balanced, self).__init__()

        self.args = args
        self.name = name
        self.num_layers = num_layers
        self.intersection = []


        self.layers = [Node_GCNConv_balanced(num_in_features, num_hidden_features, aggr=aggr)]
        self.layers += [Node_GCNConv_balanced(num_hidden_features, num_hidden_features, aggr=aggr) for _ in range(num_layers-1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x, edge_index):
        if self.intersection == []:
            edge_index_tmp, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            data = Data(edge_index=edge_index_tmp, num_nodes=x.size(0))
            tmp = to_networkx(data, to_undirected=True)
            intersection = []
            diff_ij = []
            diff_ji = []
            for index in range(len(edge_index_tmp[0])):
                s1 = set(tmp[edge_index_tmp[0][index].item()])
                s2 = set(tmp[edge_index_tmp[1][index].item()])
                intersection += [len(s1.intersection(s2))]
                diff_ij += [len(s1.difference(s2))]
                diff_ji += [len(s2.difference(s1))]
            self.intersection = torch.tensor(intersection)
            self.diff_ij = torch.tensor(diff_ij)
            self.diff_ji = torch.tensor(diff_ji)

            for i in range(self.num_layers):
                self.layers[i].intersection = self.intersection
                self.layers[i].diff_ij = self.diff_ij
                self.layers[i].diff_ji = self.diff_ji
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)


