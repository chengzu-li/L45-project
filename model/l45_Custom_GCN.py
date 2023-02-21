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

# For aggregation function: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
class Customize_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, agg='max'):
        """
        This is adopted from the GCN implemented by MessagePassing network in torch_geometric
        Args:
            in_channels:
            out_channels:
            aggr:
        """
        super(Customize_GCNConv, self).__init__(aggr=agg)

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
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class Customize_GNN(MessagePassing):
    def __init__(self, in_channels, out_channels, agg='max'):
        """
        This is adopted from the GCN implemented by MessagePassing network in torch_geometric
        Args:
            in_channels:
            out_channels:
            aggr:
        """
        super(Customize_GNN, self).__init__(aggr=agg)

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, norm):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.


        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class Customize_GCN(nn.Module):
    def __init__(self, num_layers, num_in_features, num_hidden_features, num_classes, name, agg="max"):
        """

        Args:
            num_in_features:
            num_hidden_features:
            num_classes:
            name:
            aggr: Possible choices for aggr: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
        """
        super(Customize_GCN, self).__init__()

        self.name = name
        self.num_layers = num_layers
        self.norm = []

        self.layers = [Customize_GNN(num_in_features, num_hidden_features, agg=agg)]
        self.layers += [Customize_GNN(num_hidden_features, num_hidden_features, agg=agg) for _ in range(num_layers-1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x, edge_index):
        if self.norm == []:
            edge_index_tmp, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            data = Data(edge_index=edge_index_tmp, num_nodes=x.size(0))
            tmp = to_networkx(data, to_undirected=True)
            norm = []
            for index in range(len(edge_index_tmp[0])):
                s1 = set(tmp[edge_index_tmp[0][index].item()])
                s2 = set(tmp[edge_index_tmp[1][index].item()])
                norm += [len(s1.intersection(s2)) / len(s1.union(s2))]
            # row, col = edge_index_tmp
            # deg = degree(col, x.size(0), dtype=x.dtype)
            # deg_inv_sqrt = deg.pow(-0.5)
            # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            # norm_deg = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            # self.norm = torch.tensor(norm) * norm_deg
            self.norm = torch.tensor(norm)
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index, self.norm)
            x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)


