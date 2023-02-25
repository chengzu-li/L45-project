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


class Edge_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='max'):
        """
        This is adopted from the GCN implemented by MessagePassing network in torch_geometric
        Args:
            in_channels:
            out_channels:
            aggr:
        """
        super(Edge_GCNConv, self).__init__(aggr=aggr)

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


class Novel_Edge_GCN(nn.Module):
    def __init__(self, args, num_layers, num_in_features, num_hidden_features, num_classes, name, aggr="max"):
        """

        Args:
            num_in_features:
            num_hidden_features:
            num_classes:
            name:
            aggr: Possible choices for aggr: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
        """
        super(Novel_Edge_GCN, self).__init__()

        self.args = args
        self.name = name
        self.num_layers = num_layers
        self.norm = []

        self.layers = [Edge_GCNConv(num_in_features, num_hidden_features, aggr=aggr)]
        self.layers += [Edge_GCNConv(num_hidden_features, num_hidden_features, aggr=aggr) for _ in range(num_layers-1)]
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
                node_i = edge_index_tmp[0][index]
                node_j = edge_index_tmp[1][index]

                neighbor_i = [n for n in tmp[int(node_i)]]
                neighbor_j = [n for n in tmp[int(node_j)]]

                sg_i = tmp.subgraph(neighbor_i)
                sg_j = tmp.subgraph(neighbor_j)

                if self.args.similar_measure == "edge":
                    edge_i = set(sg_i.edges)
                    edge_j = set(sg_j.edges)
                    norm += [len(edge_i.intersection(edge_j)) / len(edge_i.union(edge_j))]
                elif self.args.similar_measure == "edit_dist":
                    # edit_dist means the graph similarity measures based on graph edit distance
                    # TODO: come up with a graph similarity measure
                    raise NotImplementedError(f"Edit Distance measure not implemented. ")
                    pass

            self.norm = torch.tensor(norm)
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index, self.norm)
            x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)