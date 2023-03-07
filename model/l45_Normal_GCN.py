import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing, GCNConv, DenseGCNConv, GINConv, GraphConv
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, convert


# For aggregation function: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
class Customize_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='max'):
        """
        This is adopted from the GCN implemented by MessagePassing network in torch_geometric
        Args:
            in_channels:
            out_channels:
            aggr:
        """
        super(Customize_GCNConv, self).__init__()

        self.aggr = aggr
        self.combine_aggr = nn.Linear(out_channels*4, out_channels, bias=False)

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


class Customize_GCN(nn.Module):
    def __init__(self, args, num_layers, num_in_features, num_hidden_features, num_classes, name, aggr="max"):
        """

        Args:
            num_in_features:
            num_hidden_features:
            num_classes:
            name:
            aggr: Possible choices for aggr: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
        """
        super(Customize_GCN, self).__init__()

        self.args = args
        self.name = name
        self.num_layers = num_layers

        self.layers = [Customize_GCNConv(num_in_features, num_hidden_features, aggr=aggr)]
        self.layers += [Customize_GCNConv(num_hidden_features, num_hidden_features, aggr=aggr) for _ in range(num_layers-1)]
        self.layers = nn.ModuleList(self.layers)

        # linear layers
        self.linear = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)


