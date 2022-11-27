from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing
from geometric_linear import *
from torch_geometric.utils import degree

from torch_geometric.nn.inits import reset

'''
MultiMaskConv
Adapted from the source code https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/pna_conv.py
'''

class MMAConv(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers: (list of str): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, dropout, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):

        kwargs.setdefault('aggr', None)
        super(MMAConv, self).__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        #print("edge_dim:", edge_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.dropout = dropout

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        #self.new_input = self.message()
        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        # --------------> Masked (Learnable) Sum Aggregation <-------------------- #
        self.sum_pre_nns = ModuleList()
        self.sum2_pre_nns = ModuleList()
        self.sum3_pre_nns = ModuleList()
        self.sum4_pre_nns = ModuleList()

        # --------------> Masked (Learnable) Mean Aggregation <-------------------- #
        self.mean_pre_nns = ModuleList()
        self.mean2_pre_nns = ModuleList()
        self.mean3_pre_nns = ModuleList()
        self.mean4_pre_nns = ModuleList()

        # --------------> Masked (Learnable) Min Aggregation <-------------------- #
        self.min_pre_nns = ModuleList()
        self.min2_pre_nns = ModuleList()
        self.min3_pre_nns = ModuleList()
        self.min4_pre_nns = ModuleList()

        # --------------> Masked (Learnable) Max Aggregation <-------------------- #
        self.max_pre_nns = ModuleList()
        self.max2_pre_nns = ModuleList()
        self.max3_pre_nns = ModuleList()
        self.max4_pre_nns = ModuleList()

        # --------------> General Parameter for GNN <-------------------- #
        self.post_nns = ModuleList()
        for _ in range(towers):

            # --------------> Masked (Learnable) Sum Aggregation <-------------------- #
            modules = [Sum_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Sum_Linear(self.F_in, self.F_in)]
            self.sum_pre_nns.append(Sequential(*modules))

            modules = [Sum2_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Sum2_Linear(self.F_in, self.F_in)]
            self.sum2_pre_nns.append(Sequential(*modules))

            modules = [Sum3_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Sum3_Linear(self.F_in, self.F_in)]
            self.sum3_pre_nns.append(Sequential(*modules))

            modules = [Sum4_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Sum4_Linear(self.F_in, self.F_in)]
            self.sum4_pre_nns.append(Sequential(*modules))

            # --------------> Masked (Learnable) Mean Aggregation <-------------------- #
            modules = [Mean_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Mean_Linear(self.F_in, self.F_in)]
            self.mean_pre_nns.append(Sequential(*modules))

            modules = [Mean2_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Mean2_Linear(self.F_in, self.F_in)]
            self.mean2_pre_nns.append(Sequential(*modules))

            modules = [Mean3_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Mean3_Linear(self.F_in, self.F_in)]
            self.mean3_pre_nns.append(Sequential(*modules))

            modules = [Mean4_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Mean4_Linear(self.F_in, self.F_in)]
            self.mean4_pre_nns.append(Sequential(*modules))

            # --------------> Masked (Learnable) Min Aggregation <-------------------- #
            modules = [Min_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Min_Linear(self.F_in, self.F_in)]
            self.min_pre_nns.append(Sequential(*modules))

            modules = [Min2_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Min2_Linear(self.F_in, self.F_in)]
            self.min2_pre_nns.append(Sequential(*modules))

            modules = [Min3_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Min3_Linear(self.F_in, self.F_in)]
            self.min3_pre_nns.append(Sequential(*modules))

            modules = [Min4_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Min4_Linear(self.F_in, self.F_in)]
            self.min4_pre_nns.append(Sequential(*modules))

            # --------------> Masked (Learnable) Max Aggregation <-------------------- #
            modules = [Max_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Max_Linear(self.F_in, self.F_in)]
            self.max_pre_nns.append(Sequential(*modules))

            modules = [Max2_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Max2_Linear(self.F_in, self.F_in)]
            self.max2_pre_nns.append(Sequential(*modules))

            modules = [Max3_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Max3_Linear(self.F_in, self.F_in)]
            self.max3_pre_nns.append(Sequential(*modules))

            modules = [Max4_Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Max4_Linear(self.F_in, self.F_in)]
            self.max4_pre_nns.append(Sequential(*modules))

            # --------------> General Parameter for GNN <-------------------- #
            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
                
        for nn in self.sum_pre_nns:
            reset(nn)
        for nn in self.sum2_pre_nns:
            reset(nn)
        for nn in self.sum3_pre_nns:
            reset(nn)
        for nn in self.sum4_pre_nns:
            reset(nn)
        for nn in self.mean_pre_nns:
            reset(nn)
        for nn in self.mean2_pre_nns:
            reset(nn)
        for nn in self.mean3_pre_nns:
            reset(nn)
        for nn in self.mean4_pre_nns:
            reset(nn)
        for nn in self.min_pre_nns:
            reset(nn)
        for nn in self.min2_pre_nns:
            reset(nn)
        for nn in self.min3_pre_nns:
            reset(nn)
        for nn in self.min4_pre_nns:
            reset(nn)
        for nn in self.max_pre_nns:
            reset(nn)
        for nn in self.max2_pre_nns:
            reset(nn)
        for nn in self.max3_pre_nns:
            reset(nn)
        for nn in self.max4_pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.sum_pre_nns)]
            elif aggregator == 'sum2':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.sum2_pre_nns)]
            elif aggregator == 'sum3':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.sum3_pre_nns)]
            elif aggregator == 'sum4':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.sum4_pre_nns)]
            elif aggregator == 'mean':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.mean_pre_nns)]
            elif aggregator == 'mean2':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.mean2_pre_nns)]
            elif aggregator == 'mean3':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.mean3_pre_nns)]
            elif aggregator == 'mean4':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.mean4_pre_nns)]
            elif aggregator == 'min':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.min_pre_nns)]
            elif aggregator == 'min2':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.min2_pre_nns)]
            elif aggregator == 'min3':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.min3_pre_nns)]
            elif aggregator == 'min4':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.min4_pre_nns)]
            elif aggregator == 'max':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.max_pre_nns)]
            elif aggregator == 'max2':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.max2_pre_nns)]
            elif aggregator == 'max3':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.max3_pre_nns)]
            elif aggregator == 'max4':
                hs = [nn(h[:, i]) for i, nn in enumerate(self.max4_pre_nns)]
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')

        hs = torch.stack(hs, dim=1)
        return F.dropout(hs, self.dropout)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'sum2':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'sum3':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'sum4':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'mean2':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'mean3':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'mean4':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'min2':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'min3':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'min4':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'max2':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'max3':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'max4':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')