from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from mask_aggr import MaskAggregateLinear

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

        self.pre_nns = {}
        for i, aggr in enumerate(aggregators):
            self.pre_nns[aggr] = ModuleList()
        
        self.post_nns = ModuleList()
        for _ in range(towers):
            # --------------> Masked (Learnable) Aggregation <-------------------- #
            for aggr in aggregators:
                modules = [MaskAggregateLinear((3 if edge_dim else 2) * self.F_in, self.F_in, aggregators, aggr)]
                for _ in range(pre_layers - 1):
                    modules += [ReLU()]
                    modules += [MaskAggregateLinear(self.F_in, self.F_in, aggregators, aggr)]
                self.pre_nns[aggr].append(Sequential(*modules))
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

        for aggr in self.pre_nns:
            for nn in aggr:
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
            if aggregator.startswith(('sum', 'mean', 'min', 'max')):
                hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns[aggregator])]
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')

        hs = torch.stack(hs, dim=1)
        return F.dropout(hs, self.dropout)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        
        outs = []
        for aggregator in self.aggregators:
            if aggregator.startswith(('sum', 'mean', 'min', 'max')):
                reduce_type = aggregator[:3]
                out = scatter(inputs, index, 0, None, dim_size, reduce=reduce_type)
            elif aggregator in ('var', 'std'):
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
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