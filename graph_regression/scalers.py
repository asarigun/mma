from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor
from torch.nn.parameter import Parameter
import numpy as np
import math
import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn as nn
from torch.nn import ModuleList, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing

from geometric_linear import Linear
from torch_geometric.utils import degree
import torch.nn.functional as F

from torch_geometric.nn.inits import reset

'''
Source code of scalers from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/aggr/scaler.py
'''

def scalers(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
                  

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