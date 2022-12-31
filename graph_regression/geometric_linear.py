from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch_geometric.nn.dense.linear import Linear

class MaskAggregateLinear(Linear):
    def __init__(self, in_channels: int, out_channels: int, 
                 aggregation_list:list, aggregation: str,
                 bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__(in_channels, out_channels, bias, weight_initializer, bias_initializer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aggregation = aggregation
        self.aggregation_layers = {}
        for i, aggr in enumerate(aggregation_list):
            aggregation_name = "{}".format(aggr)
            linear = Linear(in_channels, out_channels, bias, weight_initializer, bias_initializer).to(self.device)
            self.aggregation_layers[aggregation_name] = linear

    def forward(self, input):
        if self.aggregation not in self.aggregation_layers:
            raise ValueError("Invalid aggregation type: {}".format(self.aggregation))
        return self.aggregation_layers[self.aggregation](input).to(self.device)