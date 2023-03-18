from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch_geometric.nn.dense.linear import Linear


class MaskAggregateLinear(Linear):
    """
    A PyTorch module that applies different linear transformations to node features 
    based on a given aggregation type.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregation_list: List[str],
        aggregation: str,
        mask: bool = True,
        bias: bool = True,
        weight_initializer: Optional[str] = None,
        bias_initializer: Optional[str] = None
    ):
        """
        Constructor for MaskAggregateLinear module.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            aggregation_list (List[str]): List of aggregation types.
            aggregation (str): Aggregation type to be used.
            bias (bool): Whether to include bias term. Default is True.
            weight_initializer (Optional[str]): Weight initialization method. Default is None.
            bias_initializer (Optional[str]): Bias initialization method. Default is None.
        """
        super().__init__(in_channels, out_channels, bias, weight_initializer, bias_initializer)
        
        # Set the device for computations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Store the aggregation type and its corresponding linear transformation layer
        self.mask = mask
        self.aggregation = aggregation
        self.aggregation_layers = {}
        for i, aggr in enumerate(aggregation_list):
            aggregation_name = "{}".format(aggr)
            if self.mask == "no_linear":
                self.aggregation_layers[aggregation_name] = None
            else:
                linear = Linear(in_channels, out_channels, bias, weight_initializer, bias_initializer).to(self.device)
                self.aggregation_layers[aggregation_name] = linear

    def forward(self, input):
        """
        Forward pass of MaskAggregateLinear module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, num_nodes, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, out_channels).
        """
        if self.aggregation not in self.aggregation_layers:
            raise ValueError("Invalid aggregation type: {}".format(self.aggregation))
        if self.mask == "no_linear":
            return input
        else:
            return self.aggregation_layers[self.aggregation](input).to(self.device)
