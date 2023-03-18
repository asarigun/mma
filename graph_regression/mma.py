import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.nn.modules.module import T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, global_add_pool
from mma_conv import MMAConv
import argparse
import numpy as np

'''
Multi-Mask Aggregators 
Adapted from the source code https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
'''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_dim', type=int, default=16, help='Number of hidden dimensipns.')
parser.add_argument('--out_dim', type=int, default=16, help='Number of out dimensions.')
parser.add_argument('--edge_dim', type=int, default=16, help='Number of edge dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
parser.add_argument('--tower', type=int, default=1, help='Number of towers')
parser.add_argument('--aggregators', type=str, default="mean,max,min", help='choose your aggregators')
parser.add_argument('--scalers', type=str, default="identity,amplification,attenuation", help='choose your scalers')
parser.add_argument('--L', type=int, default=4, help='Enter number of layers')
parser.add_argument('--cuda', type=str, default="cuda:0", help='choose your cuda device IDs')
parser.add_argument('--mask', type=bool, default=True, help='decide using mask or not')

# Parse arguments
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load ZINC dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
train_dataset = ZINC(path, subset=True, split='train')
val_dataset = ZINC(path, subset=True, split='val')
test_dataset = ZINC(path, subset=True, split='test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Compute in-degree histogram over training data.
deg = torch.zeros(5, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

# Define neural network model
class Net(torch.nn.Module):
    """
    Neural network class definition.
    """
    def __init__(self, args, aggregator_list, scaler_list):
        """
        Initializes the neural network.

        Args:
        - args: Command line arguments.
        - aggregator_list: List of aggregators.
        - scaler_list: List of scalers.
        """
        super(Net, self).__init__()

        # Define node and edge embeddings
        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        # Define aggregators and scalers
        aggregators = aggregator_list
        scalers = scaler_list

        # Define convolutional layers and batch normalization layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        
        # Add four convolutional layers with corresponding batch normalization layers
        for _ in range(4): # 4 ------> number of layers
            conv = MMAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           mask = args.mask, divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        # Define fully connected layers
        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Defines the forward pass of the neural network.

        Args:
        - x: Node features.
        - edge_index: Edge indices.
        - edge_attr: Edge attributes.
        - batch: Batch indices.

        Returns:
        - The output of the neural network.
        """
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        # Perform convolutional and batch normalization operations
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        
        # Perform global pooling
        x = global_add_pool(x, batch)
        
        # Perform fully connected layers
        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize neural network, move to GPU if available
model = Net(args, aggregator_list=args.aggregators.split(","), scaler_list=args.scalers.split(",")).to(args.cuda)

# Initialize optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

def train(epoch):
    """
    Train the neural network model for a given number of epochs using the training data.

    Args:
        epoch (int): The current epoch number.

    Returns:
        The average loss across the training dataset for the current epoch.
    """
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(args.cuda)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    """
    Evaluate the performance of the neural network model on the given data loader.

    Args:
        loader (DataLoader): The data loader to evaluate the model on.

    Returns:
        The average error across the dataset.
    """
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(args.cuda)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()

    return total_error / len(loader.dataset)


# Train the model for the given number of epochs
for epoch in range(args.epochs):
    # Train the model for one epoch
    loss = train(epoch)
    
    # Evaluate the model on the validation and test data
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    
    # Update the learning rate using the ReduceLROnPlateau scheduler
    scheduler.step(val_mae)
    
    # Print the epoch number, training loss, and evaluation metrics
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')

