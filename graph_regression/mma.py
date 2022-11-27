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


args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
train_dataset = ZINC(path, subset=True, split='train')
val_dataset = ZINC(path, subset=True, split='val')
test_dataset = ZINC(path, subset=True, split='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Compute in-degree histogram over training data.
deg = torch.zeros(5, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())


class Net(torch.nn.Module):
    def __init__(self,args, aggregator_list, scaler_list):
        super(Net, self).__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = aggregator_list
        scalers = scaler_list

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        
        for _ in range(4): # 4 ------> number of layers
            
            conv = MMAConv(dropout = args.dropout, in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch)
        return self.mlp(x)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args, aggregator_list = args.aggregators.split(","), scaler_list = args.scalers.split(",")).to(args.cuda)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)


def train(epoch):
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
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(args.cuda)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


for epoch in range(args.epochs):
    loss = train(epoch)
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    scheduler.step(val_mae)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')
