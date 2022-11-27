import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, MMA

'''
Adapted from the source code of MMA adapted from https://github.com/tkipf/pygcn
'''

class MMAConv(nn.Module):

    def __init__(self, add_all, activation, k, nfeat, nhid, nclass, dropout, aggregator_list, device):
        super(MMAConv, self).__init__()
        
        self.device = device        

        self.weight0 = nn.Parameter(torch.cuda.FloatTensor(nfeat, nhid))
        self.bias0 = nn.Parameter(torch.cuda.FloatTensor(nhid))
          
        self.weight1 = nn.Parameter(torch.cuda.FloatTensor(nhid, nclass))
        self.bias1 = nn.Parameter(torch.cuda.FloatTensor(nclass))

        self.weight_moment_3 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_sum = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_sum2 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_sum3 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_sum4 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_mean = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_mean2 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_mean3 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_mean4 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_max = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_max2 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_max3 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_max4 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_min = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_min2 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_min3 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_min4 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_softmax = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_softmin = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_std = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))
        self.weight_normalized_mean = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))

        self.parameters = nn.ParameterList([self.weight0, self.bias0, self.weight1, self.bias1, 
                                        self.weight_moment_3, self.weight_sum, self.weight_sum2, self.weight_sum3, 
                                        self.weight_sum4, self.weight_mean, self.weight_mean2, self.weight_mean3, self.weight_mean4, 
                                        self.weight_max, self.weight_max2, self.weight_max3, self.weight_max4, self.weight_min, 
                                        self.weight_min2, self.weight_min3, self.weight_min4, self.weight_softmax, self.weight_softmin, 
                                        self.weight_std, self.weight_normalized_mean])

        self.add_all = add_all

        self.gc1 = GraphConvolution(nfeat, nhid, self.weight0, self.bias0, device)  
        self.gc2 = MMA(self.add_all, activation, k, nhid, nclass, self.weight1, self.bias1, 
                    self.weight_moment_3,self.weight_sum,self.weight_sum2,self.weight_sum3,
                    self.weight_sum4,self.weight_mean,self.weight_mean2,self.weight_mean3,
                    self.weight_mean4,self.weight_max,self.weight_max2,self.weight_max3,self.weight_max4,
                    self.weight_min,self.weight_min2,self.weight_min3,self.weight_min4,self.weight_softmax,
                    self.weight_softmin,self.weight_std,self.weight_normalized_mean, dropout, aggregator_list, device)  
        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
