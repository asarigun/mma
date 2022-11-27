import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):

    '''
    Source code of GCN from https://github.com/tkipf/pygcn
    '''
    def __init__(self, in_features, out_features, weight, bias, device): 
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.device = device        

        self.weight = weight
        
        self.bias = bias
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
     
    def forward(self, input, adj):

        support = torch.mm(input, self.weight).to(self.device)
        output = torch.spmm(adj, support).to(self.device)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class MMA(Module):


    def __init__(self, add_all, activation, k, in_features, out_features, weight, bias, 
                weight_moment_3, weight_sum, weight_sum2, weight_sum3, weight_sum4, weight_mean, 
                weight_mean2, weight_mean3, weight_mean4, weight_max, weight_max2, weight_max3, 
                weight_max4, weight_min, weight_min2, weight_min3, weight_min4, weight_softmax, 
                weight_softmin, weight_std, weight_normalized_mean, dropout, aggregator_list, device): 
        super(MMA, self).__init__()

        '''
        MMA Adapted from code of GCN from https://github.com/tkipf/pygcn 
                                    and https://github.com/LiZhang-github/LA-GCN/tree/master/code
        '''

        self.activation = activation
        self.k = k
        self.in_features = in_features
        self.Sig = nn.Sigmoid()

        self.out_features = out_features
        self.add_all = add_all
        self.dropout = dropout
        self.device = device
        

        self.all_aggregators = {'moment_3':self.learnable_moment_3,
                                     'sum': self.learnable_sum,
                                     'sum2': self.learnable_sum2,
                                     'sum3': self.learnable_sum3, 
                                     'sum4': self.learnable_sum4,
                                     'mean': self.learnable_mean, 
                                     'mean2': self.learnable_mean2,
                                     'mean3': self.learnable_mean3,
                                     'mean4': self.learnable_mean4,
                                     'max': self.learnable_max, 
                                     'max2': self.learnable_max2,
                                     'max3': self.learnable_max3,
                                     'max4': self.learnable_max4,
                                     'min': self.learnable_min, 
                                     'min2': self.learnable_min2,
                                     'min3': self.learnable_min3,
                                     'min4': self.learnable_min4,
                                     'softmax': self.learnable_softmax, 
                                     'softmin': self.learnable_softmin, 
                                     'std': self.learnable_std, 
                                     'normalized_mean': self.learnable_normalized_mean}


        self.AGGREGATORS = dict()

        for aggr in aggregator_list:
            self.AGGREGATORS[aggr] = self.all_aggregators[aggr]

        self.weight = weight

        self.aggregators = [self.AGGREGATORS[aggr] for aggr in self.AGGREGATORS]
        self.scalers = [SCALERS[scale] for scale in SCALERS]
        self.num_aggregators = len(self.aggregators)

        self.mask_moment_3 = weight_moment_3
        self.mask_sum = weight_sum
        self.mask_sum2 = weight_sum2
        self.mask_sum3 = weight_sum3
        self.mask_sum4 = weight_sum4
        self.mask_mean = weight_mean
        self.mask_mean2 = weight_mean2
        self.mask_mean3 = weight_mean3
        self.mask_mean4 = weight_mean4
        self.mask_max = weight_max
        self.mask_max2 = weight_max2
        self.mask_max3 = weight_max3
        self.mask_max4 = weight_max4
        self.mask_min = weight_min
        self.mask_min2 = weight_min2
        self.mask_min3 = weight_min3
        self.mask_min4 = weight_min4
        self.mask_softmax = weight_softmax
        self.mask_softmin = weight_softmin
        self.mask_std = weight_std
        self.mask_normalized_mean = weight_normalized_mean

        self.bias = bias

        self.reset_parameters()

        self.avg_d = None
        self.self_loop = None

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(0))


        self.mask_stdv_moment_3 = 1. / math.sqrt(self.mask_moment_3.size(1))
        self.mask_stdv_sum = 1. / math.sqrt(self.mask_sum.size(1))
        self.mask_stdv_sum2 = 1. / math.sqrt(self.mask_sum2.size(1))
        self.mask_stdv_sum3 = 1. / math.sqrt(self.mask_sum3.size(1))
        self.mask_stdv_sum4 = 1. / math.sqrt(self.mask_sum4.size(1))
        self.mask_stdv_mean = 1. / math.sqrt(self.mask_mean.size(1))
        self.mask_stdv_mean2 = 1. / math.sqrt(self.mask_mean2.size(1))
        self.mask_stdv_mean3 = 1. / math.sqrt(self.mask_mean3.size(1))
        self.mask_stdv_mean4 = 1. / math.sqrt(self.mask_mean4.size(1))
        self.mask_stdv_max = 1. / math.sqrt(self.mask_max.size(1))
        self.mask_stdv_max2 = 1. / math.sqrt(self.mask_max2.size(1))
        self.mask_stdv_max3 = 1. / math.sqrt(self.mask_max3.size(1))
        self.mask_stdv_max4 = 1. / math.sqrt(self.mask_max4.size(1))
        self.mask_stdv_min = 1. / math.sqrt(self.mask_min.size(1))
        self.mask_stdv_min2 = 1. / math.sqrt(self.mask_min2.size(1))
        self.mask_stdv_min3 = 1. / math.sqrt(self.mask_min3.size(1))
        self.mask_stdv_min4 = 1. / math.sqrt(self.mask_min4.size(1))
        self.mask_stdv_softmax = 1. / math.sqrt(self.mask_softmax.size(1))
        self.mask_stdv_softmin = 1. / math.sqrt(self.mask_softmin.size(1))
        self.mask_stdv_std = 1. / math.sqrt(self.mask_std.size(1))
        self.mask_stdv_normalized_mean = 1. / math.sqrt(self.mask_normalized_mean.size(1))

        self.weight.data.uniform_(-stdv, stdv)
       
        self.mask_moment_3.data.uniform_(-self.mask_stdv_moment_3, self.mask_stdv_moment_3)
        self.mask_sum.data.uniform_(-self.mask_stdv_sum, self.mask_stdv_sum)
        self.mask_sum2.data.uniform_(-self.mask_stdv_sum2, self.mask_stdv_sum2)
        self.mask_sum3.data.uniform_(-self.mask_stdv_sum3, self.mask_stdv_sum3)
        self.mask_sum4.data.uniform_(-self.mask_stdv_sum4, self.mask_stdv_sum4)
        self.mask_mean.data.uniform_(-self.mask_stdv_mean, self.mask_stdv_mean)
        self.mask_mean2.data.uniform_(-self.mask_stdv_mean2, self.mask_stdv_mean2)
        self.mask_mean3.data.uniform_(-self.mask_stdv_mean3, self.mask_stdv_mean3)
        self.mask_mean4.data.uniform_(-self.mask_stdv_mean4, self.mask_stdv_mean4)
        self.mask_max.data.uniform_(-self.mask_stdv_max, self.mask_stdv_max)
        self.mask_max2.data.uniform_(-self.mask_stdv_max2, self.mask_stdv_max2)
        self.mask_max3.data.uniform_(-self.mask_stdv_max3, self.mask_stdv_max3)
        self.mask_max4.data.uniform_(-self.mask_stdv_max4, self.mask_stdv_max4)
        self.mask_min.data.uniform_(-self.mask_stdv_min, self.mask_stdv_min)
        self.mask_min2.data.uniform_(-self.mask_stdv_min2, self.mask_stdv_min2)
        self.mask_min3.data.uniform_(-self.mask_stdv_min3, self.mask_stdv_min3)
        self.mask_min4.data.uniform_(-self.mask_stdv_min4, self.mask_stdv_min4)
        self.mask_softmax.data.uniform_(-self.mask_stdv_softmax, self.mask_stdv_softmax)
        self.mask_softmin.data.uniform_(-self.mask_stdv_softmin, self.mask_stdv_softmin)
        self.mask_std.data.uniform_(-self.mask_stdv_std, self.mask_stdv_std)
        self.mask_normalized_mean.data.uniform_(-self.mask_stdv_normalized_mean, self.mask_stdv_normalized_mean)




        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def learnable_sum(self, input, adj):  #Learnable Sum Aggregator

        input_new_sum = []

        for i in range(len(self.add_all)):

            #input shape for cora (2708,16)
            
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) #node's features
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device)
            bb_nei_index2 = self.add_all[i] #find neighborhood
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64") #finding ID of each neighborhood
            bb_nei_index2 = torch.tensor(bb_nei_index2).to(self.device)     
            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device) #finding features of neighbors
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device) 
            mask0 = torch.mm(cen_nei, self.mask_sum).to(self.device) 
            mask0 = self.Sig(mask0)
            
            mask0 = F.dropout(mask0, self.dropout)
               
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) 
            input_new_sum.append(new_cen_nei)                                      
        
        input_new_sum = torch.stack(input_new_sum).to(self.device)                                     
        input_new_sum = torch.squeeze(input_new_sum).to(self.device)
        return input_new_sum

    def learnable_sum2(self, input, adj):  #Learnable Sum Aggregator

        input_new_sum2 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_sum2).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator
            input_new_sum2.append(new_cen_nei)                                      
            
        input_new_sum2 = torch.stack(input_new_sum2)                                     
        input_new_sum2 = torch.squeeze(input_new_sum2)
        return input_new_sum2

    def learnable_sum3(self, input, adj):  #Learnable Sum Aggregator

        input_new_sum3 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_sum3).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator
            input_new_sum3.append(new_cen_nei)                                      
            
        input_new_sum3 = torch.stack(input_new_sum3)                                     
        input_new_sum3 = torch.squeeze(input_new_sum3)
        return input_new_sum3

    def learnable_sum4(self, input, adj):  #Learnable Sum Aggregator

        input_new_sum4 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_sum4).to(self.device) 
            
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator
            input_new_sum4.append(new_cen_nei)                                      
            
        input_new_sum4 = torch.stack(input_new_sum4)                                     
        input_new_sum4 = torch.squeeze(input_new_sum4)
        return input_new_sum4

    
    def learnable_mean(self, input, adj):  #Learnable Mean Aggregator
        input_new_mean = []
               
        for i in range(len(self.add_all)):
            
            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device)

            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2).to(self.device)

            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device)
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)

            mask0 = torch.mm(cen_nei, self.mask_mean).to(self.device)

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                                     
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) 
               
            D = len(self.add_all[i]) 
            new_cen_nei_mean = torch.div(new_cen_nei, D).to(self.device)
            input_new_mean.append(new_cen_nei_mean)                                      
                           
        input_new_mean = torch.stack(input_new_mean).to(self.device)                                     
        input_new_mean = torch.squeeze(input_new_mean).to(self.device)
        return input_new_mean

    def learnable_mean2(self, input, adj):  #Learnable Mean Aggregator

        input_new_mean2 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_mean2).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator

            D = len(self.add_all[i]) 

            new_cen_nei_mean = torch.div(new_cen_nei, D).to(self.device)
            input_new_mean2.append(new_cen_nei_mean)                                      
            
        input_new_mean2 = torch.stack(input_new_mean2).to(self.device)                                     
        input_new_mean2 = torch.squeeze(input_new_mean2).to(self.device)
        return input_new_mean2

    def learnable_mean3(self, input, adj):  #Learnable Mean Aggregator

        input_new_mean3 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_mean3).to(self.device) 
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator

            D = len(self.add_all[i]) 

            new_cen_nei_mean = torch.div(new_cen_nei, D).to(self.device)
            input_new_mean3.append(new_cen_nei_mean)                                      
            
        input_new_mean3 = torch.stack(input_new_mean3).to(self.device)                                     
        input_new_mean3 = torch.squeeze(input_new_mean3).to(self.device)
        return input_new_mean3

    def learnable_mean4(self, input, adj):  #Learnable Mean Aggregator

        input_new_mean4 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_mean4).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator

            D = len(self.add_all[i]) 
            new_cen_nei_mean = torch.div(new_cen_nei, D).to(self.device)

            input_new_mean4.append(new_cen_nei_mean)                                      
            
        input_new_mean4 = torch.stack(input_new_mean4).to(self.device)                                     
        input_new_mean4 = torch.squeeze(input_new_mean4).to(self.device)
        return input_new_mean4
                       
                          
    def learnable_max(self, input, adj, min_value = -math.inf):     #Learnable Max Aggregator   
                       
        input_new_max = []
                       
        for i in range(len(self.add_all)):
               
            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device)
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2).to(self.device)
            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device)
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_max).to(self.device)
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                                     
            new_cen_nei_max = torch.max(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)
            input_new_max.append(new_cen_nei_max)                                      
                           
        input_new_max = torch.stack(input_new_max).to(self.device)                                  
        input_new_max = torch.squeeze(input_new_max).to(self.device)
        return input_new_max

    def learnable_max2(self, input, adj, min_value = -math.inf):     #Learnable Max Aggregator   
        
        input_new_max2 = []
        
        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_max2) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei_max = torch.max(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)

            input_new_max2.append(new_cen_nei_max)                                      
            
        input_new_max2 = torch.stack(input_new_max2).to(self.device)                                     
        input_new_max2 = torch.squeeze(input_new_max2).to(self.device)
        return input_new_max2


    def learnable_max3(self, input, adj, min_value = -math.inf):     #Learnable Max Aggregator   
        
        input_new_max3 = []
        
        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_max3).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei_max = torch.max(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)
            input_new_max3.append(new_cen_nei_max)                                      
            
        input_new_max3 = torch.stack(input_new_max3).to(self.device)                                     
        input_new_max3 = torch.squeeze(input_new_max3).to(self.device)
        return input_new_max3

    def learnable_max4(self, input, adj, min_value = -math.inf):     #Learnable Max Aggregator   
        
        input_new_max4 = []
        
        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_max4).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei_max = torch.max(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)

            input_new_max4.append(new_cen_nei_max)                                      
            
        input_new_max4 = torch.stack(input_new_max4).to(self.device)                                     
        input_new_max4 = torch.squeeze(input_new_max4).to(self.device)
        return input_new_max4                      
  
    def learnable_min(self, input, adj,max_value=math.inf):    #Learnable Min Aggregator
               
        input_new_min = []
               
        for i in range(len(self.add_all)):
               
            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device)
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2).to(self.device)
            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device)
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_min).to(self.device)
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                                     
            new_cen_nei_min = torch.min(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)
            input_new_min.append(new_cen_nei_min)   

                           
        input_new_min = torch.stack(input_new_min).to(self.device)                              
        input_new_min = torch.squeeze(input_new_min).to(self.device)
               
        return input_new_min

    def learnable_min2(self, input, adj, max_value=math.inf):    #Learnable Min Aggregator

        input_new_min2 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_min2).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei_min = torch.min(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)

            input_new_min2.append(new_cen_nei_min)                                      
            
        input_new_min2 = torch.stack(input_new_min2).to(self.device)                                     
        input_new_min2 = torch.squeeze(input_new_min2).to(self.device)
        return input_new_min2

    def learnable_min3(self, input, adj, max_value=math.inf):    #Learnable Min Aggregator

        input_new_min3 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_min3).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
            new_cen_nei_min = torch.min(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)

            input_new_min3.append(new_cen_nei_min)                                      
            
        input_new_min3 = torch.stack(input_new_min3).to(self.device)                                     
        input_new_min3 = torch.squeeze(input_new_min3).to(self.device)
        return input_new_min3

    def learnable_min4(self, input, adj, max_value=math.inf):    #Learnable Min Aggregator

        input_new_min4 = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]]).to(self.device)
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]).to(self.device)).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")

            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2).to(self.device)).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_min4).to(self.device) 

            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      

            new_cen_nei_min = torch.min(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)

            input_new_min4.append(new_cen_nei_min)                                      
            
        input_new_min4 = torch.stack(input_new_min4).to(self.device)                                     
        input_new_min4 = torch.squeeze(input_new_min4).to(self.device)
        return input_new_min4   
        
    def learnable_softmax(self, input, adj):  #Learnable variance of the features of the neighbours

        input_new_softmax = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])).to(self.device) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]).to(self.device) 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device) 
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask_softmax).to(self.device) 
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
                                      
            X = torch.sum(mask0 * bb_nei, 0, keepdims=True).to(self.device)

            X_exp = torch.exp(X).to(self.device)

            X_sum = torch.sum(X_exp, dim=0, keepdim=True).to(self.device)
            softmax = torch.sum(torch.mul(torch.div(X_exp, X_sum), X), dim=0).to(self.device)
            softmax = torch.reshape(softmax, (1, self.in_features))
                  

            new_cen_nei_softmax = softmax
            input_new_softmax.append(new_cen_nei_softmax) 

        input_new_softmax = torch.stack(input_new_softmax).to(self.device)                                     
        input_new_softmax = torch.squeeze(input_new_softmax).to(self.device)  

        return input_new_softmax

    def learnable_softmin(self, input, adj):  #Learnable variance of the features of the neighbours

        input_new_softmin = []

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1])
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, bb_nei_index2) 
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            mask0 = torch.mm(cen_nei, self.mask_softmin) 
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)

                                      
            X = torch.sum(mask0 * bb_nei, 0, keepdims=True).to(self.device)
            X_exp = torch.exp(-X).to(self.device)
            X_sum = torch.sum(X_exp, dim=0, keepdim=True).to(self.device)
            softmin = torch.sum(torch.mul(torch.div(X_exp, X_sum), X), dim=0).to(self.device)   
            softmin = torch.reshape(softmin, (1, self.in_features))

            new_cen_nei_softmin = softmin
            input_new_softmin.append(new_cen_nei_softmin) 

        input_new_softmin = torch.stack(input_new_softmin)                                     
        input_new_softmin = torch.squeeze(input_new_softmin)
        
        return input_new_softmin


    def learnable_std(self, input, adj):  #Learnable variance of the features of the neighbours

        input_new_std = []

        EPS = 1e-5

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2)) 
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            mask0 = torch.mm(cen_nei, self.mask_std) 
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
                                      
            X = torch.sum(mask0 * bb_nei, 0, keepdims=True)

            D = len(self.add_all[i]) 

            X_sum_squares = torch.sum(torch.mul(X, X)) 
            X_mean_squares = torch.div(X_sum_squares, D)  
            X_mean = self.learnable_mean(input, adj)  
            var = torch.relu(X_mean_squares - torch.mul(X_mean, X_mean))  
            std = torch.sqrt(var + EPS) 

            new_cen_nei_std = std
            input_new_std.append(new_cen_nei_std) 

        input_new_std = torch.stack(input_new_std)                                     
        input_new_std = torch.squeeze(input_new_std)
        return input_new_std

    def learnable_normalized_mean(self, input, adj):  #Learnable variance of the features of the neighbours

        input_new_normalized_mean = []

        EPS = 1e-5

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, bb_nei_index2)
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            mask0 = torch.mm(cen_nei, self.mask_normalized_mean) 
            if self.activation=="new_sigmoid":

                self.Sig(mask0-self.k)-self.Sig(-mask0-self.k)
            else:
                mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
                                      
            X = torch.sum(mask0 * bb_nei, 0, keepdims=True)
                        
            D = len(self.add_all[i]) 

            D = torch.tensor([D]) 
            rD = torch.pow(D, -0.5) 

            adj =  torch.matmul(torch.matmul(rD, adj), rD)  
            X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)


            new_cen_nei_normalized_mean = X_sum

            input_new_new_normalized_mean.append(new_cen_nei_normalized_mean) 

        input_new_normalized_mean = torch.stack(input_new_normalized_mean)                                     
        input_new_normalized_mean = torch.squeeze(input_new_normalized_mean)
        return input_new_normalized_mean

    def learnable_moment_3(self, input, adj):  #Learnable variance of the features of the neighbours

        input_new_moment_3 = []

        EPS = 1e-5
        n=3

        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2)) 
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            mask0 = torch.mm(cen_nei, self.mask_moment_3) 
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
                                      
            X = torch.sum(mask0 * bb_nei, 0, keepdims=True)
                        
            D = len(self.add_all[i]) 
            X_mean = self.learnable_mean(X, adj)
            X_n = torch.div(torch.sum(torch.mul(torch.pow(X - X_mean.unsqueeze(2), n), adj.unsqueeze(-1)), dim=2), D)
            rooted_X_3 = torch.sign(X_n) * torch.pow(torch.abs(X_n) + EPS, 1. / n)
    
            new_cen_nei_moment_3 = rooted_X_3
            input_new_new_moment_3.append(new_cen_nei_moment_3) 

        input_new_moment_3 = torch.stack(input_new_moment_3)                                     
        input_new_moment_3 = torch.squeeze(input_new_moment_3)
        return input_new_moment_3

    def forward(self, input, adj):

        m = torch.cat([aggregate(input, adj) for aggregate in self.aggregators], dim=0).to(self.device)
        m = torch.cat([scale(m, adj, self.num_aggregators) for scale in self.scalers], dim=1).to(self.device) 

        weight = torch.cat([self.weight,self.weight,self.weight], dim=0).to(self.device)
        
        support = torch.mm(m, weight).to(self.device)
        adj = torch.cat((adj,)*len(self.AGGREGATORS),1).to(self.device)
        output = torch.spmm(adj, support).to(self.device)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

