import torch
from utils import *


############################################################################
# This section of code adapted from https://github.com/lukecavabarrett/pna #
############################################################################


def avg_d_log(all_degrees):

    avg_d_log = torch.mean(torch.log(all_degrees + 1)).to('cuda:2')

    return avg_d_log


def avg_d_exp(all_degrees):

    return torch.mean(torch.exp(torch.div(1, all_degrees).to('cuda:2')) - 1).to('cuda:2')


def scale_identity(input, add_all, num_aggregators, avg_d=None):
    return input


def scale_amplification(input, add_all, num_aggregators, avg_d=None):

    all_degrees = [len(node_nei) for node_nei in add_all]
    all_degrees = torch.tensor(all_degrees).to('cuda:2')

    scale = (torch.log(all_degrees + 1).to('cuda:2') / avg_d_log(all_degrees)).unsqueeze(-1)

    if num_aggregators == 1 : 
        scale = scale
    elif num_aggregators == 2 :
        scale = torch.cat((scale,scale),0).to('cuda:2')
    elif num_aggregators == 3 :
        scale = torch.cat((scale,scale,scale),0).to('cuda:2')
    elif num_aggregators == 4 :
        scale = torch.cat((scale,scale,scale,scale),0).to('cuda:2')
    input_scaled = torch.mul(scale, input).to('cuda:2')
    return input_scaled


def scale_attenuation(input, add_all, num_aggregators, avg_d=None):

    all_degrees = [len(node_nei) for node_nei in add_all]
    all_degrees = torch.tensor(all_degrees).to('cuda:2')
    
    scale = (avg_d_log(all_degrees) / torch.log(all_degrees + 1).to('cuda:2')).unsqueeze(-1)
    if num_aggregators == 1 :
        scale = scale
    elif num_aggregators == 2 :
        scale = torch.cat((scale,scale),0).to('cuda:2')
    elif num_aggregators == 3 :
        scale = torch.cat((scale,scale,scale),0).to('cuda:2')
    elif num_aggregators == 4 :
        scale = torch.cat((scale,scale,scale,scale),0).to('cuda:2')


    X_scaled = torch.mul(scale, input).to('cuda:2')
    return X_scaled

SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation} 



