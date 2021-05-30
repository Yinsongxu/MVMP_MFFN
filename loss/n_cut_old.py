from __future__ import absolute_import
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import torch_geometric as tg
from torch_sparse import coalesce,spmm
class Ncut(nn.Module):
    def __init__(self):
        super(Ncut, self).__init__()
        a = torch.arange(0,8)
        a = a.unsqueeze(1)
        a = a.expand(8,8)
        a = a.reshape(-1)
        self.labels = torch.zeros(torch.Size([a.size(0), 8])).scatter_(1, a.unsqueeze(1).data.cpu(), 1).cuda() 
        index = torch.arange(0,64)
        self.deg_index = torch.stack((index, index), 1).t().cuda()
        self.ones = torch.ones(8).cuda()
    def forward(self, edge_index, weight):
        
        """
        Args:
            edge_index(torch.Tensor):
                shape(2, num_edges)
            weight(torch.Tensor):
                shape(num_edges) 
        """
        num_nodes=64
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        weight = torch.cat([weight,weight],dim=0)
        edge_index, weight = coalesce(edge_index, weight, num_nodes, num_nodes)

        numerator=spmm(edge_index,weight,64,64,self.labels).t()*self.labels.t()
        numerator = numerator.sum(1)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(weight, row, dim=0, dim_size=64)
        #print(weight)
        denominator = spmm(self.deg_index,deg,64,64,self.labels).t()*self.labels.t()
        denominator = denominator.sum(1)
        #print(denominator)
        return (self.ones-numerator/denominator).sum(0)/8
    '''

    affinity = mx.symbol.BlockGrad(sim)

    pred_t = mx.symbol.transpose(pred)
    pred_b_t = mx.symbol.expand_dims(pred_t, 2)

    numerator = mx.symbol.batch_dot(mx.symbol.expand_dims(mx.symbol.dot(pred_t, affinity), 1), 1 - pred_b_t)
    numerator = mx.symbol.squeeze(numerator)

    degree = mx.symbol.sum(affinity, axis=1)

    denominator = mx.symbol.dot(pred_t, degree)

    loss = mx.symbol.mean(numerator / denominator)
    '''
