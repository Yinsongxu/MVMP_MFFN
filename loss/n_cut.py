from __future__ import absolute_import
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import torch_geometric as tg
from torch_sparse import coalesce,spmm
import torch.nn.functional as F
from utils import cosine_distance
class Ncut(nn.Module):
    def __init__(self, gamma):
        super(Ncut, self).__init__()
        

        self.softplus = nn.Softplus()
        self.mask = ~torch.eye(64).bool().cuda()
        self.gamma = gamma
    def forward(self, feat, labels, edge):
        
        """
        Args:
            edge_index(torch.Tensor):
                shape(2, num_edges)
            weight(torch.Tensor):
                shape(num_edges) 
        """
        #edge, _ = tg.utils.remove_self_loops(edge)
        #edge, _ = tg.utils.add_self_loops(edge,  num_nodes = 64)
        N = feat.size(0)
        feat = F.normalize(feat, p=2, dim=1)
        affinity = self.gamma * torch.mm(feat, feat.t()).view(8,-1)
        pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        pos = pos.view(8,-1)
        neg = neg.view(8,-1)
        dist_ap, dist_an = [], []
        for i in range(8):
            dist_ap.append(-1*affinity[i][pos[i]])
            dist_an.append(affinity[i][neg[i]])
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        loss = self.softplus(torch.logsumexp(dist_ap, dim=0) + torch.logsumexp(dist_an, dim=0))
        return loss/8.
        
        '''
        
        affinity = affinity - (torch.eye(128)*999.).cuda()
        #print(affinity)
        affinity = affinity.reshape(8,-1)
        affinity = self.softmax(affinity)
        affinity = affinity.reshape(128,128)
        #print(affinity)
        numerator = torch.matmul(self.labels.t(), affinity)*self.labels.t()
        numerator = numerator.sum(1)
        numerator = -torch.log(self.ones - numerator)
        return numerator.sum(0)/8
        '''
        ''' 
        numerator = torch.matmul(self.labels.t(), affinity)
        pos =   torch.masked_select(numerator, self.mask.t()).reshape(8,-1)
        neg =   -torch.masked_select(numerator, ~self.mask.t()).reshape(8,-1)
        pos = torch.logsumexp(pos, dim=1)
        #print(pos.size())
        #pos = torch.exp(pos).sum(1)
        #print(pos)
        neg = torch.logsumexp(neg, dim=1)
        return self.softplus(pos+neg).sum(0)/8
        #return self.softplus(l).sum(0)
        '''
        

        