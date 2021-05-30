from __future__ import absolute_import
import torch
import torch.nn as nn
class SmoothLoss(nn.Module):
    def __init__(self, use_gpu=True, k=9):
        super(SmoothLoss, self).__init__()
        self.k = k
        self.ranking_loss = nn.MarginRankingLoss(margin=0)
    
        self.softplus = nn.Softplus()
    
    def forward(self, edge_index, edge_weight, labels):
        """
        Args:
            edge_index(torch.Tensor): edge index in affinity graph
                shape(2, edge_num)
            edge_weight(torch.Tensor): featuer matrix
                shape(1, edge_num)
            labels(torch.LongTensor): ground truth labels
                shape(batch_size,)
        """
        #print(edge_index.size())
        edge_num = edge_index.size(1)
        source = edge_index[0]
        target = edge_index[1]
        source_labels = labels[source]
        target_labels = labels[target]
        mask_neg = source_labels - target_labels != 0 
        mask_pos = ~mask_neg
        
        pos = torch.masked_select(edge_weight, mask_pos)
        neg = torch.masked_select(edge_weight, mask_neg)
        #print(pos)
        #print(neg)

        pos = torch.logsumexp(pos, dim=0)
        neg = torch.logsumexp(neg, dim=0)
        return self.softplus(pos+neg).sum(0)/8
