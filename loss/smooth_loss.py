from __future__ import absolute_import
import torch
import torch.nn as nn
class SmoothLoss(nn.Module):
    def __init__(self, use_gpu=True, k=9):
        super(SmoothLoss, self).__init__()
        self.k = k
        self.ranking_loss = nn.MarginRankingLoss(margin=0)
    
    
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
        '''
        mask_neg = mask_neg.float()
        mask_pos = mask_pos.float()
        num_neg = mask_neg.sum(-1)
        num_pos = mask_pos.sum(-1)
        '''
        mask_neg = mask_neg.reshape(-1, self.k)

        mask_pos = mask_pos.reshape(-1, self.k)
        #print(source.size())
        
        '''
        if num_neg<1:
            smoothness = edge_weight*mask_pos/num_pos
        else:
            smoothness = edge_weight*(mask_pos/num_pos-mask_neg/num_neg)
        '''
        #mask = (mask_pos-mask_neg)
        #print(edge_weight*mask_pos/num_pos)
        #smoothness = mask*edge_weight
        edge_weight = edge_weight.reshape(-1, self.k)
        mask_negf = mask_neg.float()
        num_neg = mask_negf.sum(1)
        mask_posf = mask_pos.float()
        num_pos = mask_posf.sum(1)
        smoothness = torch.exp_((edge_weight.mul(mask_posf).sum(1))/num_pos)+torch.exp_(-1*(edge_weight.mul(mask_negf).sum(1))/num_neg)
        smoothness = smoothness.sum(0)
        batchsize = edge_weight.size(0)
        dist_ap, dist_an=[],[]
        for i in range(batchsize):
            dist_ap.append(edge_weight[i][mask_pos[i]].max().unsqueeze(0))
            dist_an.append(edge_weight[i][mask_neg[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        #return torch.log(smoothness+2)
        return torch.log(smoothness+2)+2*torch.exp(self.ranking_loss(dist_an, dist_ap , y))