# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from network.head import *
from network.resnet import *
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class WCL(nn.Module):
    def __init__(self, dim_hidden=4096, dim=256):
        super(WCL, self).__init__()
        self.net = resnet50()
        self.head1 = ProjectionHead(dim_in=2048, dim_out=dim, dim_hidden=dim_hidden)
        self.head2 = ProjectionHead(dim_in=2048, dim_out=dim, dim_hidden=dim_hidden)

    @torch.no_grad()
    def build_connected_component(self, dist):
        b = dist.size(0)
        dist = dist - torch.eye(b, b, device='cuda') * 2
        x = torch.arange(b, device='cuda').unsqueeze(1).repeat(1,1).flatten()
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels, device='cuda')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, x1, x2, t=0.1):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        b = x1.size(0)
        bakcbone_feat1 = self.net(x1)
        bakcbone_feat2 = self.net(x2)
        feat1 = F.normalize(self.head1(bakcbone_feat1))
        feat2 = F.normalize(self.head1(bakcbone_feat2))

        other1 = concat_other_gather(feat1)
        other2 = concat_other_gather(feat2)

        prob = torch.cat([feat1, feat2]) @ torch.cat([feat1, feat2, other1, other2]).T / t
        diagnal_mask = (1 - torch.eye(prob.size(0), prob.size(1), device='cuda')).bool()
        logits = torch.masked_select(prob, diagnal_mask).reshape(prob.size(0), -1)

        first_half_label = torch.arange(b-1, 2*b-1).long().cuda()
        second_half_label = torch.arange(0, b).long().cuda()
        labels = torch.cat([first_half_label, second_half_label])

        feat1 = F.normalize(self.head2(bakcbone_feat1))
        feat2 = F.normalize(self.head2(bakcbone_feat2))
        all_feat1 = concat_all_gather(feat1)
        all_feat2 = concat_all_gather(feat2)
        all_bs = all_feat1.size(0)

        mask1_list = []
        mask2_list = []
        if rank == 0:
            mask1 = self.build_connected_component(all_feat1 @ all_feat1.T).float()
            mask2 = self.build_connected_component(all_feat2 @ all_feat2.T).float()
            mask1_list = list(torch.chunk(mask1, world_size))
            mask2_list = list(torch.chunk(mask2, world_size))
            mask1 = mask1_list[0]
            mask2 = mask2_list[0]
        else:
            mask1 = torch.zeros(b, all_bs, device='cuda')
            mask2 = torch.zeros(b, all_bs, device='cuda')
        torch.distributed.scatter(mask1, mask1_list, 0)
        torch.distributed.scatter(mask2, mask2_list, 0)

        diagnal_mask = torch.eye(all_bs, all_bs, device='cuda')
        diagnal_mask = torch.chunk(diagnal_mask, world_size)[rank]
        graph_loss =  self.sup_contra(feat1 @ all_feat1.T / t, mask2, diagnal_mask)
        graph_loss += self.sup_contra(feat2 @ all_feat2.T / t, mask1, diagnal_mask)
        graph_loss /= 2
        return logits, labels, graph_loss



# utils
@torch.no_grad()
def concat_other_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    other = torch.cat(tensors_gather[:rank] + tensors_gather[rank+1:], dim=0)
    return other



@torch.no_grad()
def concat_all_gather(tensor, replace=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    if replace:
        tensors_gather[rank] = tensor
    other = torch.cat(tensors_gather, dim=0)
    return other


