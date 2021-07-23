import torch.nn as nn
import torch
import lib.model.faster_rcnn.dynamics as dynamics
import torch.nn.functional as F
from model.faster_rcnn.gcn import GCN


class DynamicGCN(nn.Module):
    def __init__(self, total_classes, device='cuda:0'):
        super(DynamicGCN, self).__init__()
        self.classes = total_classes
        self.device = device
        if self.classes == 81:
            self.gcn = GCN(in_channels=81, out_channels=81, hidden_layers='81,81,81,81')
        else:
            self.gcn = GCN(in_channels=21, out_channels=21, hidden_layers='21')

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def _get_W(self, x):
        x = (x - x.mean(dim=1).unsqueeze(1))
        norms = x.norm(dim=1)
        W = torch.mm(x, x.t()) / torch.ger(norms, norms)
        W = self.set_negative_to_zero(W.cuda())
        return W

    def forward(self, gcn_input, anchor_len, anchor_labels, probs):
        # 根据support和RoI特征计算相似度
        similarity = self._get_W(gcn_input)
        # 图的结点个数
        num_nodes = len(gcn_input)
        device = torch.cuda.current_device()
        # 构造每个结点的初始概率分布
        prob_init = torch.zeros(num_nodes, self.classes).cuda(device)
        anchor_id = range(anchor_len)
        cls_id = range(anchor_len, num_nodes)
        # RoI的预测概率作为embedding
        prob_init[cls_id, :] = probs[cls_id, :]
        # support image标签作为embedding
        prob_init[anchor_id, anchor_labels] = 1.
        try:
            assert torch.allclose(prob_init.sum(dim=1), torch.ones(num_nodes).cuda())
        except:
            print(prob_init[cls_id, :])
            print(prob_init[anchor_id, :])
        prob_gcn = self.gcn(x=prob_init, edges=similarity)
        return prob_gcn
