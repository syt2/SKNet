import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim
#
#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
#
# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()
#
#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer('one_hot', one_hot.unsqueeze(0))
#
#         self.confidence = 1.0 - label_smoothing
#
#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
#
#         return F.kl_div(output, model_prob, reduction='sum')
