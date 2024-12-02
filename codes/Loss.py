import torch
import torch.nn.functional as F
from torch.nn import Sequential


class CorrMSELoss(torch.nn.Module):
    def __init__(self, weight_corr, weight_value):
        super(CorrMSELoss, self).__init__()
        self.weight_corr = -abs(weight_corr) / (abs(weight_corr) + abs(weight_value))
        self.weight_value = abs(weight_value) / (abs(weight_corr) + abs(weight_value))

    def forward(self, pred, label):
        core = torch.concat((pred, label), dim=0).reshape(2, -1)
        corr = torch.corrcoef(core)[0, 1]
        value = torch.nn.MSELoss()(pred, label)
        return 10 * (self.weight_corr * corr + self.weight_value * value)