import torch
import torch.nn as nn


class SufficiencyLoss(nn.Module):
    def __init__(self, temperature=1, reduction='mean'):
        super(SufficiencyLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.weights = [1, 1, 1, 1, 1]

    def forward(self, pred_z_list, pred_v_list, mask):
        num_preds = len(pred_z_list)
        sufficiency_loss = 0
        for i in range(num_preds):
            pred_z = pred_z_list[i]
            pred_v = pred_v_list[i]

            b, c, _, _ = pred_z.shape

            ce = nn.BCEWithLogitsLoss(reduction=self.reduction)
            ce_loss = ce(pred_z, mask)

            pred_z = pred_z.reshape(b * c, -1)
            pred_v = pred_v.reshape(b * c, -1)

            kld = torch.nn.KLDivLoss()
            kld_loss = kld(self.log_softmax(pred_v.detach() / self.temperature),
                           self.softmax(pred_z / self.temperature))

            loss = (kld_loss + ce_loss) / 2
            sufficiency_loss += (loss * self.weights[i])

        return sufficiency_loss / num_preds
