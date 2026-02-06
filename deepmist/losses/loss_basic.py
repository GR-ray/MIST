import torch
import torch.nn as nn


class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        intersection = pred * mask
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        iou = (intersection_sum + self.smooth) / (pred_sum + mask_sum - intersection_sum + self.smooth)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            return 1 - iou.sum()
        else:
            raise NotImplementedError(f'reduction type {self.reduction} not implemented')


class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask, dim=(1, 2, 3))
        total_sum = torch.sum((pred + mask), dim=(1, 2, 3))
        dice = 2 * intersection / (total_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else:
            raise NotImplementedError(f'reduction type {self.reduction} not implemented')


class BceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, mask):
        loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return loss_fn(pred, mask)


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, mask):
        loss_fn = nn.L1Loss(reduction=self.reduction)
        return loss_fn(pred, mask)
