import torch
import torch.nn as nn


class SLSIoULoss(nn.Module):
    def __init__(self, warm_epoch=5, with_shape=True):
        super(SLSIoULoss, self).__init__()
        self.warm_epoch = warm_epoch
        self.with_shape = with_shape

    def forward(self, pred_log, target, epoch):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > self.warm_epoch:
            siou_loss = alpha * loss
            if self.with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


class SDMLoss(nn.Module):
    def __init__(self, warm_epoch=5, with_shape=True, with_distance=True, dynamic=True, delta=0.5):
        super(SDMLoss, self).__init__()
        self.warm_epoch = warm_epoch
        self.with_shape = with_shape
        self.with_distance = with_distance
        self.dynamic = dynamic
        self.delta = delta

    def forward(self, pred_log, target, epoch):
        pred = torch.sigmoid(pred_log)
        h = pred.shape[2]
        w = pred.shape[3]
        smooth = 0.0

        R_oc = 512 * 512 / (w * h)
        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        if epoch > self.warm_epoch:
            siou_loss = alpha * loss
            if self.dynamic:
                lloss = LLoss(pred, target)
                beta = (target_sum * self.delta * R_oc) / 81
                # beta = torch.where(beta > self.delta, torch.tensor(self.delta), beta)
                # beta = torch.where(beta > self.delta, torch.tensor(self.delta, device=beta.device), beta)
                beta = torch.where(beta > self.delta,
                                   torch.tensor(self.delta, dtype=beta.dtype, device=beta.device), beta)
                beta = beta.mean()
                if self.with_distance:
                    loss = (1 + beta) * (1 - siou_loss.mean()) + (1 - beta) * lloss  # SDM loss
                else:
                    loss = 1 - siou_loss.mean()
            else:
                if self.with_distance:
                    lloss = LLoss(pred, target)
                    loss = 1 - siou_loss.mean() + lloss
                else:
                    loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)
    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan(pred_centery / (pred_centerx + smooth))
                                                           - torch.arctan(
            target_centery / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss
