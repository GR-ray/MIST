import numpy as np
import torch
from skimage import measure


class mIoUMetric:
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.reset()

    def update(self, pred, mask):
        correct, labeled = self.batch_pix_accuracy(pred, mask, self.threshold)
        inter, union = self.batch_intersection_union(pred, mask, self.threshold)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pix_acc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pix_acc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    @staticmethod
    def batch_pix_accuracy(pred, mask, threshold):
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask.float(), axis=1)
        elif len(mask.shape) == 4:
            mask = mask.float()
        else:
            raise ValueError('Unknown mask dimension.')

        assert pred.shape == mask.shape, "The shapes of prediction and mask don't match."
        predict = (pred > threshold).float()  # 'pred > 0' equals to 'sigmoid(pred) > 0.5'
        pixel_labeled = (mask > 0).float().sum()
        pixel_correct = (((predict == mask).float()) * ((mask > 0).float())).sum()
        assert pixel_correct <= pixel_labeled, 'Correct area should be smaller than labeled.'
        return pixel_correct, pixel_labeled

    @staticmethod
    def batch_intersection_union(pred, mask, threshold):
        mini = 1
        maxi = 1
        bins = 1
        predict = (pred > threshold).float()  # 'pred > 0' equals to 'sigmoid(pred) > 0.5'
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask.float(), axis=1)
        elif len(mask.shape) == 4:
            mask = mask.float()
        else:
            raise ValueError('Unknown mask dimension.')
        intersection = predict * ((predict == mask).float())

        area_inter, _ = np.histogram(intersection.cpu(), bins=bins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict.cpu(), bins=bins, range=(mini, maxi))
        area_mask, _ = np.histogram(mask.cpu(), bins=bins, range=(mini, maxi))
        area_union = area_pred + area_mask - area_inter

        assert (area_inter <= area_union).all(), 'Error: intersection area should be smaller than union area.'
        return area_inter, area_union


class nIoUMetric:
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.reset()

    def update(self, pred, mask):
        inter_arr, union_arr = self.batch_intersection_union(pred, mask, self.threshold)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        nIoU = IoU.mean()
        Detect = np.where(IoU > 0.5, 1, 0)
        DR = Detect.mean()
        return nIoU, DR

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    @staticmethod
    def batch_intersection_union(pred, mask, threshold):
        mini = 1
        maxi = 1
        bins = 1
        predict = (pred > threshold).float()  # 'pred > 0' equals to 'sigmoid(pred) > 0.5'
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask.float(), axis=1)
        elif len(mask.shape) == 4:
            mask = mask.float()
        else:
            raise ValueError('Unknown mask dimension.')
        intersection = predict * ((predict == mask).float())

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_mask_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            area_inter, _ = np.histogram(intersection[b].cpu(), bins=bins, range=(mini, maxi))
            area_inter_arr[b] = area_inter
            area_pred, _ = np.histogram(predict[b].cpu(), bins=bins, range=(mini, maxi))
            area_pred_arr[b] = area_pred
            area_mask, _ = np.histogram(mask[b].cpu(), bins=bins, range=(mini, maxi))
            area_mask_arr[b] = area_mask
            area_union = area_pred + area_mask - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all(), 'Error: intersection area should be smaller than union area.'
        return area_inter_arr, area_union_arr


class PdFaMetric:
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.reset()

    def update(self, pred, mask):
        h, w = pred.shape[2], pred.shape[3]
        self.all_pixels += (h * w)
        predict = np.array((pred > self.threshold).cpu()).astype('int64')  # 'pred > 0' equals to 'sigmoid(pred) > 0.5'
        predict = np.reshape(predict, (h, w))
        mask = np.array(mask.cpu()).astype('int64')
        mask = np.reshape(mask, (h, w))

        image = measure.label(predict, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(mask, connectivity=2)
        coord_label = measure.regionprops(label)

        self.all_targets += len(coord_label)
        image_area_total = []
        image_area_match = []
        distance_match = []

        for k in range(len(coord_image)):
            area_image = np.array(coord_image[k].area)
            image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    distance_match.append(distance)
                    image_area_match.append(area_image)
                    del coord_image[m]
                    break

        dismatch = [x for x in image_area_total if x not in image_area_match]
        self.dismatch_pixels += np.sum(dismatch)
        self.match_targets += len(distance_match)

    def get(self):
        final_FA = self.dismatch_pixels / self.all_pixels
        final_PD = self.match_targets / self.all_targets
        return final_FA, final_PD

    def reset(self):
        self.dismatch_pixels = 0
        self.all_pixels = 0
        self.match_targets = 0
        self.all_targets = 0


class ROCMetric:
    def __init__(self, bins=10):
        self.bins = bins
        self.reset()

    def update(self, pred, mask):
        pred = torch.sigmoid(pred)
        for iBin in range(self.bins + 1):
            threshold = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg, i_class_pos = self.cal_tp_pos_fp_neg(pred, mask, threshold)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + np.spacing(1))
        fp_rates = self.fp_arr / (self.neg_arr + np.spacing(1))
        recall = self.tp_arr / (self.pos_arr + np.spacing(1))
        precision = self.tp_arr / (self.class_pos + np.spacing(1))
        f_score = 2 * precision * recall / (precision + recall + np.spacing(1))
        return tp_rates, fp_rates, recall, precision, f_score

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    @staticmethod
    def cal_tp_pos_fp_neg(pred, mask, threshold):
        predict = (pred > threshold).float()
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask.float(), axis=1)
        elif len(mask.shape) == 4:
            mask = mask.float()
        else:
            raise ValueError('Unknown mask dimension.')

        intersection = predict * ((predict == mask).float())
        tp = intersection.sum()
        fp = (predict * ((predict != mask).float())).sum()
        tn = ((1 - predict) * ((predict == mask).float())).sum()
        fn = (((predict != mask).float()) * (1 - predict)).sum()
        pos = tp + fn
        neg = fp + tn
        class_pos = tp + fp
        return tp, pos, fp, neg, class_pos
