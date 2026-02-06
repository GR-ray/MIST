from copy import deepcopy
from deepmist.losses.loss_basic import SoftIoULoss, DiceLoss, BceLoss, L1Loss
from deepmist.losses.loss_sufficiency import SufficiencyLoss
from deepmist.losses.loss_sls_iou_sdm import SLSIoULoss, SDMLoss


def build_loss(loss_cfg):
    loss_cfg = deepcopy(loss_cfg)
    num_preds = loss_cfg.pop('num_preds')
    loss_fn = {}
    loss_weight = {}
    train_loss = {}
    train_iter_loss = {}
    test_loss = {}
    use_sufficiency_loss = False
    for loss in loss_cfg:
        loss_fn[loss_cfg[loss]['type']] = choose_loss(loss_cfg[loss])
        loss_weight[loss_cfg[loss]['type']] = loss_cfg[loss]['weight']
        if loss_cfg[loss]['type'] == 'SufficiencyLoss':
            use_sufficiency_loss = True
            train_loss[loss_cfg[loss]['type']] = 0.
            train_iter_loss[loss_cfg[loss]['type']] = 0.
            test_loss[loss_cfg[loss]['type']] = 0.
        else:
            if not isinstance(loss_weight[loss_cfg[loss]['type']], (tuple, list)):
                loss_weight[loss_cfg[loss]['type']] = [loss_weight[loss_cfg[loss]['type']]] * num_preds
            assert len(loss_weight[loss_cfg[loss]['type']]) == num_preds
            for idx in range(num_preds):
                train_loss[loss_cfg[loss]['type'] + '_' + str(idx)] = 0.
                train_iter_loss[loss_cfg[loss]['type'] + '_' + str(idx)] = 0.
                test_loss[loss_cfg[loss]['type'] + '_' + str(idx)] = 0.
    assert len(loss_fn) > 0
    return loss_fn, loss_weight, train_loss, train_iter_loss, test_loss, use_sufficiency_loss


def choose_loss(sub_loss_cfg):
    sub_loss_cfg = deepcopy(sub_loss_cfg)
    loss_type = sub_loss_cfg.pop('type')
    sub_loss_cfg.pop('weight')
    if loss_type == 'SoftIoULoss':
        criterion = SoftIoULoss(**sub_loss_cfg)
    elif loss_type == 'DiceLoss':
        criterion = DiceLoss(**sub_loss_cfg)
    elif loss_type == 'BceLoss':
        criterion = BceLoss(**sub_loss_cfg)
    elif loss_type == 'L1Loss':
        criterion = L1Loss(**sub_loss_cfg)
    elif loss_type == 'SufficiencyLoss':
        criterion = SufficiencyLoss(**sub_loss_cfg)
    elif loss_type == 'SLSIoULoss':
        criterion = SLSIoULoss(**sub_loss_cfg)
    elif loss_type == 'SDMLoss':
        criterion = SDMLoss(**sub_loss_cfg)
    else:
        raise NotImplementedError(
            f"Invalid loss type '{loss_type}'. Only SoftIoULoss, DiceLoss, BceLoss, L1Loss, SufficiencyLoss, SLSIoULoss, and SDMLoss are supported.")
    return criterion
