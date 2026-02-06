from copy import deepcopy

import torch

from deepmist.utils.yaml_configs import dict_wrapper


def linear_annealing(init, fin, step, annealing_steps):
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def set_optimizer(optim_params, optim_cfg):
    optim_cfg = deepcopy(optim_cfg)
    optim_type = optim_cfg.pop('type')
    init_lr = optim_cfg.pop('init_lr')
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(optim_params, lr=init_lr, **optim_cfg)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(optim_params, lr=init_lr, **optim_cfg)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(optim_params, lr=init_lr, **optim_cfg)
    elif optim_type == 'Adagrad':
        optimizer = torch.optim.Adagrad(optim_params, lr=init_lr, **optim_cfg)
    else:
        raise NotImplementedError(
            f"Invalid optimizer type '{optim_type}'. Only SGD, Adam, AdamW and Adagrad are supported.")
    return optimizer, init_lr


def set_lr_scheduler(optimizer, total_epochs, total_iters, lr_sche_cfg):
    warmup_iters = lr_sche_cfg['warmup_iters']
    sche_cfg = deepcopy(lr_sche_cfg['scheduler'])
    lr_sche_type = sche_cfg.pop('type')
    step_interval = sche_cfg.pop('step_interval')
    if step_interval == 'epoch':
        T_max = total_epochs
    elif step_interval == 'iter':
        T_max = total_iters
    else:
        return ValueError(f"Scheduler step_interval '{step_interval}' error.")

    if lr_sche_type == 'LambdaLR':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / total_iters) ** 0.9)
    elif lr_sche_type == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_cfg)
    elif lr_sche_type == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **sche_cfg)
    elif lr_sche_type == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, **sche_cfg)
    else:
        raise NotImplementedError(
            f"Invalid lr scheduler type '{lr_sche_type}'."
            f" Only LambdaLR, StepLR, MultiStepLR and CosineAnnealingLR are supported.")

    return lr_scheduler, step_interval, warmup_iters


def update_lr(optimizer, init_lr, lr_scheduler, step_interval, warmup_iters, cur_iter, iter_idx):
    if cur_iter > 1:
        if step_interval == 'epoch':
            if iter_idx == 0:
                lr_scheduler.step()
        elif step_interval == 'iter':
            lr_scheduler.step()
        else:
            return ValueError(f'Scheduler step_interval {step_interval} error.')

    # set up warm-up learning rate
    if cur_iter <= warmup_iters:
        # get initial lr for each group
        # modify warming-up learning rates
        # currently only support linearly warm up
        warmup_lr = init_lr / warmup_iters * cur_iter
        # set learning rate
        for param in optimizer.param_groups:
            param['lr'] = warmup_lr


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def reset_loss_dict(loss_dict):
    for k in loss_dict.keys():
        loss_dict[k] = 0.


def get_loss_dict(loss_dict):
    loss_dict_cp = deepcopy(loss_dict)
    reset_loss_dict(loss_dict)
    return dict_wrapper(loss_dict_cp)
