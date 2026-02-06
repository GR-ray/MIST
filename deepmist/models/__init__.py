import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from copy import deepcopy
from deepmist.models.multiframe.MISTNet.model_MISTNet import MISTNet


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def build_model(model_cfg):
    model_cfg = deepcopy(model_cfg)
    model_name = model_cfg.pop('name')
    # multi-frame
    if model_name == 'MISTNet':
        model = MISTNet(**model_cfg)
    else:
        raise NotImplementedError(f"Invalid model name '{model_name}'.")
    # model.apply(init_weights)
    return model, model_name


def run_model(model, model_name, use_sufficiency_loss, frames):
    # multi-frame
    if model_name in ['MISTNet']:
        if use_sufficiency_loss:
            preds, pred_z_list, pred_v_list = model(frames)
            return preds, pred_z_list, pred_v_list
        else:
            preds = model(frames)
    else:
        raise NotImplementedError(f"Invalid model name '{model_name}'.")
    return preds
