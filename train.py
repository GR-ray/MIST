import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import auc
from tqdm import tqdm

from deepmist.datasets import build_dataset, DataLoaderX
from deepmist.losses import build_loss
from deepmist.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from deepmist.models import build_model, run_model
from deepmist.utils import (ordered_yaml, set_optimizer, set_lr_scheduler, update_lr, get_current_lr, reset_loss_dict,
                            get_loss_dict, set_logger, log_train_iter_info, log_train_info, log_test_info, make_dir)


def set_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, args):
        # seed
        set_seed(args.seed)

        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # logger
        self.logger, self.tb_logger = set_logger(self.cfg)

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.train_dataset, self.val_dataset, self.val_hard_dataset = build_dataset(self.cfg['dataset'], mode='train')
        self.train_loader = DataLoaderX(self.train_dataset, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                                        num_workers=self.cfg['train']['num_workers'], pin_memory=True, drop_last=True)
        self.val_loader = DataLoaderX(self.val_dataset, batch_size=1,
                                      num_workers=self.cfg['train']['num_workers'], pin_memory=True)
        self.val_hard_loader = DataLoaderX(self.val_hard_dataset, batch_size=1,
                                           num_workers=self.cfg['train']['num_workers'], pin_memory=True)

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # optimizer
        optim_params = self.model.parameters()
        self.optimizer, self.init_lr = set_optimizer(optim_params, self.cfg['model']['optimizer'])

        # lr scheduler
        self.total_epochs = self.cfg['train']['total_epochs']
        self.iters_per_epoch = len(self.train_loader)
        self.total_iters = self.total_epochs * self.iters_per_epoch
        self.lr_scheduler, self.step_interval, self.warmup_iters = \
            set_lr_scheduler(self.optimizer, self.total_epochs, self.total_iters, self.cfg['model']['lr_scheduler'])

        # loss
        self.loss_fn, self.loss_weight, self.train_loss, self.train_iter_loss, self.test_loss, self.use_sufficiency_loss = \
            build_loss(self.cfg['model']['loss'])

        # metric
        self.mIoU_metric = mIoUMetric()
        self.nIoU_metric = nIoUMetric()
        self.PdFa_metric = PdFaMetric()
        self.bins = 10
        self.ROC_metric = ROCMetric(bins=self.bins)

        # resume
        self.cur_iter = 0
        self.cur_epoch = 1
        if self.cfg['train'].get('resume'):
            checkpoint = torch.load(self.cfg['train']['resume'], map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.cur_epoch = checkpoint['epoch'] + 1
            self.cur_iter = checkpoint['epoch'] * self.iters_per_epoch
            self.logger.info(f"resume training from epoch: {checkpoint['epoch']}")

        # other settings
        self.log_interval = self.cfg['train']['log_interval']
        self.save_interval = self.cfg['train']['save_interval']
        self.val_interval = self.cfg['train']['val_interval']
        self.pred_vis_dir = os.path.join(self.cfg['train']['exp_root'], 'pred_vis')
        self.checkpoint_dir = os.path.join(self.cfg['train']['exp_root'], 'checkpoints')
        make_dir(self.pred_vis_dir)
        make_dir(self.checkpoint_dir)
        self.validation_history = []

    def train(self, epoch):
        self.logger.info(f"[{self.cfg['train']['exp_name']}][epoch:{epoch}][trainset]\n")
        self.model.train()
        epoch_st_time = iter_st_time = time.time()
        for iter_idx, data in enumerate(self.train_loader):
            self.cur_iter += 1
            # update learning rate
            update_lr(self.optimizer, self.init_lr, self.lr_scheduler, self.step_interval, self.warmup_iters,
                      self.cur_iter, iter_idx)

            # optimize one iter
            frames, mask, _, _, _ = data
            frames, mask = frames.to(self.device), mask.to(self.device)
            if self.use_sufficiency_loss:
                preds, pred_z_list, pred_v_list = run_model(self.model, self.model_name, self.use_sufficiency_loss, frames)
            else:
                preds = run_model(self.model, self.model_name, self.use_sufficiency_loss, frames)
            total_loss = 0.  # total loss of one iter
            if not isinstance(preds, (list, tuple)):
                preds = [preds]
            for loss_type, criterion in self.loss_fn.items():
                if loss_type == 'SufficiencyLoss':
                    loss = criterion(pred_z_list, pred_v_list, mask) * self.loss_weight[loss_type]
                    self.train_loss[loss_type] += loss.detach().clone()
                    self.train_iter_loss[loss_type] += loss.detach().clone()
                    total_loss += loss
                else:
                    for i, pred in enumerate(preds):
                        if loss_type in ['SLSIoULoss', 'SDMLoss']:
                            loss = criterion(pred, mask, epoch) * self.loss_weight[loss_type][i]
                        else:
                            loss = criterion(pred, mask) * self.loss_weight[loss_type][i]
                        self.train_loss[loss_type + '_' + str(i)] += loss.detach().clone()
                        self.train_iter_loss[loss_type + '_' + str(i)] += loss.detach().clone()
                        total_loss += loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # log iters within an interval
            if (iter_idx + 1) % self.cfg['train']['log_interval'] == 0:
                iter_time = time.time() - iter_st_time
                log_train_iter_info(epoch, iter_idx + 1, self.iters_per_epoch, get_current_lr(self.optimizer),
                                    self.cfg['train']['log_interval'], iter_time, get_loss_dict(self.train_iter_loss),
                                    self.logger)
                iter_st_time = time.time()
        reset_loss_dict(self.train_iter_loss)

        # log one epoch
        epoch_time = time.time() - epoch_st_time
        log_train_info(epoch, get_current_lr(self.optimizer), epoch_time, get_loss_dict(self.train_loss),
                       self.logger, self.tb_logger)

    def validate(self, epoch, split='all'):
        self.logger.info(f"[{self.cfg['train']['exp_name']}][epoch:{epoch}][testset]\n")
        self.model.eval()
        # reset metrics
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()
        self.PdFa_metric.reset()
        self.ROC_metric.reset()

        if split == 'all':
            val_loader = self.val_loader
        elif split == 'hard':
            val_loader = self.val_hard_loader
        else:
            raise ValueError(f"Invalid split '{split}'. It must be 'all' or 'hard'.")

        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(val_loader)):
                frames, mask, h, w, name = data
                frames, mask = frames.to(self.device), mask.to(self.device)
                if self.use_sufficiency_loss:
                    preds, pred_z_list, pred_v_list = run_model(self.model, self.model_name, self.use_sufficiency_loss, frames)
                else:
                    preds = run_model(self.model, self.model_name, self.use_sufficiency_loss, frames)
                if not isinstance(preds, (list, tuple)):
                    preds = [preds]
                if split == 'all':
                    for loss_type, criterion in self.loss_fn.items():
                        if loss_type == 'SufficiencyLoss':
                            loss = criterion(pred_z_list, pred_v_list, mask) * self.loss_weight[loss_type]
                            self.test_loss[loss_type] += loss.detach().clone()
                        else:
                            for i, pred in enumerate(preds):
                                if loss_type in ['SLSIoULoss', 'SDMLoss']:
                                    loss = criterion(pred, mask, epoch) * self.loss_weight[loss_type][i]
                                else:
                                    loss = criterion(pred, mask) * self.loss_weight[loss_type][i]
                                self.test_loss[loss_type + '_' + str(i)] += loss.detach().clone()

                pred = preds[0]  # Note: distinguish between 0 and -1 when using deep supervision
                # Restore the original image size
                pred = pred[:, :, :h, :w]
                mask = mask[:, :, :h, :w]

                # update metrics
                self.mIoU_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.PdFa_metric.update(pred, mask)
                self.ROC_metric.update(pred, mask)

                # visualize predicted masks (only the last epoch)
                if split == 'all' and epoch == self.total_epochs:
                    pred_sigmoid = torch.sigmoid(pred)
                    final_pred = pred_sigmoid.data.cpu().numpy()[0, 0, :, :]
                    mask_pred = Image.fromarray(np.uint8(final_pred * 255))
                    save_dir = os.path.join(self.pred_vis_dir, name[0].split('/')[0])
                    make_dir(save_dir)
                    mask_pred.save(os.path.join(self.pred_vis_dir, name[0]))

        # get metrics
        _, mIoU = self.mIoU_metric.get()
        nIoU, _ = self.nIoU_metric.get()
        FA, PD = self.PdFa_metric.get()
        tp_rates, fp_rates, recall, precision, f_score = self.ROC_metric.get()
        AUC = auc(fp_rates, tp_rates)
        precision_05 = precision[(self.bins + 1) // 2]
        recall_05 = recall[(self.bins + 1) // 2]
        f_score_05 = f_score[(self.bins + 1) // 2]

        # store the test info
        if split == 'all':
            self.validation_history.append({
                'epoch': epoch,
                'mIoU': mIoU,
                'nIoU': nIoU,
                'PD': PD,
                'FA': FA,
                'AUC': AUC,
                'precision': precision_05,
                'recall': recall_05,
                'f_score': f_score_05,
            })

        # log the test info
        test_loss = get_loss_dict(self.test_loss) if split == 'all' else None
        log_test_info(epoch, test_loss, mIoU, nIoU, PD, FA, AUC, precision_05, recall_05,
                      f_score_05, self.logger, self.tb_logger, split)

    def save_model(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'config': self.cfg
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Successfully saved the checkpoint in {checkpoint_path}.\n')

    def log_final_validation_summary(self):
        header = f"{'Epoch':<12} | {'mIoU':<12} | {'nIoU':<12} | {'PD':<12} | {'FA':<12} | {'AUC':<12} | {'F-Score':<12}"
        table_width = len(header)

        self.logger.info("=" * table_width)
        self.logger.info(f"FINAL VALIDATION RESULTS SUMMARY")
        self.logger.info("=" * table_width)
        
        self.logger.info(header)
        self.logger.info("-" * table_width)

        for results in self.validation_history:
            log_line = (
                f"{results['epoch']:<12} | "
                f"{results['mIoU']:<12.5f} | "
                f"{results['nIoU']:<12.5f} | "
                f"{results['PD']:<12.5f} | "
                f"{results['FA']:<12.5e} | "
                f"{results['AUC']:<12.5f} | "
                f"{results['f_score']:<12.5f}"
            )
            self.logger.info(log_line)

        self.logger.info("=" * table_width)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = Trainer(args)
    for epoch in range(trainer.cur_epoch, trainer.total_epochs + 1):
        # train
        trainer.train(epoch)
        # save
        if epoch % trainer.save_interval == 0:
            trainer.save_model(epoch)
        # val
        if epoch >= 30 and epoch % trainer.val_interval == 0:
            trainer.validate(epoch, split='all')
            # trainer.validate(epoch, split='hard')

    trainer.log_final_validation_summary()

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Training')
    parser.add_argument('--config', type=str,
                        default='./configs/train_MISTNet_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='the ids of gpus')

    return parser.parse_args()
 

if __name__ == '__main__':
    args = args_parser()
    main(args)
