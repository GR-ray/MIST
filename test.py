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
from scipy.io import savemat

from deepmist.datasets import build_dataset, DataLoaderX
from deepmist.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from deepmist.models import build_model, run_model
from deepmist.utils import ordered_yaml, make_dir


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


class Tester(object):
    def __init__(self, args):
        # seed
        set_seed(args.seed)

        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.val_dataset, self.val_hard_dataset = build_dataset(self.cfg['dataset'], mode='val')
        self.val_loader = DataLoaderX(self.val_dataset, batch_size=1,
                                      num_workers=self.cfg['test']['num_workers'], pin_memory=True)
        self.val_hard_loader = DataLoaderX(self.val_hard_dataset, batch_size=1,
                                           num_workers=self.cfg['test']['num_workers'], pin_memory=True)

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # metric
        self.mIoU_metric = mIoUMetric()
        self.nIoU_metric = nIoUMetric()
        self.PdFa_metric = PdFaMetric()
        self.bins = 200  # 10, 200
        self.ROC_metric = ROCMetric(bins=self.bins)

        # checkpoint
        checkpoint = torch.load(self.cfg['test']['checkpoint'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        print(f"Successfully load the checkpoint from {self.cfg['test']['checkpoint']}.\n")

        # visualize
        self.pred_vis_dir = os.path.join(self.cfg['test']['exp_root'], 'pred_vis')
        make_dir(self.pred_vis_dir)

    def test(self, split='all'):
        print(f"[testset]\n")
        self.model.eval()
        # reset metrics
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()
        self.PdFa_metric.reset()
        self.ROC_metric.reset()

        # initial inference time
        total_infer_time = 0

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
                if hasattr(self.model.module, 'use_sufficiency_loss') and self.model.module.use_sufficiency_loss:
                    iter_start_time = time.time()
                    preds, _, _ = run_model(self.model, self.model_name, True, frames)
                    iter_end_time = time.time()
                else:
                    iter_start_time = time.time()
                    preds = run_model(self.model, self.model_name, False, frames)
                    iter_end_time = time.time()

                # update inference time
                total_infer_time += (iter_end_time - iter_start_time)

                if not isinstance(preds, (list, tuple)):
                    preds = [preds]
                pred = preds[0]  # Note: distinguish between 0 and -1 when using deep supervision
                # Restore the original image size
                pred = pred[:, :, :h, :w]
                mask = mask[:, :, :h, :w]

                # update metrics
                self.mIoU_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.PdFa_metric.update(pred, mask)
                self.ROC_metric.update(pred, mask)

                # visualize predicted masks
                if split == 'all':
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

        # get FPS
        fps = (iter_idx + 1) / total_infer_time

        # print the test info
        message = f'\n[{split}]'
        message += f'[mIoU:{mIoU:.5f}][nIoU:{nIoU:.5f}]'
        message += f'[PD:{PD:.5f}][FA:{FA:.5e}]'
        message += f'[AUC:{AUC:.5f}][precision:{precision_05:.5f}][recall:{recall_05:.5f}][f_score:{f_score_05:.5f}]'
        message += f'[FPS:{fps:.5f}]'
        print(message)

        # save mat
        save_dict = {'mIoU': mIoU, 'nIoU': nIoU,
                     'PD': PD, 'FA': FA,
                     'tp_rates': tp_rates, 'fp_rates': fp_rates, 'AUC': AUC,
                     'precision': precision_05, 'recall': recall_05, 'f_score': f_score_05, 'FPS': fps}
        savemat(os.path.join(self.cfg['test']['exp_root'], f'{split}_metrics.mat'), save_dict)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    tester = Tester(args)
    tester.test(split='all')
    tester.test(split='hard')


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Test')
    parser.add_argument('--config', type=str,
                        default='./configs/test_MISTNet_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='0', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)
