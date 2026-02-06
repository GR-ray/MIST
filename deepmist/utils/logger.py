import logging

from deepmist.utils.file_and_path import *
from deepmist.utils.yaml_configs import dict2str


def set_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


def get_root_logger(logger_name='deepmist', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


def get_env_info():
    import torch
    import torchvision
    msg = ('\nVersion Information: '
           f'\n\tPyTorch: {torch.__version__}'
           f'\n\tTorchVision: {torchvision.__version__}')
    return msg


def set_logger(cfg):
    exp_name = cfg['train']['exp_name']
    exp_root = make_exp_root(os.path.join('./results', exp_name))
    cfg['train']['exp_root'] = exp_root

    log_file = os.path.join(exp_root, f'train_{exp_name}_{get_time_str()}.log')
    logger = get_root_logger(logger_name='deepmist', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(cfg))
    tb_logger = set_tb_logger(log_dir=os.path.join(exp_root, 'tb_log'))
    return logger, tb_logger


def log_train_iter_info(epoch, cur_iter, total_iter, lr, log_interval, iter_time, train_iter_loss, logger):
    message = f'[epoch:{epoch}, iter:{cur_iter}/{total_iter}, lr:{lr:.3e}][time ({str(log_interval)} iters):{iter_time:.3f}]\n'
    for loss_type, loss_value in train_iter_loss.items():
        message += f'[{loss_type}:{loss_value:.6f}]'
    message += '\n'
    logger.info(message)


def log_train_info(epoch, lr, epoch_time, train_loss, logger, tb_logger):
    message = f'[epoch:{epoch}, lr:{lr:.3e}][time (epoch):{epoch_time:.3f}]\n'
    for loss_type, loss_value in train_loss.items():
        message += f'[{loss_type}:{loss_value:.6f}]'
        tb_logger.add_scalar(f'train_losses/{loss_type}', loss_value, epoch)
    message += '\n'
    logger.info(message)


def log_test_info(epoch, test_loss, mIoU, nIoU, PD, FA, AUC, precision, recall, f_score, logger, tb_logger, partition):
    message = ''
    if partition == 'all':
        for loss_type, loss_value in test_loss.items():
            message += f'[{loss_type}:{loss_value:.6f}]'
            tb_logger.add_scalar(f'test_losses/{loss_type}', loss_value, epoch)
        tb_logger.add_scalar(f'test_metrics/mIoU', mIoU, epoch)
        tb_logger.add_scalar(f'test_metrics/nIoU', nIoU, epoch)
        tb_logger.add_scalar(f'test_metrics/PD', PD, epoch)
        tb_logger.add_scalar(f'test_metrics/FA', FA, epoch)
        tb_logger.add_scalar(f'test_metrics/AUC', AUC, epoch)
        tb_logger.add_scalar(f'test_metrics/precision', precision, epoch)
        tb_logger.add_scalar(f'test_metrics/recall', recall, epoch)
        tb_logger.add_scalar(f'test_metrics/f_score', f_score, epoch)
    message += f'\n[{partition}][mIoU:{mIoU:.5f}][nIoU:{nIoU:.5f}][PD:{PD:.5f}][FA:{FA:.5e}][AUC:{AUC:.5f}]' \
               f'[precision:{precision:.5f}][recall:{recall:.5f}][f_score:{f_score:.5f}]\n'
    logger.info(message)
