import os
import time


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def make_dir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def make_exp_root(root):
    exp_root = os.path.join(root, get_time_str())
    make_dir(exp_root)
    return exp_root
