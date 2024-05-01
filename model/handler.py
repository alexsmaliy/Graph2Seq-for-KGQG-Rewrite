import torch

import config

from utils.metrics import AverageMeter

class ModelHandler(object):
    def __init__(self):
        self.dev_loss = AverageMeter()
        self.train_loss = AverageMeter()
        # self.dev_metrics = {} # TODO
        # self.train_metrics = {} # TODO
        # if config.USE_CUDA and torch.cuda.is_available():
