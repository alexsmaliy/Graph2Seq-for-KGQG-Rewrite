import torch

import config

from utils import logging, metrics

class ModelHandler(object):
    def __init__(self):
        self.dev_loss = metrics.AverageMeter()
        self.train_loss = metrics.AverageMeter()
        # self.dev_metrics = {} # TODO
        # self.train_metrics = {} # TODO
        # if config.USE_CUDA and torch.cuda.is_available():
        self.logger = logging.Logger(config.LOG_DIR)
