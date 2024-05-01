import torch

import config

from utils.metrics import AverageMeter

class ModelHandler(object):
    def __init__(self):
        self.train_loss = AverageMeter()
        # if config.USE_CUDA and torch.cuda.is_available():
