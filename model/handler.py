import torch

import config

from utils import data_loading, logging, metrics

class ModelHandler(object):
    def __init__(self):
        self.dev_loss = metrics.AverageMeter()
        self.train_loss = metrics.AverageMeter()
        # self.dev_metrics = {} # TODO
        # self.train_metrics = {} # TODO
        # if config.USE_CUDA and torch.cuda.is_available():
        self.logger = logging.Logger(config.LOG_DIR)
        self.run_log = self.logger.run_log
        self.metrics_log = self.logger.metrics_log

        device_count = torch.cuda.device_count()
        self.logger.log(f"USE_CUDA is True, device count = {device_count}", self.run_log)

        if config.USE_CUDA and device_count > 0:
            self.device = torch.device(f"cuda:{config.CUDA_DEVICE_ID}")
        else:
            self.device = torch.device("cpu")

        data_loading.get_datasets(self.logger)
