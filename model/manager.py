import torch

import config

from . import model
from utils import loading, logging, metrics, processing

class ModelManager(object):
    def __init__(self):
        self.dev_loss = metrics.AverageMeter()
        self.train_loss = metrics.AverageMeter()
        # self.dev_metrics = {} # TODO
        # self.train_metrics = {} # TODO
        self.logger = logging.Logger()
        self.run_log = self.logger.run_log
        self.metrics_log = self.logger.metrics_log

        device_count = torch.cuda.device_count()
        self.logger.log(f"USE_CUDA is True, device count = {device_count}")

        if config.USE_CUDA and device_count > 0:
            self.device = torch.device(f"cuda:{config.CUDA_DEVICE_ID}")
        else:
            self.device = torch.device("cpu")

        datasets = loading.get_datasets(self.logger)
        train_data, dev_data, test_data = datasets["train"], datasets["dev"], datasets["test"]

        datastream = processing.DataStream # TODO
        self.vectorize_input = processing.vectorize_input # TODO

        self.n_train_examples = 0
        self.model = model.Model(train_data, self.device, self.logger)
