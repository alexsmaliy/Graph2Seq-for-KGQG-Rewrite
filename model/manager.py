import torch
import torch.backends.cudnn as cudnn

import config

from . import model
from utils import Dataset, Datasets, loading, logging, metrics, processing

class ModelManager(object):
    def __init__(self):
        self.dev_loss = metrics.AverageMeter()
        self.train_loss = metrics.AverageMeter()

        self._train_metrics = {
            "Bleu_1": metrics.AverageMeter(),
            "Bleu_2": metrics.AverageMeter(),
            "Bleu_3": metrics.AverageMeter(),
            "Bleu_4": metrics.AverageMeter(),
            "ROUGE_L": metrics.AverageMeter(),
        }
        self._dev_metrics = {
            "Bleu_1": metrics.AverageMeter(),
            "Bleu_2": metrics.AverageMeter(),
            "Bleu_3": metrics.AverageMeter(),
            "Bleu_4": metrics.AverageMeter(),
            "ROUGE_L": metrics.AverageMeter(),
        }

        self.logger = logging.Logger()
        self.run_log = self.logger.run_log
        self.metrics_log = self.logger.metrics_log

        device_count = torch.cuda.device_count()
        self.logger.log(f"USE_CUDA is True, device count = {device_count}")

        if config.USE_CUDA and device_count > 0:
            self.device = torch.device(f"cuda:{config.CUDA_DEVICE_ID}")
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        datasets = loading.get_datasets(self.logger)
        train_data: Dataset = datasets["train"]
        dev_data: Dataset = datasets["dev"]
        test_data: Dataset = datasets["test"]

        datastream = processing.DataStream # TODO
        self.vectorize_input = processing.vectorize_input

        self.n_train_examples = 0
        self.model = model.Model(train_data, self.device, self.logger)
        self.model.network = self.model.network.to(self.device)
        self.is_test = False
