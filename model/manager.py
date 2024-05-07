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

        self.vectorize_input = processing.vectorize_input
        self.n_train_examples = 0
        self.model = model.Model(train_data, self.device, self.logger)
        self.model.network = self.model.network.to(self.device)

        if train_data:
            self.train_loader = processing.DataStream(
                train_data,
                self.model.vocab_model.word_vocab,
                is_shuffle=True, is_loop=True, is_sort=True, ext_vocab=True,
            )
            self._n_train_batches = self.train_loader.get_num_batch()
        else:
            self.train_loader = None

        if dev_data:
            self.dev_loader = processing.DataStream(
                dev_data,
                self.model.vocab_model.word_vocab,
                is_shuffle=False, is_loop=True, is_sort=True, ext_vocab=True,
            )
            self._n_dev_batches = self.dev_loader.get_num_batch()
        else:
            self.dev_loader = None

        if test_data:
            self.test_loader = processing.DataStream(
                test_data,
                self.model.vocab_model.word_vocab,
                is_shuffle=False, is_loop=False, is_sort=True, batch_size=config.BATCH_SIZE, ext_vocab=True,
            )
            self._n_test_batches = self.test_loader.get_num_batch()
            self._n_test_examples = len(test_data)
        else:
            self.test_loader = None

        self.is_test = False
