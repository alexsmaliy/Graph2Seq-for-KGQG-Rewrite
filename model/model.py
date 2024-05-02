from torch import device

import config
from .graph2seq import Graph2SeqModule
from .vocab import load_or_init
from utils import Dataset, Logger


class Model(object):
    def __init__(self, train_data: Dataset, device: device, logger: Logger) -> None:
        self.module = Graph2SeqModule
        logger.log(f"Running {self.module.__name__}")
        self.vocab_model = load_or_init(train_data, logger)
