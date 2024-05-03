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
        self.vocab_model.node_id_vocab = None
        self.vocab_model.node_type_id_vocab = None
        self.vocab_model.edge_type_id_vocab = None
        assert train_data is not None
        # self._init_network() # TODO
