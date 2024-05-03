import numpy as np
import torch
import torch.nn as nn

import config
from .graph2seq import Graph2SeqModule
from .vocab import load_or_init
from utils import Dataset, Logger


class Model(object):
    def __init__(self, train_data: Dataset, device: torch.device, logger: Logger) -> None:
        self.logger = logger
        self.module = Graph2SeqModule
        self.logger.log(f"Running {self.module.__name__}")
        self.vocab_model = load_or_init(train_data, logger)
        self.vocab_model.node_id_vocab = None
        self.vocab_model.node_type_id_vocab = None
        self.vocab_model.edge_type_id_vocab = None
        assert train_data is not None
        self._init_network(device)

    def _init_network(self, device: torch.device):
        word_embedding = self._init_embedding(
            self.vocab_model.word_vocab.vocab_size,
            self.vocab_model.word_vocab.embed_dim,
            self.vocab_model.word_vocab.embeddings,
        )
        self.logger.log(f"Instantiating NN module {self.module.__name__}...")
        self.network = self.module(
            word_embedding,
            self.vocab_model.word_vocab,
            self.logger,
            device,
        )

    def _init_embedding(self, vocab_size: int, embedding_size: int, embeddings: np.ndarray):
        self.logger.log(f"Initializing embedding for {vocab_size} words of size {embedding_size}...")
        return nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=self.vocab_model.word_vocab.pad_ind, # the embedding vector at this index doesn't get updated
            _weight=torch.from_numpy(embeddings).float()
        )