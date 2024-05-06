import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from modules import Graph2SeqModule
from modules.vocab import load_or_init
from utils import Dataset, Logger

class Model(object):
    def __init__(self, train_data: Dataset, device: torch.device, logger: Logger) -> None:
        self.logger = logger
        self.device = device
        self.module = Graph2SeqModule
        self.logger.log(f"Running {self.module.__name__}")
        self.vocab_model = load_or_init(train_data, logger)
        self.vocab_model.node_id_vocab = None
        self.vocab_model.node_type_id_vocab = None
        self.vocab_model.edge_type_id_vocab = None

        if config.PRETRAINED:
            state_dict_opt = self.init_saved_network(config.PRETRAINED)
        else:
            assert train_data is not None
            self._init_network(self.device)

        num_params = 0
        for name, p in self.network.named_parameters():
            self.logger.log(f"{name}: {p.size()}")
            num_params += p.numel()
        self.logger.log(f"#Parameters = {num_params}")

        self.criterion = nn.NLLLoss(ignore_index=self.vocab_model.word_vocab.pad_ind)
        self._init_optimizer()
        self.wmd = None

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

    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, config.SAVED_WEIGHTS_FILE)
        self.logger.log(f"Loading saved model: {fname}")

        saved_params = torch.load(str(fname), map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        self.saved_epoch = saved_params.get('epoch', 0)

        word_embedding = self._init_embedding(
            self.vocab_model.word_vocab.vocab_size,
            config.WORD_EMBED_DIM,
        )
        self.network = self.module(word_embedding, self.vocab_model.word_vocab, self.logger, self.device)

        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

        return state_dict.get('optimizer', None) if state_dict else None

    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, str(os.path.join(dirname, config.SAVED_WEIGHTS_FILE)))
        except BaseException:
            self.logger.log(f"Tried to save model to {dirname}/{config.SAVED_WEIGHTS_FILE}, but failed!")

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=config.LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True,
        )

    def _init_embedding(self, vocab_size: int, embedding_size: int, embeddings: np.ndarray = None):
        self.logger.log(f"Initializing embedding for {vocab_size} words of size {embedding_size}...")
        return nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=self.vocab_model.word_vocab.pad_ind, # the embedding vector at this index doesn't get updated
            _weight=torch.from_numpy(embeddings).float() if embeddings is not None else None,
        )
