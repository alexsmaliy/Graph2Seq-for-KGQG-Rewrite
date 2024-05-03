import torch
from torch import nn

import config
from utils import Logger
from .encoder import EncoderRNN
from .vocab import Vocabulary

class Graph2SeqModule(nn.Module):
    def __init__(self, word_embedding: nn.Embedding, word_vocab: Vocabulary, logger: Logger, device: torch.device):
        super(Graph2SeqModule, self).__init__()
        self.name = "Graph2Seq"
        self.device = device
        self.logger = logger
        self.word_embedding = word_embedding
        self.word_vocab = word_vocab
        self.vocab_size = word_vocab.vocab_size

        self.logger.log("Setting requires_grad = False on word embedding params.")
        for param in self.word_embedding.parameters():
            param.requires_grad_(False) # do not record gradient

        self.ans_match_embed = nn.Embedding(
            3,
            config.ANS_MATCH_EMB_DIM,
            padding_idx=word_vocab.pad_ind
        )

        dec_hidden_size = config.DEC_HIDDEN_SIZE
        # if RNN == LSTM
        ghs = config.GRAPH_HIDDEN_SIZE
        dhs = config.DEC_HIDDEN_SIZE
        self.logger.log(f"ENC-DEC adapter is 2 linear layers of {ghs} x {dhs}")
        self.enc_dec_adapter = nn.ModuleList([
            nn.Linear(ghs, dhs) for _ in range(2)
        ])

        self.node_name_word_encoder = EncoderRNN() #TODO
