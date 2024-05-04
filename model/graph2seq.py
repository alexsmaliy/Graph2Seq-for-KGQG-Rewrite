import torch
from torch import nn

import config
from .decoder import RNNDecoder
from .encoder import EncoderRNN, GraphNN
from .vocab import Vocabulary
from utils import Logger

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
        enc_hidden_size = config.RNN_SIZE
        enc_rnn_dropout = config.ENC_RNN_DROPOUT
        graph_hidden_size = config.GRAPH_HIDDEN_SIZE
        word_embed_dim = config.WORD_EMBED_DIM

        # if RNN == LSTM
        self.logger.log(f"ENC-DEC adapter is 2 linear layers of ({graph_hidden_size} x {dec_hidden_size})")
        self.enc_dec_adapter = nn.ModuleList([
            nn.Linear(graph_hidden_size, dec_hidden_size) for _ in range(2)
        ])

        self.node_name_word_encoder = EncoderRNN(
            word_embed_dim,
            enc_hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_type='lstm',
            rnn_dropout=enc_rnn_dropout,
            device=self.device,
        )
        self.edge_type_word_encoder = EncoderRNN(
            word_embed_dim,
            enc_hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_type='lstm',
            rnn_dropout=enc_rnn_dropout,
            device=self.device,
        )
        self.graph_encoder = GraphNN() #TODO
        self.decoder = RNNDecoder() #TODO
