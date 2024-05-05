import torch
from torch import nn

import config
import utils
from .decoder import DecoderRNN
from .encoder import EncoderRNN
from .graph_encoder import GraphNN
from .vocab import Vocabulary
from utils import Logger, send_to_device, UNK_TOKEN

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
            rnn_dropout=enc_rnn_dropout,
            device=self.device,
            logger=self.logger,
        )
        self.edge_type_word_encoder = EncoderRNN(
            word_embed_dim,
            enc_hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_dropout=enc_rnn_dropout,
            device=self.device,
            logger=self.logger,
        )

        self.graph_encoder = GraphNN(self.device, self.logger)

        self.decoder = DecoderRNN(
            self.vocab_size,
            config.WORD_EMBED_DIM,
            config.DEC_HIDDEN_SIZE,
            tied_embedding=self.word_embedding,
            device=self.device,
            logger=self.logger,
        )

    def filter_out_of_vocab(self, t, ext_vocab_size):
        """replace any OOV index in `tensor` with UNK token"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = t.clone()
            result[t >= self.vocab_size] = UNK_TOKEN
            return result
        return t

    def gather(self, input_tensor1, input_tensor2, num1, num2, max_num_graph_elements):
        input_tensor = torch.cat([input_tensor1, input_tensor2], 1)
        max_num1 = input_tensor1.size(1)
        index_tensor = []

        for i in range(input_tensor.size(0)):
            selected_index = list(
                range(num1[i].item())
            ) + list(
                range(max_num1, max_num1 + num2[i].item())
            )
            if len(selected_index) < max_num_graph_elements:
                selected_index += [
                    max_num_graph_elements - 1 for _ in range(max_num_graph_elements - len(selected_index))
                ]
            index_tensor.append(selected_index)

        index_tensor = send_to_device(
            torch.LongTensor(index_tensor).unsqueeze(-1).expand(-1, -1, input_tensor.size(-1)),
            self.device,
        )
        return torch.gather(input_tensor, 1, index_tensor)

    def get_coverage_vector(self, enc_attn_weights):
        """combine the past attention weights into one vector"""
        return torch.sum(torch.cat(enc_attn_weights), dim=0)

    def forward(self):
        pass
