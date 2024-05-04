from typing import Literal

import torch
import torch.nn as nn

from utils import Logger

class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size: int, hidden_size: int, /,
        bidirectional: bool,
        num_layers: int,
        rnn_dropout: float,
        device: torch.device,
        logger: Logger,
    ):
        super(EncoderRNN, self).__init__()
        assert not bidirectional or (bidirectional and hidden_size % 2 == 0), \
            "bidi LSTM needs even-dimensional hidden layer!"

        self.logger = logger
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.input_size = input_size
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout

        self.logger.log(f"Building LSTM of ({input_size}x{hidden_size})")
        self.model = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            batch_first=True, bidirectional=bidirectional,
            # device = device # why not???
        )


class GraphNN(object):
    pass
