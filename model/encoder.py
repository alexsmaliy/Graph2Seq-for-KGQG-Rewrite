from typing import Literal

import torch
import torch.nn as nn

RNNType = Literal["lstm"] | Literal["gru"]

class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size: int, hidden_size: int, /,
        bidirectional: bool,
        num_layers: int,
        rnn_type: RNNType,
        rnn_dropout: float,
        device: torch.device,
    ):
        super(EncoderRNN, self).__init__()
        pass

class GraphNN(object):
    pass
