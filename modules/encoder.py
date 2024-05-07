import torch
import torch.nn as nn

from utils import dropout, Logger, send_to_device

class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size: int, hidden_size: int,
        *,
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
            # device = device #TODO try it?
        )

    def forward(self, x: torch.Tensor, xlen: torch.Tensor):
        """
            x is a 3-tensor of [batch_size * max_length * emb_dim]
            xlen is a 1-tensor of [batch_size]
        """
        self.logger.log("Entered LSTM forward pass!") # TODO remove
        sorted_xlen, indexes = torch.sort(xlen, dim=0, descending=True) # sort by batch size
        lst = sorted_xlen.data.tolist()
        x = nn.utils.rnn.pack_padded_sequence(x[indexes], lst, batch_first=True)

        h_0 = send_to_device(
            torch.zeros(self.num_directions * self.num_layers, xlen.size(0), self.hidden_size),
            self.device,
        )
        c_0 = send_to_device(
            torch.zeros(self.num_directions * self.num_layers, xlen.size(0), self.hidden_size),
            self.device
        )

        packed_h, (packed_h_t, packed_c_t) = self.model(x, (h_0, c_0))
        packed_h_t: torch.Tensor = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
        packed_c_t: torch.Tensor = torch.cat((packed_c_t[-1], packed_c_t[-2]), 1)

        _, inverse_indexes = torch.sort(indexes, 0)

        hh, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)
        hh_inv: torch.Tensor = hh[inverse_indexes]
        hh_inv = dropout(hh_inv, self.rnn_dropout, shared_axes=[-2], training=self.training)
        hh_inv = hh_inv.transpose(0, 1) # [max_length, batch_size, emb_dim]

        packed_h_t_inv: torch.Tensor = packed_h_t[inverse_indexes]
        packed_h_t_inv = dropout(packed_h_t_inv, self.rnn_dropout, training=self.training)
        packed_h_t_inv = packed_h_t_inv.unsqueeze(0) # [1, batch_size, emb_dim]

        packed_c_t_inv: torch.Tensor = packed_c_t[inverse_indexes]
        packed_c_t_inv = dropout(packed_c_t_inv, self.rnn_dropout, training=self.training)
        packed_c_t_inv = packed_c_t_inv.unsqueeze(0) # [1, batch_size, emb_dim]
        rnn_state_t = (packed_h_t_inv, packed_c_t_inv)

        return hh_inv, rnn_state_t
