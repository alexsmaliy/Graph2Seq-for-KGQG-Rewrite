import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .attention import AttentionModule
from utils import dropout, EPS, INF, Logger, send_to_device

class DecoderRNN(nn.Module):
    def __init__(self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        *,
        tied_embedding: nn.Embedding,
        device: torch.device,
        logger: Logger,
    ):
        super(DecoderRNN, self).__init__()
        self.logger = logger
        self.dec_attn = False
        self.device = device
        self.enc_attn = True
        self.enc_attn_cover = True # ???
        self.enc_hidden_size = config.GRAPH_HIDDEN_SIZE
        self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.in_drop = 0.0
        self.out_drop = 0.0
        self.out_embed_size = None
        self.pointer = True
        self.rnn_drop = 0.3
        self.rnn_type = 'lstm'
        self.vocab_size = vocab_size

        self.model = nn.LSTM(embed_size, self.hidden_size)
        self.fc_dec_input = nn.Linear(self.enc_hidden_size + embed_size, embed_size)
        self.enc_attn_fn = AttentionModule(self.hidden_size, 2 * self.hidden_size, self.enc_hidden_size)
        self.combined_size += self.enc_hidden_size

        cover_weight = torch.Tensor(1, 1, self.hidden_size)
        self.cover_weight = nn.Parameter(nn.init.xavier_uniform_(cover_weight))

        self.ptr = nn.Linear(self.combined_size + embed_size + self.hidden_size, 1)

        if tied_embedding is not None and embed_size != self.combined_size:
            # use pre_out layer if combined size is different from embedding size
            self.out_embed_size = embed_size
        if self.out_embed_size:  # use pre_out layer
            self.pre_out = nn.Linear(self.combined_size, self.out_embed_size, bias=False)
            size_before_output = self.out_embed_size
        else:  # don't use pre_out layer
            size_before_output = self.combined_size

        self.out = nn.Linear(size_before_output, vocab_size, bias=False)
        self.out.weight = tied_embedding.weight

    def forward(
        self,
        embedded,
        rnn_state,
        encoder_hiddens=None,
        decoder_hiddens=None,
        coverage_vector=None, *,
        input_mask=None,
        input_node_mask=None,
        encoder_word_idx=None,
        ext_vocab_size=None,
        log_prob=True,
        prev_enc_context=None,
    ):
        """
            :param embedded: (batch size, embed size)

            :param rnn_state: LSTM: ((1, batch size, decoder hidden size), (1, batch size, decoder hidden size)),
                              GRU:(1, batch size, decoder hidden size)

            :param encoder_hiddens: (src seq len, batch size, hidden size), for attention mechanism

            :param decoder_hiddens: (past dec steps, batch size, hidden size), for attention mechanism

            :param encoder_word_idx: (batch size, src seq len), for pointer network

            :param ext_vocab_size: the dynamic word_vocab size, determined by the max num of OOV words contained
                                   in any src seq in this batch, for pointer network

            :param log_prob: return log probability instead of probability

            :return: 4-tuple:
                     1. word prob or log word prob, (batch size, dynamic word_vocab size);
                     2. rnn_state, RNN hidden (and/or ceil) state after this step, (1, batch size, decoder hidden size);
                     3. attention weights over encoder states, (batch size, src seq len);
                     4. prob of copying by pointing as opposed to generating, (batch size, 1)

            Perform single-step decoding.
        """
        batch_size = embedded.size(0)
        combined = send_to_device(torch.zeros(batch_size, self.combined_size), self.device)
        embedded = dropout(embedded, self.in_drop, training=self.training)

        if prev_enc_context is None:
            prev_enc_context = send_to_device(torch.zeros(batch_size, encoder_hiddens.size(-1)), self.device)
        dec_input_emb = self.fc_dec_input(torch.cat([embedded, prev_enc_context], -1))

        output, rnn_state = self.model(dec_input_emb.unsqueeze(0), rnn_state)
        output = dropout(output, self.rnn_drop, training=self.training)
        rnn_state = tuple([dropout(x, self.rnn_drop, training=self.training) for x in rnn_state])
        hidden = torch.cat(rnn_state, -1).squeeze(0)
        combined[:, :self.hidden_size] = output.squeeze(0)
        offset = self.hidden_size
        enc_attn, prob_ptr = None, None  # for visualization

        # energy and attention: (num encoder states, batch size, 1)
        num_enc_steps = encoder_hiddens.size(0)
        enc_total_size = encoder_hiddens.size(2)
        if self.enc_attn_cover and coverage_vector is not None:
            # (batch size, num encoder states, encoder hidden size)
            addition_vec = coverage_vector.unsqueeze(-1) * self.cover_weight
        else:
            addition_vec = None
        enc_energy = self.enc_attn_fn(
            hidden,
            encoder_hiddens.transpose(0, 1).contiguous(),
            attn_mask=None,
            addition_vec=addition_vec,
        )
        enc_attn = input_mask * enc_energy - (1 - input_mask) * INF

        enc_attn = F.softmax(enc_attn, dim=-1)
        # context: (batch size, encoder hidden size, 1)
        enc_context = torch.bmm(encoder_hiddens.permute(1, 2, 0), enc_attn.unsqueeze(-1)).squeeze(2)
        combined[:, offset:offset+enc_total_size] = enc_context
        offset += enc_total_size

        if self.out_embed_size:
            out_embed = torch.tanh(self.pre_out(combined))
        else:
            out_embed = combined
        out_embed = dropout(out_embed, self.out_drop, training=self.training)

        logits = self.out(out_embed)  # (batch size, word_vocab size)

        output = send_to_device(torch.zeros(batch_size, ext_vocab_size), self.device)
        # distribute probabilities between generator and pointer
        pgen_cat = [embedded, hidden, enc_context]

        prob_ptr = torch.sigmoid(self.ptr(torch.cat(pgen_cat, -1)))
        prob_gen = 1 - prob_ptr
        gen_output = F.softmax(logits, dim=1)
        output[:, :self.vocab_size] = prob_gen * gen_output

        enc_attn2 = input_node_mask * enc_energy - (1 - input_node_mask) * INF
        ptr_output = F.softmax(enc_attn2, dim=-1)[:, :encoder_word_idx.shape[1]]
        output.scatter_add_(1, encoder_word_idx, prob_ptr * ptr_output)
        if log_prob:
            output = torch.log(output + EPS)

        return output, rnn_state, enc_attn, prob_ptr, enc_context
