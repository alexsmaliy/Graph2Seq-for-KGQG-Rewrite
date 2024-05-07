import random

import torch
from torch import nn

import config
from .decoder import DecoderRNN
from .encoder import EncoderRNN
from .graph_encoder import GraphNN
from .vocab import Vocabulary
from utils import create_mask, dropout, EPS, Logger, send_to_device, PAD_TOKEN, UNK_TOKEN

class Graph2SeqOutput(object):
    def __init__(self,
        encoder_outputs, encoder_state, decoded_tokens, loss=0, loss_value=0, enc_attn_weights=None, ptr_probs=None
    ):
        self.encoder_outputs = encoder_outputs
        self.encoder_state = encoder_state
        self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
        self.loss = loss  # scalar
        self.loss_value = loss_value  # float value, excluding coverage loss
        self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
        self.ptr_probs = ptr_probs  # (out seq len, batch size)

class Graph2SeqModule(nn.Module):
    def __init__(self, word_embedding: nn.Embedding, word_vocab: Vocabulary, logger: Logger, device: torch.device):
        super(Graph2SeqModule, self).__init__()
        self.name = "Graph2Seq"
        self.device = device
        self.logger = logger
        self.word_embed = word_embedding
        self.word_vocab = word_vocab
        self.vocab_size = word_vocab.vocab_size

        self.logger.log("Setting requires_grad = False on word embedding params.")
        for param in self.word_embed.parameters():
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
            tied_embedding=self.word_embed,
            device=self.device,
            logger=self.logger,
        )

    def filter_out_of_vocab(self, t, ext_vocab_size):
        """replace any OOV index in `tensor` with UNK token"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = t.clone()
            result[t >= self.vocab_size] = self.word_vocab.unk_ind
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

    def forward(
        self, ex, target_tensor=None, criterion=None, criterion_reduction=True, criterion_nll_only=False, rl_loss=False,
        *, forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False, saved_out: Graph2SeqOutput=None,
        visualize: bool=None, include_cover_loss=False) -> Graph2SeqOutput:
        """
            :param ex:
            :param target_tensor: tensor of word indices, (batch size x tgt seq len)
            :param input_lengths: see explanation in `EncoderRNN`
            :param criterion: the loss function; if set, loss will be returned
            :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
            :param partial_forcing: see explanation in `Params` (training only)
            :param ext_vocab_size: see explanation in `DecoderRNN`
            :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                           of greedily selecting the token of the highest probability at each step
            :param saved_out: the output of this function in a previous run; if set, the encoding step will
                              be skipped and we reuse the encoder states saved in this object
            :param visualize: whether to return data for attention and pointer visualization; if None,
                              return if no `criterion` is provided
            :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

            Run the graph2seq model for training or testing.
        """
        input_graphs = ex['in_graphs']
        batch_size, max_num_nodes = input_graphs['node_name_words'].shape[:2]
        max_num_edges = input_graphs['edge_type_words'].shape[1]
        num_nodes = input_graphs['num_nodes']
        num_edges = input_graphs['num_edges']

        max_num_graph_elements = input_graphs['max_num_graph_nodes']
        num_virtual_nodes = num_nodes + num_edges
        input_mask = create_mask(num_virtual_nodes, max_num_graph_elements, self.device)
        input_node_mask = create_mask(num_nodes, max_num_graph_elements, self.device)

        log_prob = False
        visualize = False

        if target_tensor is None:
            target_length = config.MAX_DEC_STEPS
            target_mask = None
        else:
            target_tensor = target_tensor.transpose(1, 0)
            target_length = target_tensor.size(0)
            target_mask = create_mask(ex['target_lens'], target_length, self.device)

        if forcing_ratio == 1:
            # if fully teacher-forced, it may be possible to eliminate the for-loop over decoder steps
            # for generality, this optimization is not investigated
            use_teacher_forcing = True
        elif forcing_ratio > 0:
            if partial_forcing:
                use_teacher_forcing = None  # decide later individually in each step
            else:
                use_teacher_forcing = random.random() < forcing_ratio
        else:
            use_teacher_forcing = False

        if saved_out:  # reuse encoder states of a previous run
            encoder_outputs = saved_out.encoder_outputs
            encoder_state = saved_out.encoder_state
        else:
            encoder_outputs, encoder_state = self.run(
                batch_size, max_num_nodes, max_num_edges, num_nodes, num_edges, max_num_graph_elements, input_mask,
                input_graphs, target_tensor, criterion, criterion_reduction, criterion_nll_only, rl_loss,
                forcing_ratio=forcing_ratio, partial_forcing=partial_forcing, ext_vocab_size=ext_vocab_size,
                sample=sample, saved_out=saved_out, visualize=visualize, include_cover_loss=include_cover_loss
            )

        r = Graph2SeqOutput(
            encoder_outputs, encoder_state,
            torch.zeros(target_length, batch_size, dtype=torch.long),
        )

        if visualize:
            r.enc_attn_weights = torch.zeros(target_length, batch_size, max_num_graph_elements)
            r.ptr_probs = torch.zeros(target_length, batch_size)

        decoder_state = tuple([self.enc_dec_adapter[i](x) for i, x in enumerate(encoder_state)])
        decoder_hiddens = []
        enc_attn_weights = []
        enc_context = None
        dec_prob_ptr_tensor = []

        decoder_input = send_to_device(torch.tensor([self.word_vocab.sos_ind] * batch_size), self.device)

        # loop to generate tokens
        for di in range(target_length):
            decoder_embedded = self.word_embed(self.filter_out_of_vocab(decoder_input, ext_vocab_size))
            decoder_embedded = dropout(decoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
            if enc_attn_weights:
                coverage_vector = self.get_coverage_vector(enc_attn_weights)
            else:
                coverage_vector = None
            decoder_output, decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = self.decoder(
                decoder_embedded,
                decoder_state,
                encoder_outputs,
                torch.cat(decoder_hiddens) if decoder_hiddens else None,
                coverage_vector,
                input_mask=input_mask,
                input_node_mask=input_node_mask,
                encoder_word_idx=input_graphs['g_oov_idx'],
                ext_vocab_size=ext_vocab_size,
                log_prob=log_prob,
                prev_enc_context=enc_context,
            )

            dec_prob_ptr_tensor.append(dec_prob_ptr)

            # save the decoded tokens
            if not sample:
                _, top_idx = decoder_output.data.topk(1)  # top_idx shape: (batch size, k=1)
            else:
                prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
                top_idx = torch.multinomial(prob_distribution, 1)

            top_idx = top_idx.squeeze(1).detach()  # detach from history as input
            r.decoded_tokens[di] = top_idx

            # decide the next input
            if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
                decoder_input = target_tensor[di]  # teacher forcing
            else:
                decoder_input = top_idx

            # compute loss
            if criterion:
                if target_tensor is None:
                    gold_standard = top_idx
                else:
                    gold_standard = target_tensor[di] if not rl_loss else decoder_input
                if not log_prob:
                    decoder_output = torch.log(decoder_output + EPS)

                eps = config.EPS_LABEL_SMOOTHING
                n_class = decoder_output.size(1)
                one_hot = send_to_device(
                    torch.zeros_like(decoder_output).scatter(1, gold_standard.view(-1, 1), 1),
                    self.device,
                )
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
                non_pad_mask = gold_standard.ne(self.word_vocab.pad_ind).float()
                nll_loss = -(one_hot * decoder_output).sum(dim=1)
                nll_loss = nll_loss * non_pad_mask

                if criterion_reduction:
                    nll_loss = nll_loss.sum() / torch.sum(non_pad_mask)

                r.loss += nll_loss
                r.loss_value += nll_loss

            # update attention history and compute coverage loss
            cover_loss = config.COVER_LOSS
            if criterion and cover_loss > 0:
                if not criterion_nll_only and coverage_vector is not None and criterion and cover_loss > 0:
                    if criterion_reduction:
                        coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * cover_loss
                        r.loss += coverage_loss
                        if include_cover_loss:
                            r.loss_value += coverage_loss.item()
                    else:
                        coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn), dim=-1) * self.cover_loss
                        r.loss += coverage_loss
                        if include_cover_loss:
                            r.loss_value += coverage_loss
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
            if visualize:
                r.enc_attn_weights[di] = dec_enc_attn.data
                r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data

        return r

    def run(self, batch_size, max_num_nodes, max_num_edges, num_nodes, num_edges, max_num_graph_elements, input_mask,
        input_graphs, target_tensor=None, criterion=None, criterion_reduction=True, criterion_nll_only=False, rl_loss=False,
        *, forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False, saved_out: Graph2SeqOutput=None,
        visualize: bool=None, include_cover_loss: bool=False
    ):
        node_name_word_emb = self.word_embed(self.filter_out_of_vocab(input_graphs['node_name_words'], ext_vocab_size))
        node_name_word_emb = dropout(node_name_word_emb, config.WORD_DROPOUT, shared_axes=[-2], training=self.training)
        node_name_lens = input_graphs['node_name_lens'].view(-1)
        input_node_name_cat = [node_name_word_emb]

        input_node_name_cat = torch.cat(input_node_name_cat, -1)
        input_node_name_cat = input_node_name_cat.view(-1, input_node_name_cat.size(-2), input_node_name_cat.size(-1))
        node_name_word_vec = self.node_name_word_encoder(input_node_name_cat, node_name_lens)[1]
        node_name_word_vec = node_name_word_vec[0]

        node_name_word_vec = node_name_word_vec.squeeze(0).view(input_graphs['node_name_words'].shape[:2] + (-1,))
        input_node_cat = [node_name_word_vec]

        edge_type_word_emb = self.word_embed(self.filter_out_of_vocab(input_graphs['edge_type_words'], ext_vocab_size))
        edge_type_word_emb = dropout(edge_type_word_emb, config.WORD_DROPOUT, shared_axes=[-2], training=self.training)
        edge_type_word_emb = edge_type_word_emb.view(-1, edge_type_word_emb.size(-2), edge_type_word_emb.size(-1))
        edge_type_lens = input_graphs['edge_type_lens'].view(-1)
        edge_type_word_emb = self.edge_type_word_encoder(edge_type_word_emb, edge_type_lens)[1]
        edge_type_word_emb = edge_type_word_emb[0]
        edge_type_word_emb = edge_type_word_emb.squeeze(0).view(batch_size, max_num_edges, -1)
        input_edge_cat = [edge_type_word_emb]

        node_ans_match_emb = self.ans_match_embed(input_graphs['node_ans_match'])
        input_node_cat.append(node_ans_match_emb)
        input_edge_cat.append(
            send_to_device(
                torch.zeros(
                    edge_type_word_emb.shape[:2] + (node_ans_match_emb.size(-1),),
                ),
                self.device,
            )
        )
        input_node_vec = torch.cat(input_node_cat, -1)
        input_edge_vec = torch.cat(input_edge_cat, -1)

        init_node_vec = self.gather(input_node_vec, input_edge_vec, num_nodes, num_edges, max_num_graph_elements)
        init_edge_vec = None

        node_embedding, graph_embedding = self.graph_encoder(
            init_node_vec,
            init_edge_vec,
            (input_graphs['node2edge'], input_graphs['edge2node']),
            node_mask=input_mask,
            ans_state=None,
        )
        encoder_outputs = node_embedding
        encoder_state = (graph_embedding, graph_embedding)

        return encoder_outputs, encoder_state
