import pdb

import torch

import config
from utils import batch_decoded_index2word, create_mask, dropout, INF, send_to_device

def beam_search(batch, network, vocab):
    with torch.no_grad():
        ext_vocab_size = batch["oov_dict"].ext_vocab_size if batch["oov_dict"] else None

        hypotheses = batch_beam_search(
            network,
            batch,
            ext_vocab_size,
            config.BEAM_SIZE,
            min_out_len=config.MIN_OUT_LENGTH,
            max_out_len=config.MAX_OUT_LENGTH,
            len_in_words=False,
            block_ngram_repeat=config.BLOCK_NGRAM_REPEAT,
        )

        try:
            to_decode = [each[0].tokens[1:] for each in hypotheses] # the first token is SOS
        except:
            pdb.set_trace()

        decoded_batch = batch_decoded_index2word(to_decode, vocab, batch['oov_dict'])
        return decoded_batch

def batch_beam_search(
        network, ex, ext_vocab_size=None, beam_size=4,
        *, min_out_len=1, max_out_len=None, len_in_words=False, block_ngram_repeat=0,
):
    """
        Use beam search to generate summaries.
    """
    input_graphs = ex["in_graphs"]
    batch_size, max_num_nodes = input_graphs["node_name_words"].shape[:2]
    max_num_edges = input_graphs["edge_type_words"].shape[1]
    num_nodes = input_graphs["num_nodes"]
    num_edges = input_graphs["num_edges"]

    max_num_graph_elements = input_graphs["max_num_graph_nodes"]
    num_virtual_nodes = num_nodes + num_edges
    input_mask = create_mask(num_virtual_nodes, max_num_graph_elements, network.device)
    input_node_mask = create_mask(num_nodes, max_num_graph_elements, network.device)

    if max_out_len is None:
        max_out_len = config.MAX_DEC_STEPS - 1 # don't count EOS token

    node_name_word_emb = network.word_embed(network.filter_out_of_vocab(input_graphs["node_name_words"], ext_vocab_size))
    node_name_word_emb = dropout(
        node_name_word_emb,
        config.WORD_DROPOUT,
        shared_axes=[-2],
        training=network.training,
    )
    node_name_lens = input_graphs["node_name_lens"].view(-1)
    input_node_name_cat = [node_name_word_emb]

    input_node_name_cat = torch.cat(input_node_name_cat, -1)
    input_node_name_cat = input_node_name_cat.view(-1, input_node_name_cat.size(-2), input_node_name_cat.size(-1))
    node_name_word_vec = network.node_name_word_encoder(input_node_name_cat, node_name_lens)[1][0]
    node_name_word_vec = node_name_word_vec.squeeze(0).view(input_graphs["node_name_words"].shape[:2] + (-1,))
    input_node_cat = [node_name_word_vec]

    edge_type_word_emb = network.word_embed(network.filter_out_of_vocab(input_graphs["edge_type_words"], ext_vocab_size))
    edge_type_word_emb = dropout(edge_type_word_emb, config.WORD_DROPOUT, shared_axes=[-2], training=network.training)
    edge_type_word_emb = edge_type_word_emb.view(-1, edge_type_word_emb.size(-2), edge_type_word_emb.size(-1))
    edge_type_lens = input_graphs["edge_type_lens"].view(-1)
    edge_type_word_emb = network.edge_type_word_encoder(edge_type_word_emb, edge_type_lens)[1][0]
    # (batch_size, max_num_edges, hidden_size)
    edge_type_word_emb = edge_type_word_emb.squeeze(0).view(batch_size, max_num_edges, -1)
    input_edge_cat = [edge_type_word_emb]

    node_ans_match_emb = network.ans_match_embed(input_graphs["node_ans_match"])
    input_node_cat.append(node_ans_match_emb)
    input_edge_cat.append(
        send_to_device(
            torch.zeros(edge_type_word_emb.shape[:2] + (node_ans_match_emb.size(-1),)),
            network.device,
        )
    )

    input_node_vec = torch.cat(input_node_cat, -1)
    input_edge_vec = torch.cat(input_edge_cat, -1)
    init_node_vec = network.gather(input_node_vec, input_edge_vec, num_nodes, num_edges, max_num_graph_elements)
    init_edge_vec = None

    node_embedding, graph_embedding = network.graph_encoder(
        init_node_vec,
        init_edge_vec,
        (input_graphs["node2edge"], input_graphs["edge2node"]),
    )
    encoder_outputs = node_embedding
    encoder_state = (graph_embedding, graph_embedding)
    decoder_state = tuple([network.enc_dec_adapter[i](x) for i, x in enumerate(encoder_state)])

    batch_results = []
    for batch_idx in range(batch_size):
        res = run_batch(
            batch_idx=batch_idx,
            beam_size=beam_size,
            decoder_state=decoder_state,
            encoder_outputs=encoder_outputs,
            ext_vocab_size=ext_vocab_size,
            input_graphs=input_graphs,
            input_mask=input_mask,
            input_node_mask=input_node_mask,
            min_out_len=min_out_len,
            max_out_len=max_out_len,
            network=network,
        )
        batch_results.append(res)
    return batch_results

def run_batch(*,
    batch_idx, beam_size, decoder_state, encoder_outputs, ext_vocab_size, input_graphs, input_mask, input_node_mask,
    min_out_len, max_out_len, network,
):
    single_encoder_outputs = encoder_outputs[:, batch_idx: batch_idx + 1].expand(-1, beam_size, -1).contiguous()
    single_input_tensor = input_graphs["g_oov_idx"][batch_idx: batch_idx + 1].expand(beam_size, -1).contiguous()
    single_input_mask = input_mask[batch_idx: batch_idx + 1].expand(beam_size, -1).contiguous()
    single_input_node_mask = input_node_mask[batch_idx: batch_idx + 1].expand(beam_size, -1).contiguous()
    single_decoder_state = tuple([each[:, batch_idx: batch_idx + 1] for each in decoder_state])

    # decode
    hypos = [
        Hypothesis([network.word_vocab.sos_ind], [], single_decoder_state, [], [], 1)
    ]
    results, backup_results = [], []
    enc_context = None
    step = 0
    while len(hypos) > 0 and step <= max_out_len:
        step, hypos, results, backup_results = run_hypos(
            hypos=hypos,
            step=step,
            results=results,
            backup_results=backup_results,
            beam_size=beam_size,
            ext_vocab_size=ext_vocab_size,
            min_out_len=min_out_len,
            max_out_len=max_out_len,
            network=network,
            single_encoder_outputs=single_encoder_outputs,
            single_input_mask=single_input_mask,
            single_input_node_mask=single_input_node_mask,
            single_input_tensor=single_input_tensor,
            enc_context=enc_context,
        )
    if not results:
        results = backup_results
    return sorted(results, key=lambda h: -h.avg_log_prob)[:beam_size]

def run_hypos(*,
    hypos: list["Hypothesis"], step, results, backup_results,
    beam_size, ext_vocab_size, min_out_len, max_out_len, network,
    single_encoder_outputs, single_input_mask, single_input_node_mask, single_input_tensor, enc_context,
):
    n_hypos = len(hypos)
    if n_hypos < beam_size:
        hypos.extend(hypos[-1] for _ in range(beam_size - n_hypos))
    decoder_input = send_to_device(torch.tensor([h.tokens[-1] for h in hypos]), network.device)
    single_decoder_state = (
        torch.cat([h.dec_state[0] for h in hypos], 1),
        torch.cat([h.dec_state[1] for h in hypos], 1),
    )
    decoder_hiddens = None
    enc_attn_weights = []
    coverage_vector = None
    decoder_embedded = network.word_embed(network.filter_out_of_vocab(decoder_input, ext_vocab_size))
    decoder_output, single_decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = network.decoder(
        decoder_embedded,
        single_decoder_state,
        single_encoder_outputs,
        decoder_hiddens,
        coverage_vector,
        input_mask=single_input_mask,
        input_node_mask=single_input_node_mask,
        encoder_word_idx=single_input_tensor,
        ext_vocab_size=ext_vocab_size,
        prev_enc_context=enc_context,
    )

    top_v, top_i = decoder_output.data.topk(beam_size)  # shape of both: (beam size, beam size)
    new_hypos = []
    for in_idx in range(n_hypos):
        for out_idx in range(beam_size):
            new_tok = top_i[in_idx][out_idx].item()
            new_prob = top_v[in_idx][out_idx].item()
            non_word = new_tok == network.word_vocab.eos_ind
            tmp_decoder_state = [x[0][in_idx].unsqueeze(0).unsqueeze(0) for x in single_decoder_state]
            new_hypo = hypos[in_idx].create_next(
                new_tok,
                new_prob,
                tmp_decoder_state,
                False,
                dec_enc_attn[in_idx].unsqueeze(0).unsqueeze(0)
                if dec_enc_attn is not None else None,
                non_word,
            )
            new_hypos.append(new_hypo)
    block_ngram_repeats(new_hypos, config.BLOCK_NGRAM_REPEAT)

    new_hypos = sorted(new_hypos, key=lambda h: -h.avg_log_prob)[:beam_size]
    hypos = []
    new_complete_results, new_incomplete_results = [], []

    for nh in new_hypos:
        length = len(nh)
        if nh.tokens[-1] == network.word_vocab.eos_ind:
            if len(new_complete_results) < beam_size and min_out_len <= length <= max_out_len:
                new_complete_results.append(nh)
        elif len(hypos) < beam_size and length < max_out_len:
            hypos.append(nh)
        elif length == max_out_len and len(new_incomplete_results) < beam_size:
            new_incomplete_results.append(nh)

    if new_complete_results:
        results.extend(new_complete_results)
    elif new_incomplete_results:
        backup_results.extend(new_incomplete_results)
    step += 1

    return step, hypos, results, backup_results

def block_ngram_repeats(hypos, block_ngram_repeat, exclusion_tokens=None):
    if exclusion_tokens is None:
        exclusion_tokens = set()
    cur_len = len(hypos[0].tokens)
    if block_ngram_repeat > 0 and cur_len > 1:
        for path_idx in range(len(hypos)):
            # skip SOS
            hyp = hypos[path_idx].tokens[1:]
            ngrams = set()
            fail = False
            gram = []
            for i in range(cur_len - 1):
                # Last n tokens, n = block_ngram_repeat
                gram = (gram + [hyp[i]])[-block_ngram_repeat:]
                # skip the blocking if any token in gram is excluded
                if set(gram) & exclusion_tokens:
                    continue
                if tuple(gram) in ngrams:
                    fail = True
                ngrams.add(tuple(gram))
            if fail:
                hypos[path_idx].log_probs[-1] = -INF

class Hypothesis(object):
    def __init__(self, tokens, log_probs, dec_state, dec_hiddens, enc_attn_weights, num_non_words):
        self.tokens: list[int] = tokens
        self.log_probs: list[float] = log_probs
        self.dec_state = dec_state  # shape: (1, 1, hidden_size)
        self.dec_hiddens = dec_hiddens  # list of dec_hidden_state
        self.enc_attn_weights = enc_attn_weights  # list of shape: (1, 1, src_len)
        self.num_non_words = num_non_words

    def __repr__(self):
        return repr(self.tokens)

    def __len__(self):
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(self, token, log_prob, dec_state, add_dec_states, enc_attn, non_word):
        dec_hidden_state = dec_state[0]
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            dec_state=dec_state,
            dec_hiddens=self.dec_hiddens + [dec_hidden_state] if add_dec_states else self.dec_hiddens,
            enc_attn_weights=self.enc_attn_weights + [enc_attn] if enc_attn is not None else self.enc_attn_weights,
            num_non_words=self.num_non_words + 1 if non_word else self.num_non_words,
        )
