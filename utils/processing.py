import numpy as np
from scipy.sparse import *
import torch

import config
from .loading import Dataset
from .tensors import pad_2d_vals_no_size, pad_3d_vals_no_size
from .vocab import Vocabulary
# from utils import Dataset, pad_2d_vals_no_size, pad_3d_vals_no_size

def find_sublist(src_list, a_list):
    indices = []
    for i in range(len(src_list)):
        if src_list[i: i + len(a_list)] == a_list:
            start_idx = i
            end_idx = i + len(a_list)
            indices.append((start_idx, end_idx))
    return indices

def seq2ext_vocab_id(idx_in_batch, seq, word_vocab, oov_dict):
    matched_pos = {}
    for key in oov_dict.word2index:
        if key[0] == idx_in_batch:
            indices = find_sublist(seq, list(key[1]))
            for pos in indices:
                matched_pos[pos] = key

    matched_pos = sorted(matched_pos.items(), key=lambda d: d[0][0])

    seq_idx = []
    i = 0
    while i < len(seq):
        if len(matched_pos) == 0 or i < matched_pos[0][0][0]:
            seq_idx.append(word_vocab.get_index(seq[i]))
            i += 1
        else:
            pos, key = matched_pos.pop(0)
            seq_idx.append(oov_dict.word2index.get(key))
            i += len(key[1])
    return seq_idx

def vectorize_batch_graph(graphs, word_vocab: Vocabulary, oov_dict, ext_vocab=False):
    max_num_graph_nodes = max([g.get('num_virtual_nodes', len(g['g_node_name_words'])) for g in graphs])
    max_num_graph_edges = max([g.get('num_virtual_edges', len(g['g_edge_type_words'])) for g in graphs])

    batch_num_nodes = []
    batch_num_edges = []
    batch_node_name_words = []
    batch_node_name_lens = []
    batch_node_type_words = []
    batch_node_type_lens = []
    batch_node_ans_match = []
    batch_edge_type_words = []
    batch_edge_type_lens = []
    batch_node2edge = []
    batch_edge2node = []
    if ext_vocab:
        batch_g_oov_idx = []

    for example_id, g in enumerate(graphs):
        # nodes
        node_name_idx = []
        if ext_vocab:
            g_oov_idx = []
        for each in g['g_node_name_words']:
            if ext_vocab:
                oov_idx = oov_dict.add_word(example_id, tuple(each))
                g_oov_idx.append(oov_idx)
            tmp_node_name_idx = []
            for word in each: # seq level
                idx = word_vocab.get_index(word)
                tmp_node_name_idx.append(idx)
            node_name_idx.append(tmp_node_name_idx)
        batch_node_name_words.append(node_name_idx)
        batch_node_name_lens.append([max(len(x), 1) for x in node_name_idx])
        batch_node_ans_match.append(g['g_node_ans_match'])

        # node types
        node_type_idx = []
        for each in g['g_node_type_words']: # node level
            tmp_node_type_idx = []
            for word in each: # seq level
                idx = word_vocab.get_index(word)
                tmp_node_type_idx.append(idx)
            node_type_idx.append(tmp_node_type_idx)
        batch_node_type_words.append(node_type_idx)
        batch_node_type_lens.append([max(len(x), 1) for x in node_type_idx])

        # edge types
        edge_type_idx = []
        for each in g['g_edge_type_words']:
            tmp_edge_type_idx = []
            for word in each: # seq level
                idx = word_vocab.get_index(word)
                tmp_edge_type_idx.append(idx)
            edge_type_idx.append(tmp_edge_type_idx)
        batch_edge_type_words.append(edge_type_idx)
        batch_edge_type_lens.append([max(len(x), 1) for x in edge_type_idx])

        batch_num_nodes.append(len(node_name_idx))
        batch_num_edges.append(len(edge_type_idx))

        if ext_vocab:
            batch_g_oov_idx.append(g_oov_idx)
            assert len(g_oov_idx) == len(node_name_idx)

        # Adjacency matrix
        node2edge = lil_matrix(np.zeros((max_num_graph_edges, max_num_graph_nodes)), dtype=np.float32)
        edge2node = lil_matrix(np.zeros((max_num_graph_nodes, max_num_graph_edges)), dtype=np.float32)
        for node1, val in g['g_adj'].items():
            for node2, edge in val.items(): # node1 -> edge -> node2
                if node1 == node2: # Ignore self-loops for now
                    continue
                node2edge[edge, node1] = 1
                edge2node[node2, edge] = 1

        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)

    batch_num_nodes = np.array(batch_num_nodes, dtype=np.int32)
    batch_num_edges = np.array(batch_num_edges, dtype=np.int32)
    batch_node_name_words = pad_3d_vals_no_size(batch_node_name_words, fills=word_vocab.pad_ind)
    batch_edge_type_words = pad_3d_vals_no_size(batch_edge_type_words, fills=word_vocab.pad_ind)
    batch_node_name_lens = pad_2d_vals_no_size(batch_node_name_lens, fills=1)
    batch_node_ans_match = pad_2d_vals_no_size(batch_node_ans_match, fills=0)
    batch_edge_type_lens = pad_2d_vals_no_size(batch_edge_type_lens, fills=1)

    batch_graphs = {
        "max_num_graph_nodes": max_num_graph_nodes,
        "num_nodes": batch_num_nodes,
        "num_edges": batch_num_edges,
        "node_name_words": batch_node_name_words,
        "edge_type_words": batch_edge_type_words,
        "node_name_lens": batch_node_name_lens,
        "node_ans_match": batch_node_ans_match,
        "edge_type_lens": batch_edge_type_lens,
        "node2edge": batch_node2edge,
        "edge2node": batch_edge2node,
    }

    if ext_vocab:
        batch_g_oov_idx = pad_2d_vals_no_size(batch_g_oov_idx, fills=word_vocab.pad_ind)
        batch_graphs['g_oov_idx'] = batch_g_oov_idx

    return batch_graphs

def vectorize_input(batch, training=True, device=None):
    if not batch:
        return None

    batch_size = len(batch.out_seqs)

    in_graphs = {}
    for k, v in batch.in_graphs.items():
        if k in ['node2edge', 'edge2node', 'max_num_graph_nodes']:
            in_graphs[k] = v
        else:
            in_graphs[k] = torch.LongTensor(v).to(device) if device else torch.LongTensor(v)

    out_seqs = torch.LongTensor(batch.out_seqs)
    out_seq_lens = torch.LongTensor(batch.out_seq_lens)

    with torch.set_grad_enabled(training):
        return {
            "batch_size": batch_size,
            "in_graphs": in_graphs,
            "targets": out_seqs.to(device) if device else out_seqs,
            "target_lens": out_seq_lens.to(device) if device else out_seq_lens,
            "target_src": batch.out_seq_src,
            "oov_dict": batch.oov_dict,
        }

def _make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [
        (
            i * batch_size,
            min(size, (i+1)*batch_size),
        ) for i in range(nb_batch)
    ]

class OutOfVocabDict(object):
    def __init__(self, base_oov_idx):
        self.word2index: dict[tuple[int, str], int] = {}
        self.index2word: dict[tuple[int, int], str] = {}
        self.next_index: dict[int, int] = {}
        self.base_oov_idx = base_oov_idx
        self.ext_vocab_size = base_oov_idx

    def add_word(self, idx_in_batch, word) -> int:
        key = (idx_in_batch, word)
        index = self.word2index.get(key, None)
        if index is not None:
            return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index

class Batch(object):
    def __init__(self, instances, word_vocab: Vocabulary, ext_vocab=False):
        self.instances = instances
        self.batch_size = len(instances)
        self.oov_dict = None

        self.out_seq_src = []
        self.out_seqs = []
        self.out_seq_lens = []

        if ext_vocab:
            base_oov_idx = len(word_vocab)
            self.oov_dict = OutOfVocabDict(base_oov_idx)

        batch_graph = [each[0].graph for each in instances]
        self.in_graphs = vectorize_batch_graph(
            batch_graph, word_vocab, self.oov_dict, ext_vocab=ext_vocab,
        )

        for i, (_, seq2, seq3) in enumerate(instances):
            if ext_vocab:
                seq2_idx = seq2ext_vocab_id(i, seq2.words, word_vocab, self.oov_dict)
            else:
                seq2_idx = []
                for word in seq2.words:
                    idx = word_vocab.get_index(word)
                    seq2_idx.append(idx)
            self.out_seq_src.append(seq2.src)
            self.out_seqs.append(seq2_idx)
            self.out_seq_lens.append(len(seq2_idx))

        self.out_seqs = pad_2d_vals_no_size(self.out_seqs, fills=word_vocab.pad_ind)
        self.out_seq_lens = np.array(self.out_seq_lens, dtype=np.int32)

class DataStream(object):
    def __init__(self, all_instances: Dataset, word_vocab: Vocabulary,
        is_shuffle=False, is_loop=False, is_sort=True, batch_size=-1, ext_vocab=False,
    ):
        if batch_size == -1:
            batch_size = config.BATCH_SIZE
        if is_sort:
            all_instances = sorted(
                all_instances,
                key=lambda instance: len(instance[0].graph['g_node_name_words']),
            )
        self.num_instances = len(all_instances)

        batch_spans = _make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances: Dataset = all_instances[batch_start:batch_end]
            cur_batch = Batch(
                cur_instances, word_vocab, ext_vocab=ext_vocab,
            )
            self.batches.append(cur_batch)
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = is_shuffle
        if self.isShuffle:
            np.random.shuffle(self.index_array)
        self.isLoop = is_loop
        self.cur_pointer = 0

    def next_batch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop:
                return None
            self.cur_pointer = 0
            if self.isShuffle:
                np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def get_batch(self, i):
        if i >= self.num_batch:
            return None
        return self.batches[i]

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def reset(self):
        if self.isShuffle:
            np.random.shuffle(self.index_array)
        self.cur_pointer = 0
