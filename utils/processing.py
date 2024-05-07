import numpy as np
import torch

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
    def __init__(self, instances, word_vocab, node_vocab, node_type_vocab, edge_type_vocab, ext_vocab=False):
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
            batch_graph,
            word_vocab,
            node_vocab,
            node_type_vocab,
            edge_type_vocab,
            self.oov_dict,
            kg_emb=False,
            f_node_type=False,
            ext_vocab=ext_vocab,
        )

        for i, (_, seq2, seq3) in enumerate(instances):
            if ext_vocab:
                seq2_idx = seq2ext_vocab_id(i, seq2.words, word_vocab, self.oov_dict)
            else:
                seq2_idx = []
                for word in seq2.words:
                    idx = word_vocab.getIndex(word)
                    seq2_idx.append(idx)
            self.out_seq_src.append(seq2.src)
            self.out_seqs.append(seq2_idx)
            self.out_seq_lens.append(len(seq2_idx))

        self.out_seqs = padding_utils.pad_2d_vals_no_size(self.out_seqs, fills=word_vocab.PAD)
        self.out_seq_lens = np.array(self.out_seq_lens, dtype=np.int32)

class DataStream(object):
    def __init__(self,
        all_instances, word_vocab, node_vocab, node_type_vocab, edge_type_vocab,
        config=None, isShuffle=False, isLoop=False, isSort=True, batch_size=-1, ext_vocab=False, bert_tokenizer=None
    ):
        if batch_size == -1:
            batch_size = config.BATCH_SIZE
        if isSort:
            all_instances = sorted(
                all_instances,
                key=lambda instance: len(instance[0].graph['g_node_name_words']),
            )
        self.num_instances = len(all_instances)

        batch_spans = _make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start:batch_end]
            cur_batch = Batch(
                cur_instances, word_vocab, node_vocab, node_type_vocab, edge_type_vocab, ext_vocab=ext_vocab,
            )
            self.batches.append(cur_batch)
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle:
            np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0
