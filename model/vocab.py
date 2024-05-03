from collections import Counter
from collections.abc import Iterable
import os
import pickle
from typing import TypedDict

import config
from utils import Dataset, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, UNK_TOKEN, Logger, map_to_index

class VocabStats(TypedDict):
    word_count: Counter
    node_id_count: Counter
    node_type_count: Counter
    edge_type_count: Counter

def _count_stats(dataset: Dataset) -> VocabStats:
    word_count = Counter()
    node_id_count = Counter()
    node_type_count = Counter()
    edge_type_count = Counter()
    for tup in dataset:
        graph_seq, out_seq_seq, answer_seqs = tup[0], tup[1], tup[2]

        g = graph_seq.graph
        nnw = g["g_node_name_words"]
        ntw = iter(g["g_node_type_words"])
        for i in range(len(nnw)):
            word_count.update(nnw[i])
            word_count.update(next(ntw, []))

        for lst in g["g_edge_type_words"]:
            word_count.update(lst)

        gni = g["g_node_ids"].keys()
        node_id_count.update(gni)

        nti = g["g_node_type_ids"]
        node_type_count.update(nti)

        eti = g["g_edge_type_ids"]
        edge_type_count.update(eti)

        word_count.update(out_seq_seq.words)

        if answer_seqs is not None:
            for seq in answer_seqs:
                word_count.update(seq.words)

    return {
        "word_count": word_count,
        "node_id_count": node_id_count,
        "node_type_count": node_type_count,
        "edge_type_count": edge_type_count,
    }

class Vocabulary(object):
    def __init__(self, logger: Logger):
        self.pad_ind = 0
        self.sos_ind = 1
        self.eos_ind = 2
        self.unk_ind = 3
        self.reserved = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.index2word = self.reserved.copy()
        self.word2index: dict[str, int] = map_to_index(self.reserved)
        self.word2count: Counter[str] = Counter()
        self.embeddings = None
        self.logger = logger

    def build(self, count: Counter):
        vocab_size = config.MAX_VOCAB_SIZE
        min_word_freq = config.MIN_WORD_FREQ
        self.word2count = count
        self._add_words(count.keys())
        self._prune(vocab_size, min_word_freq)

    def _add_words(self, words: Iterable[str]):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        assert len(self.word2index.keys()) == len(self.index2word), \
            "While building vocabulary: word2index and index2word have unequal number of entries!"

    def _prune(self, max_vocab_size: int, min_word_freq: int):
        """keep max N most frequent ones, then remove all words below min_word_freq"""
        word_counts = self.word2count.items()
        word_counts = sorted(word_counts, key=lambda tup: tup[1], reverse=True)[:max_vocab_size]
        if min_word_freq > 1:
            word_counts = [(word, count) for (word, count) in word_counts if count >= min_word_freq]
        words, counts = zip(*word_counts)

        index2word = self.reserved.copy()
        index2word.extend(w for w in words if w not in set(self.reserved))

        word2index = map_to_index(index2word)

        word2count = Counter()
        word2count.update(dict(word_counts))

        self.index2word = index2word
        self.word2index = word2index
        self.word2count = word2count

class VocabModel(object):
    def __init__(self, dataset: Dataset, logger: Logger):
        logger.log("Building vocab model!")
        counts = _count_stats(dataset)
        logger.log(f"Number of words: {len(counts['word_count'])}")
        logger.log(f"Number of node IDs: {len(counts['node_id_count'])}")
        logger.log(f"Number of node types: {len(counts['node_type_count'])}")
        logger.log(f"Number of edge types: {len(counts['edge_type_count'])}")

        word_vocab = Vocabulary(logger)
        word_vocab.build(counts["word_count"])

        node_vocab = Vocabulary(logger)

        node_type_vocab = Vocabulary(logger)

        edge_type_vocab = Vocabulary(logger)

        self.word_vocab = word_vocab
        self.node_vocab = node_vocab
        self.node_type_vocab = node_type_vocab
        self.edge_type_vocab = edge_type_vocab

def load_or_init(dataset: Dataset, logger: Logger):
    fpath = config.VOCAB_MODEL_PATH
    if os.path.exists(fpath) and False: # TODO remove
        logger.log(f"Loading vocab model from: {fpath}")
        with open(fpath, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = VocabModel(dataset, logger)
        logger.log(f"Caching built vocab model to: {fpath}")
        with open(fpath, "wb") as f:
            pickle.dump(vocab, f)
    return vocab
