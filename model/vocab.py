from collections import Counter
from collections.abc import Iterable
import numpy as np
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
    embedding_dtype = np.float32
    embedding_scale = 0.08
    def __init__(self, logger: Logger):
        self.pad_ind = 0 # the embedding vector at this index doesn't get updated in pytorch
        self.sos_ind = 1
        self.eos_ind = 2
        self.unk_ind = 3
        self.reserved = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.index2word = self.reserved.copy()
        self.word2index: dict[str, int] = map_to_index(self.reserved)
        self.word2count: Counter[str] = Counter()
        self.embeddings = None
        self.vocab_size = 0
        self.embed_dim = 0
        self.logger = logger

    def build(self, count: Counter):
        max_vocab_size = config.MAX_VOCAB_SIZE
        min_word_freq = config.MIN_WORD_FREQ
        self.word2count = count
        if len(count.keys()) > 0:
            self._add_words(count.keys())
            self._prune(max_vocab_size, min_word_freq)

    def _add_words(self, words: Iterable[str]):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        assert len(self.word2index.keys()) == len(self.index2word), \
            "While building vocabulary: word2index and index2word have unequal number of entries!"

    def _prune(self, max_vocab_size: int, min_word_freq: int):
        """keep max N most frequent ones, then remove all words below min_word_freq"""
        self.logger.log(f"Pruning loaded vocab for max size {max_vocab_size} and min freq {min_word_freq}: {len(self.index2word)} items before")
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
        self.vocab_size = len(self.index2word)
        self.logger.log(f"Pruning loaded vocab: {len(self.index2word)} items after")

    def init_embeddings(self, dim: int):
        emb = np.random.uniform(
            low=-self.embedding_scale,
            high=self.embedding_scale,
            size=(len(self.index2word), dim)
        )
        emb = np.array(emb, dtype=Vocabulary.embedding_dtype)
        emb[self.pad_ind, :] = np.zeros(dim)
        self.embeddings = emb

    def load_embeddings(self, fpath: str):
        self.logger.log(f"Loading GloVe embeddings from {fpath}")
        seen_indexes = set()
        n = 0
        with open(fpath, "rb") as f:
            for line in f:
                n += 1
                line = line.split()
                word = line[0].decode("utf-8")
                word_index = self.word2index.get(word.lower(), None)
                if word_index is None or word_index in seen_indexes:
                    continue
                embedding = np.array(line[1:], dtype=Vocabulary.embedding_dtype)
                if self.embeddings is None:
                    embedding_dim = len(embedding)
                    self.init_embeddings(embedding_dim)
                self.embeddings[word_index, :] = embedding
                seen_indexes.add(word_index)
        self.embed_dim = self.embeddings.shape[1]
        self.logger.log({
            "embedding_words": n,
            "embedding_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
            "matched_vocab": len(seen_indexes),
            "matched_ratio": round(len(seen_indexes) / self.vocab_size, 3),
        }, as_json=True)

class VocabModel(object):
    def __init__(self, dataset: Dataset, logger: Logger):
        logger.log("Building vocab model!")
        counts = _count_stats(dataset)

        word_vocab = Vocabulary(logger)
        logger.log(f"Building vocabulary of words: input has {len(counts['word_count'])}")
        word_vocab.build(counts["word_count"])
        if config.PRETRAINED_WORD_EMBEDDINGS:
            logger.log("Loading pretrained word embeddings.")
            word_vocab.load_embeddings(config.PRETRAINED_WORD_EMBEDDINGS)
        else:
            logger.log("Using randomized word embeddings.")
            word_vocab.init_embeddings(config.WORD_EMBED_DIM)

        node_id_vocab = Vocabulary(logger)
        logger.log(f"Building vocabulary of node IDs: input has {len(counts['node_id_count'])}")
        node_id_vocab.build(counts["node_id_count"])

        node_type_vocab = Vocabulary(logger)
        logger.log(f"Building vocabulary of node types: input has {len(counts['node_type_count'])}")
        node_type_vocab.build(counts["node_type_count"])

        edge_type_vocab = Vocabulary(logger)
        logger.log(f"Building vocabulary of edge types: input has {len(counts['edge_type_count'])}")
        edge_type_vocab.build(counts["edge_type_count"])

        self.word_vocab = word_vocab
        self.node_id_vocab = node_id_vocab
        self.node_type_vocab = node_type_vocab
        self.edge_type_vocab = edge_type_vocab

def load_or_init(dataset: Dataset, logger: Logger):
    fpath = config.VOCAB_MODEL_PATH
    if os.path.exists(fpath):
        logger.log(f"Loading vocab model from: {fpath}")
        with open(fpath, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = VocabModel(dataset, logger)
        logger.log(f"Caching built vocab model to: {fpath}")
        with open(fpath, "wb") as f:
            pickle.dump(vocab, f)
    return vocab
