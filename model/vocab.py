from collections import Counter
import os
import pickle
from typing import TypedDict

import config
from utils import Dataset, Logger

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

class VocabModel(object):
    def __init__(self, dataset: Dataset, logger: Logger):
        logger.log("Building vocab model!")
        counts = _count_stats(dataset)
        logger.log(f"Number of words: {len(counts['word_count'])}")
        logger.log(f"Number of node IDs: {len(counts['node_id_count'])}")
        logger.log(f"Number of node types: {len(counts['node_type_count'])}")
        logger.log(f"Number of edge types: {len(counts['edge_type_count'])}")


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
