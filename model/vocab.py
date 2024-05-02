import os
import pickle

import config
from utils import Dataset, Logger

class VocabModel(object):
    def __init__(self, dataset: Dataset, logger: Logger):
        logger.log(f"Building vocab model!")

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
