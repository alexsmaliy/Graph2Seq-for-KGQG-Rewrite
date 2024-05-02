import numpy as np
import random
import torch

import config

from model.handler import ModelHandler


def set_random_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_random_seed_everywhere(config.RANDOM_SEED)
    model = ModelHandler()
    # model.train()
    # model.test()


if __name__ == "__main__":
    main()