import config
import numpy as np
import random
import torch


def set_random_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_random_seed_everywhere(config.RANDOM_SEED)
    print("moo")


if __name__ == "__main__":
    main()