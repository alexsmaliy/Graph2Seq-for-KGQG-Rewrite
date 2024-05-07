import numpy as np
import random
import torch

import config
from model.manager import ModelManager

def set_random_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_random_seed_everywhere(config.RANDOM_SEED)
    model_manager = ModelManager()
    model_manager.train()
    model_manager.test()

if __name__ == "__main__":
    main()
