MODEL = "graph2seq"

TRAINING_DATASET = "./data/mhqg-pq/train.json"
DEVELOP_DATASET = "./data/mhqg-pq/dev.json"
TESTING_DATASET = "./data/mhqg-pq/test.json"

VOCAB_MODEL_PATH = "./data/vocab_model"
PRETRAINED_WORD_EMBEDDINGS = "./data/glove.840B.300d.txt"
MAX_VOCAB_SIZE = 20000
MIN_WORD_FREQ = 3

USE_CUDA = True
CUDA_DEVICE_ID = 1

USE_LOGGING = True
LOG_DIR = "log"

RANDOM_SEED = 42
