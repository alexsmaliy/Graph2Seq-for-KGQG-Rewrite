MODEL = "graph2seq"

TRAINING_DATASET = "./data/mhqg-pq/train.json"
DEVELOP_DATASET = "./data/mhqg-pq/dev.json"
TESTING_DATASET = "./data/mhqg-pq/test.json"

VOCAB_MODEL_PATH = "./data/vocab_model"
PRETRAINED_WORD_EMBEDDINGS = "./data/glove.840B.300d.txt"
WORD_EMBEDDINGS_DIM = 300
MAX_VOCAB_SIZE = 20000
MIN_WORD_FREQ = 3

ANS_MATCH_EMB_DIM = 24 # varies from model to model?
HIDDEN_SIZE = 300
GRAPH_HIDDEN_SIZE = HIDDEN_SIZE + ANS_MATCH_EMB_DIM
DEC_HIDDEN_SIZE = 300

USE_CUDA = True
CUDA_DEVICE_ID = 1

USE_LOGGING = True
LOG_DIR = "log"

RANDOM_SEED = 42
