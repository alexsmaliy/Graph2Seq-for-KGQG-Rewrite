MODEL = "graph2seq"

TRAINING_DATASET = "./data/mhqg-pq/train.json"
DEVELOP_DATASET = "./data/mhqg-pq/dev.json"
TESTING_DATASET = "./data/mhqg-pq/test.json"

VOCAB_MODEL_PATH = "./data/vocab_model"
PRETRAINED_WORD_EMBEDDINGS = "./data/glove.840B.300d.txt"
WORD_EMBED_DIM = 300
MAX_VOCAB_SIZE = 20000
MIN_WORD_FREQ = 3

ANS_MATCH_EMB_DIM = 24 # varies from model to model?

COVER_LOSS = 0.0
DEC_HIDDEN_SIZE = 300
ENC_RNN_DROPOUT = 0.3
EPS_LABEL_SMOOTHING = 0.2
HIDDEN_SIZE = 300
GRAPH_HIDDEN_SIZE = HIDDEN_SIZE + ANS_MATCH_EMB_DIM
GRAPH_HOPS = 4
MAX_DEC_STEPS = 26 # or 37? varies from model to model?
RNN_SIZE = 300
WORD_DROPOUT = 0.4

USE_CUDA = True
CUDA_DEVICE_ID = 1

USE_LOGGING = True
LOG_DIR = "log"

RANDOM_SEED = 42
