MODEL = "graph2seq"
PRETRAINED = None
SAVED_WEIGHTS_FILE = "params.saved"

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

BATCH_SIZE = 30
EARLY_STOP_METRIC = "Bleu_4"
FORCING_DECAY = 0.9999
FORCING_RATIO = 0.8
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
MAX_RL_RATIO = 0.99
PATIENCE = 10
RL_RATIO = 0.0
RL_RATIO_POWER = 1 # 0.7 in some config comments?
RL_START_EPOCH = 1
SAVE_PARAMS = True
VERBOSE = 1000 # print every N batches

BEAM_SIZE = 5 # for beam search
BLOCK_NGRAM_REPEAT = 0 # stop n-gram repetition during decoding
OUT_PREDICTIONS = True # log predictions

GRAD_CLIP = 10

USE_CUDA = True
CUDA_DEVICE_ID = 1

USE_LOGGING = True
LOG_DIR = "log"

RANDOM_SEED = 42
