# model
OB_RADIUS = 15       # observe radius, neighborhood radius
OB_HORIZON = 25      # 7 number of observation frames
PRED_HORIZON = 50   # 12 number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
RNN_HIDDEN_DIM = 256

# training
LEARNING_RATE = 1e-4 
BATCH_SIZE = 64
EPOCHS = 6      # total number of epochs for training
EPOCH_BATCHES = 10 # 20 number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 2    # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20   # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate
MAX_NEIGHBORS = 10
# evaluation
WORLD_SCALE = 1
