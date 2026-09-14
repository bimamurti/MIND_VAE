# model
OB_RADIUS = 10# observe radius, neighborhood radius
MAX_NEIGHBORS = 10
OB_HORIZON = 24      # 8 number of observation frames
PRED_HORIZON = 36   # 12 number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
RNN_HIDDEN_DIM = 256

# training
LEARNING_RATE = 2e-4 
BATCH_SIZE = 128
EPOCHS = 2       # total number of epochs for training
EPOCH_BATCHES = None # 20 number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 1   # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 100   # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate

# evaluation
WORLD_SCALE = 1
