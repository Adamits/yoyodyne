"""Default values for flags and modules."""

import numpy

# All elements should be styled as CONSTANTS.

# Scalar Constants
EPSILON = 1e-7
LOG_EPSILON = numpy.log(EPSILON)
NEG_LOG_EPSILON = -numpy.log(EPSILON)
INF = numpy.inf
NEG_INF = -numpy.inf

# Data configuration arguments.
SOURCE_COL = 1
TARGET_COL = 2
FEATURES_COL = 0
SOURCE_SEP = ""
TARGET_SEP = ""
FEATURES_SEP = ";"
TIED_VOCABULARY = True

# Architecture arguments.
ARCH = "attentive_lstm"
SOURCE_ATTENTION_HEADS = 4
FEATURES_ATTENTION_HEADS = 1
BIDIRECTIONAL = True
DECODER_LAYERS = 1
EMBEDDING_SIZE = 128
ENCODER_LAYERS = 1
HARD_ATTENTION_CONTEXT = 0
HARD_ATTENTION_MONOTONIC = False
HIDDEN_SIZE = 512
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
EVAL_METRICS = set()

# Training arguments.
BATCH_SIZE = 32
BETA1 = 0.9
BETA2 = 0.999
DROPOUT = 0.2
LABEL_SMOOTHING = 0.0
LEARNING_RATE = 0.001
ORACLE_EM_EPOCHS = 5
ORACLE_FACTOR = 1
OPTIMIZER = "adam"
NUM_CHECKPOINTS = 1
CHECKPOINT_METRIC = "accuracy"
PATIENCE_METRIC = "loss"
LOG_WANDB = False
TEACHER_FORCING = True

# LR Scheduler arguments.
WARMUP_STEPS = 0
START_FACTOR = 1 / 3
END_FACTOR = 1.0
TOTAL_DECAY_STEPS = 5
MIN_LR = 0.0
REDUCEONPLATEAU_FACTOR = 0.1
REDUCEONPLATEAU_METRIC = "loss"
REDUCEONPLATEAU_PATIENCE = 10

# Decoding arguments.
BEAM_WIDTH = 1
