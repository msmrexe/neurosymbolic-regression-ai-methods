"""
Central configuration file for the Symbolic Regression AI project.
Contains shared paths, constants, and device settings.
"""

import torch
import os

# --- 1. Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Path Configuration ---
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Log directory
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Output directory (for models, plots, etc.)
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
HIDDEN_DATA_PATH = os.path.join(DATA_DIR, "hidden_formula_dataset.csv")
TRANSFORMER_DATA_PATH = os.path.join(DATA_DIR, "transformer_pregen")

# --- 3. EQL Default Hyperparameters ---
EQL_DEFAULT_EPOCHS = 20000
EQL_DEFAULT_LR = 1e-3

# --- 4. Transformer Default Hyperparameters ---
TRANSFORMER_DEFAULT_EPOCHS = 100
TRANSFORMER_DEFAULT_BATCH_SIZE = 128
TRANSFORMER_DEFAULT_LR = 1e-4

# --- 5. Transformer Architecture Defaults ---
TRANSFORMER_DEFAULT_D_MODEL = 256
TRANSFORMER_DEFAULT_H = 8
TRANSFORMER_DEFAULT_N_ENC = 4
TRANSFORMER_DEFAULT_N_DEC = 4
TRANSFORMER_DEFAULT_DROPOUT = 0.1
TRANSFORMER_DEFAULT_SEQ_LEN = 30
TRANSFORMER_DEFAULT_NB_SAMPLES = 50
TRANSFORMER_DEFAULT_MAX_NB_VAR = 7 # y + x1..x6

# --- 6. Transformer Data Generation Defaults ---
TRANSFORMER_GEN_NB_TRAILS = 10000
TRANSFORMER_GEN_NB_NODES_MIN = 2
TRANSFORMER_GEN_NB_NODES_MAX = 15
TRANSFORMER_GEN_TIMEOUT_SIMPLIFY = 5
TRANSFORMER_GEN_NB_NESTED_MAX = 6
TRANSFORMER_GEN_NB_CONSTANTS_MIN = 1
TRANSFORMER_GEN_NB_CONSTANTS_MAX = 1
TRANSFORMER_GEN_NB_VARIABLES_MAX = 6 
TRANSFORMER_GEN_NB_SAMPLING_PER_EQ = 25
TRANSFORMER_GEN_ORDER_OF_MAG_LIMIT = 1.0e+9
# We can re-use the default nb_samples and seq_len from section 5
TRANSFORMER_GEN_NB_SAMPLE_PTS = TRANSFORMER_DEFAULT_NB_SAMPLES
TRANSFORMER_GEN_SEQ_LENGTH_MAX = TRANSFORMER_DEFAULT_SEQ_LEN
TRANSFORMER_GEN_NB_ZFILL = 8

