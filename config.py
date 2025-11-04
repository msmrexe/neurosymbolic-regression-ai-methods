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

# --- 3. Default Hyperparameters (can be overridden by script args) ---

# EQL Defaults
EQL_DEFAULT_EPOCHS = 20000
EQL_DEFAULT_LR = 1e-3

# Transformer Defaults
TRANSFORMER_DEFAULT_EPOCHS = 100
TRANSFORMER_DEFAULT_BATCH_SIZE = 128
TRANSFORMER_DEFAULT_LR = 1e-4
TRANSFORMER_DEFAULT_D_MODEL = 256
