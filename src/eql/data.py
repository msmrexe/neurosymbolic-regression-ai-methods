"""
Contains data generation and loading functions for the EQL model.
"""

import torch
import pandas as pd
import logging
from typing import Callable, Tuple, Union
from inspect import signature

logger = logging.getLogger(__name__)

def generate_data(
    func: Callable, 
    n_samples: int, 
    range_min: float = -2.0, 
    range_max: float = 2.0, 
    noise_std: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset by evaluating a function on random inputs with noise.

    Args:
        func: A callable function (e.g., lambda x, y: ...).
        n_samples: The number of data points to generate.
        range_min: The minimum value of the input range.
        range_max: The maximum value of the input range.
        noise_std: Standard deviation of Gaussian noise to add.

    Returns:
        A tuple (x, y) of tensors.
        x: Input data tensor of shape (n_samples, x_dim).
        y: Target data tensor of shape (n_samples, 1).
    """
    try:
        x_dim = len(signature(func).parameters)
    except Exception as e:
        logger.error(f"Could not determine dimension from function signature: {e}")
        raise
        
    logger.info(f"Generating {n_samples} samples for {x_dim}-dimensional function.")
    
    # Generate random inputs
    x = (range_max - range_min) * torch.rand([n_samples, x_dim]) + range_min
    
    # Generate outputs by applying the function
    try:
        y_list = [func(*x_i) for x_i in x]
        y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)
    except Exception as e:
        logger.error(f"Error evaluating function on generated data: {e}")
        raise

    # Add Gaussian noise
    noise = torch.randn(y.shape) * noise_std
    y_noisy = y + noise

    return x, y_noisy


def load_eql_data(
    csv_path: str
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
    """
    Loads a CSV dataset provided for the EQL model.
    The CSV must have 'y' as the target column and 'x_1', 'x_2', etc.,
    as input columns.

    

    Args:
        csv_path: The file path to the .csv file.

    Returns:
        A tuple (x, y) of tensors, or (None, None) if loading fails.
        x: Input data tensor.
        y: Target data tensor.
    """
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Data file not found at path: {csv_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {e}")
        return None, None

    if 'y' not in data.columns:
        logger.error(f"CSV file {csv_path} must contain a 'y' column.")
        return None, None

    # Find all input columns (e.g., 'x_1', 'x_2')
    x_cols = sorted([col for col in data.columns if col.startswith('x_')])
    
    if not x_cols:
        logger.error(f"CSV file {csv_path} must contain at least one 'x_...' column.")
        return None, None

    logger.info(f"Loaded {len(data)} samples with {len(x_cols)} features from {csv_path}.")

    x = torch.tensor(data[x_cols].values, dtype=torch.float32)
    y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)

    return x, y
