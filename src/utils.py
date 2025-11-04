"""
Shared utility functions for the Symbolic Regression project.
Includes logging configuration and plotting functions.
"""

import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any

# Import config, assuming it exists at the root
try:
    import config
except ImportError:
    # Create a fallback config for paths if config.py is not set up
    class FallbackConfig:
        LOG_DIR = "logs"
    config = FallbackConfig()

def setup_logging():
    """
    Configures the root logger to output to both console (INFO) 
    and a log file (DEBUG).
    """
    log_dir = getattr(config, "LOG_DIR", "logs")
    log_file = os.path.join(log_dir, "symbolic_regression.log")

    # Create log directory if it doesn't exist
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create log directory {log_dir}. {e}", file=sys.stderr)

    # Define formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to lowest level

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (DEBUG level)
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        print(f"Warning: Could not open log file {log_file}. {e}", file=sys.stderr)

    logger.info("Logging configured. Logs will be saved to %s", log_file)

def symbolic_function(expr_str: str, **kwargs: Any) -> Any:
    """
    Safely evaluates a symbolic expression string.
    
    Args:
        expr_str: The expression to evaluate (e.g., "C * sin(x1)").
        **kwargs: A dictionary of variable names and their numpy array values
                  (e.g., x1=np.array([...]), C=1.5).

    Returns:
        The result of the evaluated expression.
    """
    # Define a safe global context for eval
    eval_globals = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        # Add stable log/sqrt
        "log": lambda x: np.log(np.abs(x) + 1e-8),
        "sqrt": lambda x: np.sqrt(np.abs(x)),
    }
    
    # Add all provided variables (x1, x2, C, etc.)
    eval_globals.update(kwargs)
    
    try:
        return eval(expr_str, eval_globals)
    except Exception as e:
        logging.error(f"Error evaluating expression '{expr_str}': {e}")
        return np.nan

def plot_symbolic(
    x_test: torch.Tensor, 
    y_test: torch.Tensor, 
    expr_str: str, 
    save_path: str,
    title: str = "Symbolic Regression Result"
):
    """
    Plots the true data against the predicted symbolic function and saves it.
    This function assumes x_test is 1D for plotting.
    """
    logger = logging.getLogger(__name__)
    
    # Ensure data is on CPU and in NumPy format
    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    if x_test_np.shape[1] != 1:
        logger.warning(
            f"Plotting skipped: x_test has {x_test_np.shape[1]} dimensions. "
            "Can only plot 1D functions."
        )
        return

    plt.figure(figsize=(10, 6))
    
    # Generate points for the predicted function
    x_min, x_max = np.min(x_test_np), np.max(x_test_np)
    x_range = x_max - x_min
    x_values = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)
    
    # Evaluate the expression
    y_values = symbolic_function(expr_str, x1=x_values, x_1=x_values)
    
    # Plot true data (scatter)
    plt.scatter(
        x_test_np, y_test_np, 
        color='red', 
        label='Test Data (True)', 
        alpha=0.6, 
        s=10
    )
    
    # Plot predicted function (line)
    if y_values is not None:
        plt.plot(
            x_values, y_values, 
            color='blue', 
            label=f'Predicted: {expr_str}', 
            linewidth=2
        )
    
    plt.title(title)
    plt.xlabel("x_1")
    plt.ylabel("y")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    try:
        plt.savefig(save_path)
        logger.info(f"Plot successfully saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {save_path}: {e}")
    
    plt.close() # Close the figure to free memory
