"""
Contains the main training, testing, and execution functions for
the EQL pipeline.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm.auto import tqdm
from typing import List, Union, Tuple

from .model import SymbolicNet
from .expression import get_expression
from .functions import BaseFunction, BaseFunction2

logger = logging.getLogger(__name__)

def train(
    net: SymbolicNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    learning_rate: float,
    epochs: int,
    reg_weight: float = 0.0,
    summary_step: int = 1000,
    device: str = "cpu"
) -> bool:
    """
    Trains a SymbolicNet model.

    Args:
        net: The SymbolicNet instance to train.
        x_train: Training input tensor.
        y_train: Training target tensor.
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        reg_weight: L1 regularization weight (for the "Bonus" part).
        summary_step: How often to log training loss.
        device: Torch device ("cpu" or "cuda").

    Returns:
        True if training succeeded, False if loss became NaN.
    """
    net.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    pbar = tqdm(range(epochs), desc="Training EQL")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(x_train)
        
        # Calculate loss
        mse_loss = F.mse_loss(outputs, y_train)
        
        # Add L1 regularization (Bonus)
        l1_loss = torch.tensor(0.0, device=device)
        if reg_weight > 0:
            for param in net.parameters():
                l1_loss += torch.sum(torch.abs(param))
        
        total_loss = mse_loss + reg_weight * l1_loss
        
        # Check for NaN loss
        if torch.isnan(total_loss):
            logger.warning(f"NaN loss encountered at epoch {epoch}. Stopping training.")
            return False # Training failed
            
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % summary_step == 0:
            pbar.set_postfix(loss=total_loss.item())
            
    return True # Training succeeded


def test(
    net: SymbolicNet, 
    x_test: torch.Tensor, 
    y_test: torch.Tensor, 
    device: str = "cpu"
) -> float:
    """
    Tests a trained SymbolicNet model.
    """
    net.eval() # Set model to evaluation mode
    net.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    with torch.no_grad():
        test_outputs = net(x_test)
        test_loss = F.mse_loss(test_outputs, y_test)

    if torch.isnan(test_loss):
        return float('inf')
        
    return test_loss.item()


def run(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    activation_funcs: List[Union[BaseFunction, BaseFunction2]],
    var_names: List[str],
    n_layers: int,
    learning_rate: float,
    reg_weight: float,
    epochs: int,
    summary_step: int,
    trials: int,
    device: str = "cpu"
) -> Tuple[str, float]:
    """
    Runs the full EQL training and testing pipeline for multiple trials
    and returns the best-performing expression.

    Args:
        All args are passed from the main script.
        var_names: Names of input variables (e.g., ['x_1', 'x_2']).
        
    Returns:
        A tuple of (best_expression_string, best_test_loss).
    """
    
    best_expr_str = "No valid expression found."
    best_test_loss = float('inf')
    x_dim = x_train.shape[1]
    
    for trial in range(trials):
        logger.info(f"--- Starting Trial {trial + 1} / {trials} ---")
        
        # 1. Initialize a new model
        net = SymbolicNet(
            n_layers=n_layers,
            in_dim=x_dim,
            funcs=activation_funcs,
            init_stddev=0.1
        )
        
        # 2. Train the model
        success = train(
            net, 
            x_train, y_train, 
            learning_rate, 
            epochs, 
            reg_weight,
            summary_step, 
            device
        )
        
        if not success:
            logger.warning("Trial failed due to NaN loss. Skipping.")
            continue
            
        # 3. Test the model
        test_loss = test(net, x_test, y_test, device)
        logger.info(f"Trial {trial + 1} Test Loss: {test_loss:.6f}")
        
        # 4. Retrieve the expression
        try:
            weights = net.get_weights()
            expr = get_expression(
                weights, 
                activation_funcs, 
                var_names[:x_dim],
                threshold=reg_weight # Use reg_weight as a proxy for pruning
            )
            expr_str = str(expr)
        except Exception as e:
            logger.error(f"Error retrieving expression in trial {trial + 1}: {e}")
            expr_str = "Error"

        logger.info(f"Trial {trial + 1} Expression: {expr_str}")

        # 5. Update best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_expr_str = expr_str
            logger.info(f"*** New best loss found! ***")
            
    logger.info(f"\n--- EQL Run Complete ---")
    logger.info(f"Best Test Loss: {best_test_loss:.6f}")
    logger.info(f"Best Expression: {best_expr_str}")
    
    return best_expr_str, best_test_loss
