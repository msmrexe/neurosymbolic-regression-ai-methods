"""
Main script to evaluate a trained Transformer model on specific datasets.
Finds the symbolic expression and optimizes for constants.

Example Usage:
python scripts/evaluate_transformer.py --model_path outputs/transformer_best.pth
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
from scipy.optimize import minimize
import re

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    from src.utils import setup_logging, plot_symbolic, symbolic_function
    from src.eql.data import generate_data, load_eql_data
    from src.transformer.architecture import TransformerModel
    from src.transformer.utils import evaluate, VOCAB_MAP
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def parse_args():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Transformer Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model (.pth) file."
    )
    parser.add_argument(
        "--data_2_path",
        type=str,
        default=config.HIDDEN_DATA_PATH,
        help="Path to the EQL Dataset 2 (hidden formula) CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory to save plots and results."
    )

    # Model Architecture (must match the trained model)
    parser.add_argument("--d_model", type=int, default=config.TRANSFORMER_DEFAULT_D_MODEL, help="Model embedding dimension.")
    parser.add_argument("--h", type=int, default=config.TRANSFORMER_DEFAULT_H, help="Number of attention heads.")
    parser.add_argument("--n_enc", type=int, default=config.TRANSFORMER_DEFAULT_N_ENC, help="Number of Encoder layers.")
    parser.add_argument("--n_dec", type=int, default=config.TRANSFORMER_DEFAULT_N_DEC, help="Number of Decoder layers.")
    parser.add_argument("--seq_length", type=int, default=config.TRANSFORMER_DEFAULT_SEQ_LEN, help="Max sequence length.")
    parser.add_argument("--nb_samples", type=int, default=config.TRANSFORMER_DEFAULT_NB_SAMPLES, help="Number of (x,y) samples.")
    parser.add_argument("--max_nb_var", type=int, default=config.TRANSFORMER_DEFAULT_MAX_NB_VAR, help="Max variables (y + 6*x).")
    
    return parser.parse_args()

def optimize_constants(expr_str: str, x_data: np.ndarray, y_data: np.ndarray) -> str:
    """
    Finds the best value for constants (e.g., 'C') in the expression using MSE loss.
    """
    logger = logging.getLogger(__name__)
    
    # Find all instances of 'C' (or 'C_1', 'C_2' if we extend this)
    constants_to_find = sorted(list(set(re.findall(r"(C)", expr_str))))
    
    if not constants_to_find:
        return expr_str # No constants to optimize

    def objective(constants: np.ndarray, expr_str: str, x_data: np.ndarray, y_data: np.ndarray) -> float:
        
        # Build kwargs for symbolic_function
        kwargs = {}
        for i, const_name in enumerate(constants_to_find):
            kwargs[const_name] = constants[i]
            
        for i in range(x_data.shape[1]):
            kwargs[f'x{i+1}'] = x_data[:, i]
            kwargs[f'x_{i+1}'] = x_data[:, i] # Handle both 'x1' and 'x_1'
            
        y_pred = symbolic_function(expr_str, **kwargs)
        
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e10 # Penalize invalid expressions
            
        mse = np.mean((y_pred - y_data.squeeze())**2)
        return mse if np.isfinite(mse) else 1e10

    logger.info(f"Optimizing {len(constants_to_find)} constant(s): {constants_to_find}")
    initial_guess = np.ones(len(constants_to_find))
    
    result = minimize(
        objective, 
        initial_guess, 
        args=(expr_str, x_data, y_data), 
        method='L-BFGS-B'
    )
    
    final_expr = expr_str
    if result.success:
        best_constants = result.x
        logger.info(f"Optimization successful. Best constants: {best_constants}")
        for const_name, const_val in zip(constants_to_find, best_constants):
            # Replace 'C' with its optimized value, formatted to 4 decimal places
            final_expr = final_expr.replace(const_name, f"({const_val:.4f})")
    else:
        logger.warning("Constant optimization failed. Using C=1.0.")
        for const_name in constants_to_find:
            final_expr = final_expr.replace(const_name, "(1.0)")
        
    return final_expr

def evaluate_dataset_1(model, args, device):
    """Generates and evaluates on Dataset 1."""
    logger = logging.getLogger(__name__)
    logger.info("--- Evaluating Dataset 1 ---")
    
    # 1. Generate data
    func = lambda x: 2 * x + np.sin(x) + x * np.sin(x)
    x_test, y_test = generate_data(
        func, args.nb_samples, -10, 10, noise_std=0.0
    ) # No noise for eval
    
    # 2. Format for Transformer
    # Shape: (batch, samples, vars, features) -> (1, 50, 7, 1)
    dataset = torch.zeros(1, args.nb_samples, args.max_nb_var, 1)
    dataset[0, :, 0, 0] = y_test.squeeze() # y
    dataset[0, :, 1, 0] = x_test.squeeze() # x1
    
    # 3. Evaluate
    sympy_pred, string_pred = evaluate(
        model, dataset, args.seq_length, device
    )
    logger.info(f"Raw prediction (Dataset 1): {string_pred}")
    
    # 4. Optimize and Plot
    final_expr = optimize_constants(string_pred, x_test.numpy(), y_test.numpy())
    logger.info(f"Optimized prediction (Dataset 1): {final_expr}")
    
    plot_path = os.path.join(args.output_dir, "transformer_plot_dataset_1.png")
    plot_symbolic(
        x_test, y_test, final_expr, plot_path,
        title="Transformer Result (Dataset 1)"
    )

def evaluate_dataset_2(model, args, device):
    """Loads and evaluates on Dataset 2."""
    logger = logging.getLogger(__name__)
    logger.info("--- Evaluating Dataset 2 ---")
    
    # 1. Load data
    x_test, y_test = load_eql_data(args.data_2_path)
    if x_test is None:
        logger.error(f"Failed to load {args.data_2_path}. Skipping.")
        return
        
    # 2. Format for Transformer
    # Ensure we only use nb_samples
    if len(x_test) > args.nb_samples:
        x_test = x_test[:args.nb_samples]
        y_test = y_test[:args.nb_samples]
    elif len(x_test) < args.nb_samples:
        logger.warning(f"Dataset 2 has only {len(x_test)} samples, but model expects {args.nb_samples}.")
        # We'll pad with zeros
        x_pad = torch.zeros(args.nb_samples, x_test.shape[1])
        y_pad = torch.zeros(args.nb_samples, 1)
        x_pad[:len(x_test)] = x_test
        y_pad[:len(y_test)] = y_test
        x_test, y_test = x_pad, y_pad

    dataset = torch.zeros(1, args.nb_samples, args.max_nb_var, 1)
    dataset[0, :, 0, 0] = y_test.squeeze() # y
    dataset[0, :, 1, 0] = x_test[:, 0]     # x1
    dataset[0, :, 2, 0] = x_test[:, 1]     # x2
    
    # 3. Evaluate
    sympy_pred, string_pred = evaluate(
        model, dataset, args.seq_length, device
    )
    logger.info(f"Raw prediction (Dataset 2): {string_pred}")
    
    # 4. Optimize
    # Use the original, non-padded data for optimization
    x_test_np, y_test_np = load_eql_data(args.data_2_path) 
    final_expr = optimize_constants(string_pred, x_test_np.numpy(), y_test_np.numpy())
    logger.info(f"Optimized prediction (Dataset 2): {final_expr}")
    
    # (Plotting is not done for 2D data, as per `plot_symbolic` logic)
    expr_path = os.path.join(args.output_dir, "transformer_expr_dataset_2.txt")
    try:
        with open(expr_path, 'w') as f:
            f.write(f"Optimized Expression: {final_expr}\n")
        logger.info(f"Dataset 2 expression saved to {expr_path}")
    except IOError as e:
        logger.error(f"Failed to save Dataset 2 expression: {e}")


def main():
    """Main evaluation execution function."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Transformer evaluation...")
    
    device = config.DEVICE
    logger.info(f"Using device: {device}")

    # 1. Load Model
    if not os.path.exists(args.model_path):
        logger.critical(f"Model file not found: {args.model_path}")
        return
        
    vocab_size = len(VOCAB_MAP) + 2
    try:
        model = TransformerModel(
            nb_samples=args.nb_samples,
            max_nb_var=args.max_nb_var,
            d_model=args.d_model,
            vocab_size=vocab_size,
            seq_length=args.seq_length,
            h=args.h,
            N_enc=args.n_enc,
            N_dec=args.n_dec,
            dropout=0.0 # No dropout for eval
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Model successfully loaded from {args.model_path}")
    except Exception as e:
        logger.critical(f"Failed to load model. Ensure architecture args match! Error: {e}")
        return

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Run evaluations
    evaluate_dataset_1(model, args, device)
    evaluate_dataset_2(model, args, device)
    
    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()
