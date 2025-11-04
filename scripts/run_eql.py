"""
Main script to run the EQL (Equation Learner) symbolic regression pipeline.

Example Usage:
1. Run on generated Dataset 1:
   python scripts/run_eql.py --dataset_path dataset_1 --n_layers 2 --epochs 20000

2. Run on provided Dataset 2 (and find simpler expr):
   python scripts/run_eql.py --dataset_path data/eql_dataset_2.csv --n_layers 2 --reg_weight 0.01
"""

import argparse
import logging
import os
import torch
import numpy as np
import sys

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    from src.utils import setup_logging, plot_symbolic
    from src.eql.data import generate_data, load_eql_data
    from src.eql.functions import Constant, Identity, Square, Sin, Exp, Sigmoid, Product
    from src.eql.trainer import run
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running from the project root and src is in your PYTHONPATH.")
    sys.exit(1)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run EQL Symbolic Regression")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset CSV file, or 'dataset_1' to auto-generate."
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the SymbolicNet."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of independent trials to run."
    )
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=0.005,
        help="L1 regularization weight to encourage sparsity (for Bonus)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=getattr(config, "OUTPUT_DIR", "outputs"),
        help="Directory to save plots and results."
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting EQL run...")
    logger.info(f"Arguments: {vars(args)}")

    device = getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load or Generate Data
    if args.dataset_path == 'dataset_1':
        logger.info("Generating Dataset 1 (2*x + sin(x) + x*sin(x))")
        func = lambda x: 2 * x + np.sin(x) + x * np.sin(x)
        x_train, y_train = generate_data(func, 240, -2, 2, noise_std=0.1)
        x_test, y_test = generate_data(func, 60, -10, 10, noise_std=0.1)
    else:
        logger.info(f"Loading data from {args.dataset_path}")
        x_data, y_data = load_eql_data(args.dataset_path)
        if x_data is None:
            logger.critical("Failed to load data. Exiting.")
            return
        # Use the same data for training and testing
        x_train, y_train = x_data, y_data
        x_test, y_test = x_data, y_data
        
    logger.info(f"Train data shape: x={x_train.shape}, y={y_train.shape}")
    logger.info(f"Test data shape:  x={x_test.shape}, y={y_test.shape}")

    # 2. Define Activation Functions
    activation_funcs = [
        *[Constant()] * 2,
        *[Identity()] * 4,
        *[Square()] * 4,
        *[Sin(freq=1.0)] * 2,  # Using standard sin(x)
        *[Exp()] * 2,
        *[Sigmoid()] * 2,
        *[Product()] * 2,
    ]
    var_names = ["x_1", "x_2"] # Max var names for expression retrieval

    # 3. Run the EQL pipeline
    best_expr_str, best_loss = run(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        activation_funcs=activation_funcs,
        var_names=var_names,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        epochs=args.epochs,
        summary_step=1000,
        trials=args.trials,
        device=device
    )

    # 4. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save expression text
    expr_filename = f"eql_expr_{os.path.basename(args.dataset_path).split('.')[0]}.txt"
    expr_path = os.path.join(args.output_dir, expr_filename)
    try:
        with open(expr_path, 'w') as f:
            f.write(f"Best Test Loss: {best_loss}\n")
            f.write(f"Best Expression: {best_expr_str}\n")
        logger.info(f"Best expression saved to {expr_path}")
    except IOError as e:
        logger.error(f"Failed to save expression to {expr_path}: {e}")

    # Save plot (only if 1D)
    if x_train.shape[1] == 1:
        plot_filename = f"eql_plot_{os.path.basename(args.dataset_path).split('.')[0]}.png"
        plot_path = os.path.join(args.output_dir, plot_filename)
        
        # Clean expression for eval
        plot_expr = best_expr_str.replace("exp", "np.exp").replace("sin", "np.sin")
        
        plot_symbolic(
            x_test, 
            y_test, 
            plot_expr, 
            plot_path,
            title=f"EQL Result (Dataset: {args.dataset_path})"
        )
    
    logger.info("EQL run finished.")

if __name__ == "__main__":
    main()
