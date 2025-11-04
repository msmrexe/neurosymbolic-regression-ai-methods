"""
Main script to train the Transformer model for symbolic regression.

Example Usage:
python scripts/train_transformer.py --epochs 100 --batch_size 64 --d_model 256
"""

import argparse
import logging
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import glob

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    from src.utils import setup_logging
    from src.transformer.architecture import TransformerModel
    from src.transformer.dataset import SymbolicRegressionDataset
    from src.transformer.utils import (
        compute_transformer_loss, 
        compute_transformer_accuracy, 
        VOCAB_MAP, 
        PAD_TOKEN
    )
    from src.transformer.trainer import train_loop
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def parse_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    
    # Paths and Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.TRANSFORMER_DATA_PATH,
        help="Directory where generated data is stored."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="transformer_best.pth",
        help="Name for the saved model file."
    )
    parser.add_argument("--train_prop", type=float, default=0.7, help="Training data proportion.")
    parser.add_argument("--val_prop", type=float, default=0.15, help="Validation data proportion.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=config.TRANSFORMER_DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.TRANSFORMER_DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=config.TRANSFORMER_DEFAULT_LR, help="Adam optimizer learning rate.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for loss.")

    # Model Architecture (Defaults pulled from config)
    parser.add_argument("--d_model", type=int, default=config.TRANSFORMER_DEFAULT_D_MODEL, help="Model embedding dimension.")
    parser.add_argument("--h", type=int, default=config.TRANSFORMER_DEFAULT_H, help="Number of attention heads.")
    parser.add_argument("--n_enc", type=int, default=config.TRANSFORMER_DEFAULT_N_ENC, help="Number of Encoder layers.")
    parser.add_argument("--n_dec", type=int, default=config.TRANSFORMER_DEFAULT_N_DEC, help="Number of Decoder layers.")
    parser.add_argument("--dropout", type=float, default=config.TRANSFORMER_DEFAULT_DROPOUT, help="Dropout rate.")
    parser.add_argument("--seq_length", type=int, default=config.TRANSFORMER_DEFAULT_SEQ_LEN, help="Max sequence length (from generation).")
    parser.add_argument("--nb_samples", type=int, default=config.TRANSFORMER_DEFAULT_NB_SAMPLES, help="Number of (x,y) samples per expression.")
    parser.add_argument("--max_nb_var", type=int, default=config.TRANSFORMER_DEFAULT_MAX_NB_VAR, help="Max variables (y + 6*x).")

    return parser.parse_args()

def main():
    """Main training execution function."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Transformer training...")
    logger.info(f"Arguments: {vars(args)}")
    
    device = config.DEVICE
    logger.info(f"Using device: {device}")

    # 1. Load and Split Data
    logger.info(f"Loading data from {args.data_path}")
    full_dataset = SymbolicRegressionDataset(
        data_path=args.data_path, 
        max_seq_len=args.seq_length
    )
    
    if len(full_dataset) == 0:
        logger.critical("No data found. Please run data generation first. Exiting.")
        return
        
    n_train = int(len(full_dataset) * args.train_prop)
    n_val = int(len(full_dataset) * args.val_prop)
    n_test = len(full_dataset) - n_train - n_val
    logger.info(f"Splitting data: {n_train} train, {n_val} val, {n_test} test")

    # We set a fixed generator for reproducibility of splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    # test_loader is not used in this script

    # 2. Initialize Model
    # Vocab size = len(tokens) + PAD (0) + SOS (1)
    vocab_size = len(VOCAB_MAP) + 2 
    
    model = TransformerModel(
        nb_samples=args.nb_samples,
        max_nb_var=args.max_nb_var,
        d_model=args.d_model,
        vocab_size=vocab_size,
        seq_length=args.seq_length,
        h=args.h,
        N_enc=args.n_enc,
        N_dec=args.n_dec,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    # 3. Initialize Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    loss_fn = lambda p, t: compute_transformer_loss(p, t, args.label_smoothing)
    acc_fn = compute_transformer_accuracy

    # 4. Run Training
    model_save_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        epochs=args.epochs,
        device=device,
        model_save_path=model_save_path
    )
    
    logger.info("Training finished.")
    logger.info(f"Best model saved to {model_save_path}")

if __name__ == "__main__":
    main()
