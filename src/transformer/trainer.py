"""
Contains the training and validation step functions (Cell 43), as well as the
main `train_loop` function (Cell 44) that orchestrates the entire 
training process.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm
import logging
import time

logger = logging.getLogger(__name__)

def training_step(
    model: nn.Module,
    batch: tuple,
    loss_fn: callable,
    acc_fn: callable,
    optimizer: Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs a single training step (forward pass, loss, backward pass,
    optimizer step). Based on Cell 43.
    """
    model.train() # Set model to training mode
    
    # 1. Get data and move to device
    data, target_full = batch
    trainX = data.to(device)
    # Decoder input is <SOS> and all tokens *except* the last one
    trainY = target_full[:, :-1].to(device) 
    # Target is all tokens *except* <SOS>
    target = target_full[:, 1:].to(device)  
    
    # 2. Forward pass
    optimizer.zero_grad()
    preds = model(trainX, trainY) # (batch, seq_len, vocab_size)
    
    # 3. Compute loss and accuracy
    loss = loss_fn(preds, target)
    acc = acc_fn(preds, target)
    
    # 4. Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    return loss.item(), acc.item()


def validation_step(
    model: nn.Module,
    batch: tuple,
    loss_fn: callable,
    acc_fn: callable,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs a single validation step (forward pass, loss).
    No gradients are computed. Based on Cell 43.
    """
    model.eval() # Set model to evaluation mode
    
    # 1. Get data and move to device
    data, target_full = batch
    valX = data.to(device)
    valY = target_full[:, :-1].to(device)
    target = target_full[:, 1:].to(device)
    
    # 2. Forward pass (no gradients)
    with torch.no_grad():
        preds = model(valX, valY)
        loss = loss_fn(preds, target)
        acc = acc_fn(preds, target)
        
    return loss.item(), acc.item()


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: callable,
    acc_fn: callable,
    epochs: int,
    device: torch.device,
    model_save_path: str
) -> dict:
    """
    The main training loop, refactored from Cell 44.
    
    Args:
        model: The Transformer model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: The optimizer.
        loss_fn: The loss function (e.g., compute_transformer_loss).
        acc_fn: The accuracy function (e.g., compute_transformer_accuracy).
        epochs: Number of epochs to train.
        device: Torch device.
        model_save_path: Path to save the best model.
        
    Returns:
        A dictionary containing training history.
    """
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    logger.info(f"START TRAINING! [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    
    for epoch in range(1, epochs + 1):
        
        # --- Training Phase ---
        train_loss_epoch = 0.0
        train_acc_epoch = 0.0
        # Use tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"== Epoch {epoch}/{epochs} [Train] ==", leave=False)
        
        for batch in train_pbar:
            loss, acc = training_step(model, batch, loss_fn, acc_fn, optimizer, device)
            train_loss_epoch += loss
            train_acc_epoch += acc
            train_pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
            
        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_acc = train_acc_epoch / len(train_loader)

        # --- Validation Phase ---
        val_loss_epoch = 0.0
        val_acc_epoch = 0.0
        val_pbar = tqdm(val_loader, desc=f"== Epoch {epoch}/{epochs} [Val] ==", leave=False)

        for batch in val_pbar:
            loss, acc = validation_step(model, batch, loss_fn, acc_fn, device)
            val_loss_epoch += loss
            val_acc_epoch += acc
            val_pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")

        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_acc = val_acc_epoch / len(val_loader)

        # --- Logging and Model Saving ---
        logger.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{epochs} | "
            f"Train loss = {avg_train_loss:.4f} ; Train acc = {avg_train_acc:.4f} | "
            f"Val loss = {avg_val_loss:.4f} ; Val acc = {avg_val_acc:.4f}"
        )
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
            
    logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return history
