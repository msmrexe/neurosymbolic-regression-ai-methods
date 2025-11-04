"""
This script contains shared utilities for the Transformer model, including:
- Vocabulary and Symbol definitions
- Loss and Accuracy computation
- Expression parsing
- The evaluation (greedy decoding) function
"""

import torch
import torch.nn as nn
import numpy as np
import sympy
from typing import List, Tuple, Dict, Any

# --- 1. Vocabulary and Symbol Definitions ---

PAD_TOKEN = 0
SOS_TOKEN = 1

# Note: Token IDs in the vocab map start at 2
MY_VOCAB_LIST = [
    'add', 'mul', 'sin', 'cos', 'log', 'exp', 'neg', 'inv', 'sqrt', 'sq', 'cb', 
    'C', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'
]

# Map token strings to their integer ID (starting from 2)
VOCAB_MAP: Dict[str, int] = {token: i + 2 for i, token in enumerate(MY_VOCAB_LIST)}
# Map integer ID back to token string
ID_TO_TOKEN_MAP: Dict[int, str] = {i + 2: token for i, token in enumerate(MY_VOCAB_LIST)}


# The array used by the original data generator (for arity lookup)
# Note: It includes tokens like 'sub' and 'cbrt' which are not in the
# final vocabulary. We only use it for its arity info.
MY_VOCAB_ARRAY = np.array([
    ['add', 4, 2],  # binary operators
    ['sub', 3, 2],
    ['mul', 6, 2],
    ['sin', 1, 1],  # unary operators
    ['cos', 1, 1],
    ['log', 2, 1],
    ['exp', 2, 1],
    ['neg', 0, 1],
    ['inv', 3, 1],
    ['sq', 2, 1],
    ['cb', 0, 1],
    ['sqrt', 2, 1],
    ['cbrt', 0, 1],
    ['C', 8, 0],    # leaves
    ['x1', 8, 0],
    ['x2', 8, 0],
    ['x3', 4, 0],
    ['x4', 4, 0],
    ['x5', 2, 0],
    ['x6', 2, 0],
])

# Create a simple map of token -> arity for parsing
ARITY_MAP: Dict[str, int] = {
    row[0]: int(row[2]) for row in MY_VOCAB_ARRAY
}
# Add entries for any constants that might appear (arity 0)
ARITY_MAP.update({str(i): 0 for i in range(10)})

# SymPy symbols
C, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, x1, x2, x3, x4, x5, x6', real=True, positive=True)
SYMPY_SYMBOL_MAP = {'C': C, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6}


# --- 2. Loss and Accuracy Functions ---

def compute_transformer_loss(
    prediction: torch.Tensor, 
    target: torch.Tensor, 
    label_smooth: float = 0.0
) -> torch.Tensor:
    """
    Computes the cross-entropy loss with label smoothing, ignoring pad tokens.
    
    Args:
        prediction: Logits from model, shape (batch_size, seq_length, vocab_size)
        target: Ground truth token IDs, shape (batch_size, seq_length)
        label_smooth: Amount of label smoothing.
    """
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN, 
        label_smoothing=label_smooth
    )
    
    # CrossEntropyLoss expects (N, C, ...) and (N, ...)
    # Prediction: (batch, seq_len, vocab) -> (batch, vocab, seq_len)
    # Target: (batch, seq_len)
    return criterion(prediction.transpose(1, 2), target.long())


def compute_transformer_accuracy(
    prediction: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """
    Computes token-level accuracy, ignoring pad tokens.
    
    Args:
        prediction: Logits from model, shape (batch_size, seq_length, vocab_size)
        target: Ground truth token IDs, shape (batch_size, seq_length)
    """
    non_pad_mask = (target != PAD_TOKEN)
    
    # Get most likely token IDs
    preds = torch.argmax(prediction, dim=-1) # (batch_size, seq_length)
    
    correct_tokens = (preds == target.long())
    correct_non_pad = torch.logical_and(correct_tokens, non_pad_mask)
    
    total_non_pad_tokens = torch.sum(non_pad_mask).float()
    
    if total_non_pad_tokens == 0:
        return torch.tensor(1.0) # Avoid 0/0 if batch is all padding
        
    return torch.sum(correct_non_pad).float() / total_non_pad_tokens


# --- 3. Expression Parsing Functions ---

def token_ids_to_words(seq_int: List[int]) -> List[str]:
    """Converts a list of token IDs back to string tokens."""
    seq_tokens = []
    for token_id in seq_int:
        if token_id in ID_TO_TOKEN_MAP:
            seq_tokens.append(ID_TO_TOKEN_MAP[token_id])
        # Ignore PAD (0) and SOS (1)
    return seq_tokens

def is_expression_complete(seq_int: List[int]) -> bool:
    """
    Checks if a sequence of token IDs forms a complete expression tree.
    """
    arity = 1  # Start with arity 1 (root node)
    
    for token_id in seq_int:
        if token_id < 2: # Skip PAD or SOS
            continue
            
        token = ID_TO_TOKEN_MAP.get(token_id)
        if token is None: # Should not happen
            return False
            
        token_arity = ARITY_MAP.get(token, -1)
        if token_arity == -1: # Unknown token
            return False
            
        arity += (token_arity - 1)
        if arity <= 0:
            break # Tree is complete or over-full

    return arity == 0


def build_sympy_from_sequence(expr_list: List[str]) -> sympy.Expr:
    """
    Recursively converts a list of string tokens (in prefix notation)
    to a SymPy expression.
    """
    if not expr_list:
        return None
        
    token = expr_list.pop(0) # Get and remove the first token
    
    try:
        # Handle numerical constants (e.g., 'C=1.23' was simplified to '1.23')
        return sympy.Float(token)
    except ValueError:
        pass

    arity = ARITY_MAP.get(token)
    
    if arity is None:
        raise ValueError(f"Unknown token in sequence: {token}")

    # --- Arity 0 (Leaf) ---
    if arity == 0:
        return SYMPY_SYMBOL_MAP.get(token)
    
    # --- Arity 1 (Unary) ---
    arg = build_sympy_from_sequence(expr_list)
    if arity == 1:
        if token == 'sin': return sympy.sin(arg)
        if token == 'cos': return sympy.cos(arg)
        if token == 'log': return sympy.log(sympy.Abs(arg) + 1e-6) # Add epsilon for stability
        if token == 'exp': return sympy.exp(arg)
        if token == 'neg': return -arg
        if token == 'inv': return 1.0 / arg
        if token == 'sq': return arg**2
        if token == 'sqrt': return sympy.sqrt(sympy.Abs(arg)) # Use Abs for stability
        if token == 'cb': return arg**3
        raise ValueError(f"Unhandled unary token: {token}")

    # --- Arity 2 (Binary) ---
    arg2 = build_sympy_from_sequence(expr_list)
    if arity == 2:
        if token == 'add': return sympy.Add(arg, arg2)
        if token == 'mul': return sympy.Mul(arg, arg2)
        if token == 'sub': return sympy.Add(arg, -arg2) # Use Add/Neg
        raise ValueError(f"Unhandled binary token: {token}")

    raise ValueError(f"Token {token} with arity {arity} not handled.")


def build_string_from_sequence(expr_list: List[str]) -> str:
    """
    Recursively converts a list of string tokens (in prefix notation)
    to an evaluatable Python string.
    """
    if not expr_list:
        return ""
        
    token = expr_list.pop(0)
    
    try:
        float(token)
        return token # It's a number
    except ValueError:
        pass
        
    arity = ARITY_MAP.get(token)
    if arity is None: raise ValueError(f"Unknown token: {token}")

    # --- Arity 0 (Leaf) ---
    if arity == 0:
        return token # It's a variable or 'C'
    
    # --- Arity 1 (Unary) ---
    arg = build_string_from_sequence(expr_list)
    if arity == 1:
        if token == 'sin': return f"sin({arg})"
        if token == 'cos': return f"cos({arg})"
        if token == 'log': return f"log(np.abs({arg}) + 1e-6)"
        if token == 'exp': return f"exp({arg})"
        if token == 'neg': return f"(-{arg})"
        if token == 'inv': return f"(1.0 / ({arg}))"
        if token == 'sq': return f"({arg}**2)"
        if token == 'sqrt': return f"sqrt(np.abs({arg}))"
        if token == 'cb': return f"({arg}**3)"
        raise ValueError(f"Unhandled unary token: {token}")

    # --- Arity 2 (Binary) ---
    arg2 = build_string_from_sequence(expr_list)
    if arity == 2:
        if token == 'add': return f"({arg} + {arg2})"
        if token == 'mul': return f"({arg} * {arg2})"
        if token == 'sub': return f"({arg} - {arg2})"
        raise ValueError(f"Unhandled binary token: {token}")
        
    raise ValueError(f"Token {token} with arity {arity} not handled.")


# --- Functions below are used by data_generator.py ---

def _from_sympy_power(exponent):
    """Helper for build_sequence_from_sympy"""
    if exponent == 2: return ['sq']
    if exponent == 3: return ['cb']
    if exponent == 4: return ['sq', 'sq']
    if exponent == -1: return ['inv']
    if exponent == -2: return ['inv', 'sq']
    if exponent == -3: return ['inv', 'cb']
    if exponent == 0.5: return ['sqrt']
    if exponent == -0.5: return ['inv', 'sqrt']
    return ['abort'] # Abort if power is not in our vocab

def _from_sympy_multiplication(expr):
    """Helper for build_sequence_from_sympy"""
    seq = []
    is_neg = False
    nb_constants = 0
    factors = []

    for arg in expr.args:
        if arg == -1:
            is_neg = True
        elif arg.is_constant():
            nb_constants += 1
        else:
            factors.append(arg)
    
    if is_neg: seq.append('neg')
    for _ in range(len(factors) - 1): seq.append('mul')
    if nb_constants > 0:
        seq.append('mul')
        seq.append('C')
    
    for factor in factors:
        seq.extend(build_sequence_from_sympy(factor))
    return seq

def _from_sympy_addition(expr):
    """Helper for build_sequence_from_sympy"""
    seq = []
    nb_constants = 0
    terms = []
    for arg in expr.args:
        if arg.is_constant():
            nb_constants += 1
        else:
            terms.append(arg)

    for _ in range(len(terms) - 1): seq.append('add')
    if nb_constants > 0:
        seq.append('add')
        seq.append('C')

    for term in terms:
        seq.extend(build_sequence_from_sympy(term))
    return seq

def build_sequence_from_sympy(expr: sympy.Expr) -> List[str]:
    """
    Converts a simplified SymPy expression back into a standardized
    token sequence.
    """
    if not expr.args: # Leaf node (variable or constant)
        return [str(expr)]
    
    # Unary functions
    if len(expr.args) == 1:
        func_name = str(expr.func)
        if func_name in ['sin', 'cos', 'log', 'exp', 'sqrt']:
             return [func_name] + build_sequence_from_sympy(expr.args[0])
        raise ValueError(f"Unknown unary func: {func_name}")

    # Binary/N-ary functions
    if expr.is_Pow:
        base, exp = expr.args
        power_seq = _from_sympy_power(exp)
        if 'abort' in power_seq: return ['abort']
        return power_seq + build_sequence_from_sympy(base)
    
    if expr.is_Mul:
        return _from_sympy_multiplication(expr)
        
    if expr.is_Add:
        return _from_sympy_addition(expr)
        
    raise ValueError(f"Unknown expression type: {expr.func}")


def sample_points_from_expression(
    sympy_expr: sympy.Expr, 
    nb_samples: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples data points from a given SymPy expression.
    """
    # Sample variables from log-uniform distribution
    np_x = np.power(10.0, np.random.uniform(low=-1.0, high=1.0, size=(nb_samples, 6)))
    
    # Create a callable function from the expression
    f = sympy.lambdify([x1, x2, x3, x4, x5, x6], sympy_expr, 'numpy')
    
    # Generate y values
    np_y = f(np_x[:, 0], np_x[:, 1], np_x[:, 2], np_x[:, 3], np_x[:, 4], np_x[:, 5])
    
    # Ensure y is always an array
    if not isinstance(np_y, np.ndarray):
        np_y = np.full(nb_samples, np_y)
        
    return np_y, np_x


def count_variables_in_expression(sympy_expr: sympy.Expr) -> int:
    """
    Counts the number of unique variables (x1, x2, etc.) in an expression.
    """
    nb_variables = 0
    expr_str = str(sympy_expr)
    for i in range(1, 7):
        if f'x{i}' in expr_str:
            nb_variables += 1
        else:
            break # Assumes variables are normalized (x1, x2, ... not x1, x3)
    return nb_variables


def expression_tree_depth(sympy_expr: Any) -> int:
    """
    Recursively counts the maximum depth of a SymPy expression tree.
    """
    if not isinstance(sympy_expr, sympy.Expr) or not sympy_expr.args:
        return 1 # Leaf node
    
    max_child_depth = 0
    for arg in sympy_expr.args:
        max_child_depth = max(max_child_depth, expression_tree_depth(arg))
        
    return 1 + max_child_depth


# --- 4. Evaluation Function ---

def evaluate(
    model: nn.Module, 
    dataset: torch.Tensor, 
    max_seq_len: int,
    device: torch.device
) -> Tuple[sympy.Expr, str]:
    """
    Performs greedy decoding to predict an expression for a given dataset.
    
    Args:
        model: The trained TransformerModel.
        dataset: Input data tensor, shape (1, nb_samples, max_nb_var, 1)
        max_seq_len: The maximum length to decode.
        device: The torch device (e.g., 'cuda' or 'cpu').
    
    Returns:
        (sympy_pred, string_pred): The predicted expression in SymPy and
                                   string format.
    """
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        encoder_input = dataset.to(device)
        
        # 1. Run Encoder ONCE
        # (1, samples, vars, 1) -> (1, d_model)
        encoder_output = model.encoder(encoder_input) 
        
        # 2. Run Decoder step-by-step
        # Initialize decoder input with <SOS> token
        decoder_input = torch.full((1, max_seq_len + 1), PAD_TOKEN, dtype=torch.long, device=device)
        decoder_input[:, 0] = SOS_TOKEN # Start with <SOS>
        
        for i in range(max_seq_len):
            # Get the sequence so far (e.g., at i=0, this is just <SOS>)
            current_seq = decoder_input[:, :i+1] 
            
            # Create mask for the current sequence
            pad_mask = (current_seq != PAD_TOKEN).unsqueeze(1).unsqueeze(2)
            future_mask = torch.triu(
                torch.ones(i+1, i+1, device=device), diagonal=1
            ).bool()
            dec_mask = pad_mask & ~future_mask
            
            # Forward pass
            dec_output = model.decoder(current_seq, dec_mask, encoder_output)
            logits = model.final_proj(dec_output) # (1, i+1, vocab_size)
            
            # Get the *last* token's prediction
            next_token = logits.argmax(dim=-1)[:, -1] # (1,)
            decoder_input[:, i+1] = next_token
            
            # Check for completion
            # We check the sequence *after* <SOS>
            if is_expression_complete(decoder_input[0, 1:i+2].tolist()):
                break
                
    # 3. Convert token IDs back to expression
    pred_seq_ids = decoder_input[0].tolist()
    pred_tokens = token_ids_to_words(pred_seq_ids)
    
    # Copy the list for each parser, as they consume the list
    try:
        sympy_pred = build_sympy_from_sequence(list(pred_tokens))
    except Exception:
        sympy_pred = sympy.Symbol("parsing_error")

    try:
        string_pred = build_string_from_sequence(list(pred_tokens))
    except Exception:
        string_pred = "parsing_error"
    
    return sympy_pred, string_pred
