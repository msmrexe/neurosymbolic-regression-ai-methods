"""
This script contains the core functions for generating the synthetic dataset
of mathematical expressions and their corresponding (x, y) data points.

It is designed to be imported and called by 'scripts/generate_transformer_data.py'.
It relies on 'src/transformer/utils.py' for parsing and symbol definitions.
"""

import sympy
import numpy as np
import time
import signal
import os
import logging
from typing import List, Tuple

# Import shared utilities (vocab, symbols, parsers)
from .utils import (
    MY_VOCAB_ARRAY,
    VOCAB_MAP,
    C, x1, x2, x3, x4, x5, x6,
    build_sympy_from_sequence,
    build_sequence_from_sympy,
    sample_points_from_expression,
    count_variables_in_expression,
    expression_tree_depth
)

logger = logging.getLogger(__name__)


def generate_random_expression(vocab: np.ndarray) -> List[str]:
    """
    Recursively generates a random expression as a list of tokens.
    (Based on `generate_expression` from Cell 33).

    Args:
        vocab: The vocabulary array (MY_VOCAB_ARRAY).

    Returns:
        A list of string tokens representing an expression.
    """
    weights = vocab[:, 1].astype('float32')
    probs = weights / np.sum(weights)
    
    rand_idx = np.random.choice(len(vocab), p=probs)
    token = vocab[rand_idx, 0]
    arity = int(vocab[rand_idx, 2])
    
    expr = [token]
    
    if arity == 0:
        return expr
    
    # Create a new vocab for children, preventing nested trig/log functions
    # for simplicity, as in the original notebook.
    new_vocab = vocab
    if token in ['sin', 'cos']:
        # Find indices of sin and cos in the vocab array
        idx1 = np.where(vocab[:, 0]=='sin')[0][0]
        idx2 = np.where(vocab[:, 0]=='cos')[0][0]
        new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
    elif token in ['log', 'exp']:
        # Find indices of log and exp
        idx1 = np.where(vocab[:, 0]=='log')[0][0]
        idx2 = np.where(vocab[:, 0]=='exp')[0][0]
        new_vocab = np.delete(vocab, [idx1, idx2], axis=0)

    for _ in range(arity):
        expr.extend(generate_random_expression(new_vocab))
        
    return expr


def normalize_variable_indices(sympy_expr: sympy.Expr) -> sympy.Expr:
    """
    Counts the number of variables in the expression and re-assigns them
    starting from x1. (Based on `first_variables_first` from Cell 33).
    Example: log(x3) + x5 -> log(x1) + x2
    """
    tokens = [f'x{i}' for i in range(1, 7)]
    sympy_str = str(sympy_expr)
    
    present_vars = []
    for t in tokens:
        if t in sympy_str:
            present_vars.append(t)
    
    # Replace in order: x3 -> x1_temp, x5 -> x2_temp
    for new_idx, old_var_name in enumerate(present_vars, 1):
        sympy_str = sympy_str.replace(old_var_name, f'x{new_idx}_temp')
    
    # Second pass to finalize: x1_temp -> x1, x2_temp -> x2
    for new_idx in range(1, len(present_vars) + 1):
        sympy_str = sympy_str.replace(f'x{new_idx}_temp', f'x{new_idx}')
        
    return sympy.sympify(sympy_str)


def simplify_expression(
    expr_list: List[str], 
    timeout_seconds: int = 5
) -> Tuple[sympy.Expr, bool]:
    """
    Tries to convert a token list to a simplified SymPy expression,
    with a timeout to handle overly complex simplifications.
    (Based on logic from Cell 35).
    
    Returns a tuple of (sympy_expression, is_valid).
    """
    
    class TimeoutException(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutException()

    # Set the signal handler for the alarm
    signal.signal(signal.SIGALRM, handler)
    
    try:
        # 1. Build the initial SymPy expression
        sympy_expr = build_sympy_from_sequence(list(expr_list)) # Use a copy
        
        # 2. Try to simplify with a timeout
        signal.alarm(timeout_seconds)
        try:
            sympy_expr = sympy.factor(sympy_expr)
            sympy_expr = sympy.simplify(sympy_expr)
        except (TimeoutException, RecursionError):
            logger.debug(f"Simplification timed out for {expr_list}")
            return None, False
        finally:
            signal.alarm(0) # Disable alarm

        # 3. Check for invalid expressions
        if 'zoo' in str(sympy_expr) or 'I' in str(sympy_expr): # 'zoo' (complex infinity) or 'I' (imaginary)
            return None, False
            
        return sympy_expr, True
        
    except Exception as e:
        logger.debug(f"Failed to build/simplify sympy expression {expr_list}: {e}")
        return None, False


def process_and_save_datasets(
    config: dict, 
    unique_expressions: List[List[str]]
) -> int:
    """
    Iterates through unique expressions, samples them, and saves valid
    datasets to disk. (Based on main loop in Cell 35).
    """
    
    PATH_OUT = config['PATH_OUT']
    NB_SAMPLING_PER_EQ = config['NB_SAMPLING_PER_EQ']
    ORDER_OF_MAG_LIMIT = config['ORDER_OF_MAG_LIMIT']
    NB_SAMPLE_PTS = config['NB_SAMPLE_PTS']
    NB_ZFILL = config['NB_ZFILL']
    
    # Ensure directories exist
    os.makedirs(f'{PATH_OUT}/values', exist_ok=True)
    os.makedirs(f'{PATH_OUT}/ground_truth', exist_ok=True)

    dataset_count = 0
    aborted_oom = 0
    aborted_nan = 0

    for expr_seq in tqdm(unique_expressions, desc="Sampling Datasets"):
        try:
            for _ in range(NB_SAMPLING_PER_EQ):
                # Use a slightly smaller constant range for better stability
                const_val = np.round(np.random.uniform(low=-5.0, high=5.0), decimals=2)
                
                # Create the version of the sequence with the constant filled in
                sampled_seq = []
                ground_truth_seq = []
                for token in expr_seq:
                    if token == 'C':
                        sampled_seq.append(str(const_val))
                        ground_truth_seq.append(f"C={const_val}")
                    else:
                        sampled_seq.append(token)
                        ground_truth_seq.append(token)
                
                try:
                    sampled_sympy_expr = build_sympy_from_sequence(list(sampled_seq))
                    y_samples, x_samples = sample_points_from_expression(
                        sampled_sympy_expr, 
                        nb_samples=1000 # Oversample to filter NaNs
                    )
                except Exception:
                    continue # Failed to sample this variation

                # --- Filtering ---
                if np.nanmax(np.abs(y_samples)) > ORDER_OF_MAG_LIMIT:
                    aborted_oom += 1
                    continue
                
                valid_mask = ~np.isnan(y_samples)
                if np.sum(valid_mask) < NB_SAMPLE_PTS:
                    aborted_nan += 1
                    continue
                # --- End Filtering ---
                    
                valid_y = y_samples[valid_mask]
                valid_x = x_samples[valid_mask]
                
                # Sub-sample the required number of points
                sample_indices = np.random.choice(len(valid_y), NB_SAMPLE_PTS, replace=False)
                
                final_y = valid_y[sample_indices]
                final_x = valid_x[sample_indices]
                
                num_vars = count_variables_in_expression(sampled_sympy_expr)
                
                # Structure: [y, x1, x2, ..., x6] (total 7 cols)
                dataset = np.zeros((NB_SAMPLE_PTS, 7))
                dataset[:, 0] = final_y
                dataset[:, 1:(num_vars + 1)] = final_x[:, :num_vars]
                
                # --- Save data ---
                dataset_id_str = str(dataset_count).zfill(NB_ZFILL)
                np.save(f'{PATH_OUT}/values/data_{dataset_id_str}.npy', dataset)
                with open(f'{PATH_OUT}/ground_truth/equation_{dataset_id_str}.txt', 'w') as f:
                    f.write('\n'.join(ground_truth_seq))
                    
                dataset_count += 1
        
        except Exception as e:
            logger.warning(f"Error processing base sequence {expr_seq}: {e}")
            
    logger.info(f"Aborted due to magnitude (OOM): {aborted_oom}")
    logger.info(f"Aborted due to insufficient valid points (NaN): {aborted_nan}")
    return dataset_count
