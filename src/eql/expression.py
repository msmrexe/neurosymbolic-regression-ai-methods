"""
Contains functions to retrieve a SymPy symbolic expression from the
trained weights of a SymbolicNet.
"""

import sympy as sp
import numpy as np
from typing import List, Callable, Union
from .functions import BaseFunction, BaseFunction2, count_double, count_inputs

def apply_activation(
    W: sp.Matrix, 
    funcs: List[Callable], 
    n_double: int = 0
) -> sp.Matrix:
    """
    Applies SymPy activation functions to a SymPy matrix.

    Args:
        W: A SymPy matrix, typically (1, n_inputs).
        funcs: A list of callable SymPy functions.
        n_double: The number of 2-input functions.

    Returns:
        A new SymPy matrix with functions applied.
    """
    W = sp.Matrix(W)
    n_single = len(funcs) - n_double
    out_size = n_single + n_double
    
    # Create a new matrix to hold the outputs
    W_new = sp.zeros(W.shape[0], out_size)
    
    input_idx = 0
    for i in range(W.shape[0]):
        # Apply single-input functions
        for j_out in range(n_single):
            W_new[i, j_out] = funcs[j_out](W[i, input_idx])
            input_idx += 1
            
        # Apply double-input functions
        for j_out in range(n_single, out_size):
            W_new[i, j_out] = funcs[j_out](W[i, input_idx], W[i, input_idx + 1])
            input_idx += 2
            
    return W_new


def sym_pp(
    W_list: List[np.ndarray], 
    funcs: List[Callable], 
    var_names: List[str], 
    threshold: float, 
    n_double: int
) -> sp.Matrix:
    """
    Pretty-print (generate) the symbolic expression for the hidden layers.
    """
    # Create SymPy symbols for variables
    vars = [sp.Symbol(var) for var in var_names]
    expr = sp.Matrix(vars).T  # Shape (1, n_vars)

    for W in W_list:
        W_sp = sp.Matrix(W)
        W_filtered = filter_mat(W_sp, threshold=threshold)
        expr = expr * W_filtered
        expr = apply_activation(expr, funcs, n_double=n_double)
    return expr


def last_pp(eq: sp.Matrix, W: np.ndarray, threshold: float) -> sp.Matrix:
    """Pretty-print (apply) the last layer."""
    W_sp = sp.Matrix(W)
    W_filtered = filter_mat(W_sp, threshold=threshold)
    return eq * W_filtered


def get_expression(
    weights: List[np.ndarray], 
    funcs_list: List[Union[BaseFunction, BaseFunction2]], 
    var_names: List[str], 
    threshold: float = 0.01
) -> sp.Expr:
    """
    Generates the full SymPy expression from a list of weight matrices.

    Args:
        weights: List of weight matrices (as NumPy arrays) from SymbolicNet.
        funcs_list: List of function objects (e.g., [Identity(), Sin()]).
        var_names: List of input variable names (e.g., ["x_1", "x_2"]).
        threshold: Weight threshold for pruning weak connections.

    Returns:
        A simplified SymPy expression.
    """
    n_double = count_double(funcs_list)
    # Get the SymPy implementation of each function
    funcs_sp = [func.sp for func in funcs_list]
    
    hidden_weights = weights[:-1]
    output_weight = weights[-1]

    # Process hidden layers
    expr = sym_pp(
        hidden_weights, 
        funcs_sp, 
        var_names, 
        threshold=threshold, 
        n_double=n_double
    )
    
    # Process output layer
    expr = last_pp(expr, output_weight, threshold=threshold)
    
    # Simplify and return
    final_expr = expr[0, 0]
    return sp.simplify(final_expr)


def filter_mat(mat: sp.Matrix, threshold: float = 0.01) -> sp.Matrix:
    """Zeros out elements of a SymPy matrix below a threshold."""
    mat_copy = mat.copy()
    for i in range(mat_copy.shape[0]):
        for j in range(mat_copy.shape[1]):
            if abs(mat_copy[i, j]) < threshold:
                mat_copy[i, j] = 0
    return mat_copy

# Note: filter_expr and filter_expr2 from the notebook are complex and
# often unnecessary if sp.simplify() is used. `get_expression` now relies
# on sp.simplify() for a cleaner final expression.
