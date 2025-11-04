"""
Main script to run the offline data generation for the Transformer model.

Example Usage:
python scripts/generate_transformer_data.py --nb_trails 10000 --nb_sample_pts 50
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import sympy
from tqdm.auto import tqdm

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    from src.utils import setup_logging
    from src.transformer.data_generator import (
        generate_random_expression, 
        simplify_expression, 
        normalize_variable_indices,
        process_and_save_datasets
    )
    from src.transformer.utils import (
        MY_VOCAB_ARRAY,
        build_sequence_from_sympy,
        count_variables_in_expression,
        expression_tree_depth
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def parse_args():
    """Parses command-line arguments for data generation."""
    parser = argparse.ArgumentParser(description="Generate Transformer Seq2Seq Data")
    
    # Pull ALL defaults from the config.py file
    parser.add_argument("--nb_trails", type=int, default=config.TRANSFORMER_GEN_NB_TRAILS)
    parser.add_argument("--nb_nodes_min", type=int, default=config.TRANSFORMER_GEN_NB_NODES_MIN)
    parser.add_argument("--nb_nodes_max", type=int, default=config.TRANSFORMER_GEN_NB_NODES_MAX)
    parser.add_argument("--timeout_simplify", type=int, default=config.TRANSFORMER_GEN_TIMEOUT_SIMPLIFY)
    parser.add_argument("--nb_nested_max", type=int, default=config.TRANSFORMER_GEN_NB_NESTED_MAX)
    parser.add_argument("--nb_constants_min", type=int, default=config.TRANSFORMER_GEN_NB_CONSTANTS_MIN)
    parser.add_argument("--nb_constants_max", type=int, default=config.TRANSFORMER_GEN_NB_CONSTANTS_MAX)
    parser.add_argument("--nb_variables_max", type=int, default=config.TRANSFORMER_GEN_NB_VARIABLES_MAX)
    parser.add_argument("--seq_length_max", type=int, default=config.TRANSFORMER_GEN_SEQ_LENGTH_MAX)
    parser.add_argument("--nb_sampling_per_eq", type=int, default=config.TRANSFORMER_GEN_NB_SAMPLING_PER_EQ)
    parser.add_argument("--order_of_mag_limit", type=float, default=config.TRANSFORMER_GEN_ORDER_OF_MAG_LIMIT)
    parser.add_argument("--nb_sample_pts", type=int, default=config.TRANSFORMER_GEN_NB_SAMPLE_PTS)
    parser.add_argument("--nb_zfill", type=int, default=config.TRANSFORMER_GEN_NB_ZFILL)
    parser.add_argument(
        "--path_out",
        type=str,
        default=config.TRANSFORMER_DATA_PATH,
        help="Directory to save the generated data."
    )
    
    return parser.parse_args()

def main():
    """Main execution function for data generation."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Transformer data generation...")
    logger.info(f"Arguments: {vars(args)}")

    start_time = time.time()
    
    # --- Part 1: Generate many random expressions ---
    logger.info(f"Generating {args.nb_trails} raw expression trees...")
    all_my_expr = [
        generate_random_expression(MY_VOCAB_ARRAY) 
        for _ in tqdm(range(args.nb_trails), desc="Generating")
    ]
    logger.info(f"Generated {len(all_my_expr)} expressions.")

    # --- Part 2: Filter by length ---
    my_expr_filter = [
        expr for expr in all_my_expr 
        if args.nb_nodes_min <= len(expr) <= args.nb_nodes_max
    ]
    logger.info(f"Filtered by length: {len(my_expr_filter)} remaining.")

    # --- Part 3: Simplify and filter by validity/depth ---
    logger.info(f"Simplifying and filtering by depth (max {args.nb_nested_max})...")
    my_expr_sympy = []
    nb_timeout_abort = 0
    
    for expr_list in tqdm(my_expr_filter, desc="Simplifying"):
        sympy_expr, valid = simplify_expression(expr_list, args.timeout_simplify)
        
        if not valid:
            # If sympy_expr is None, it means it was a timeout
            if sympy_expr is None and not valid: 
                nb_timeout_abort += 1
            continue
            
        if expression_tree_depth(sympy_expr) <= args.nb_nested_max:
            sympy_expr = normalize_variable_indices(sympy_expr)
            if 'x1' in str(sympy_expr): # Must contain at least one variable
                my_expr_sympy.append(sympy_expr)
    
    logger.info(f"Simplified: {len(my_expr_sympy)} remaining.")
    logger.info(f"Aborted due to timeout: {nb_timeout_abort}")

    # --- Part 4: Convert back to sequence and filter ---
    logger.info("Converting to standardized sequences and filtering...")
    my_expr_seq = []
    abort_stats = {
        'pow': 0, 'const_min': 0, 'const_max': 0, 'var_max': 0, 'seq_len': 0
    }
    
    for sympy_expr in tqdm(my_expr_sympy, desc="Standardizing"):
        expr_seq = build_sequence_from_sympy(sympy_expr)
        
        if 'abort' in expr_seq:
            abort_stats['pow'] += 1
        elif expr_seq.count('C') > args.nb_constants_max:
            abort_stats['const_max'] += 1
        elif expr_seq.count('C') < args.nb_constants_min:
            abort_stats['const_min'] += 1
        elif count_variables_in_expression(sympy_expr) > args.nb_variables_max:
            abort_stats['var_max'] += 1
        elif len(expr_seq) > args.seq_length_max:
            abort_stats['seq_len'] += 1
        else:
            my_expr_seq.append(expr_seq)
            
    logger.info(f"Filtered sequences: {len(my_expr_seq)} remaining.")
    logger.info(f"Abort stats: {abort_stats}")

    # --- Part 5: Find unique expressions ---
    str_sequences = [str(seq) for seq in my_expr_seq]
    _, unique_indices = np.unique(str_sequences, return_index=True)
    my_expr_uniq_seq = [my_expr_seq[i] for i in unique_indices]
    logger.info(f"Found {len(my_expr_uniq_seq)} unique expressions.")

    # --- Part 6: Sample and save datasets ---
    logger.info(f"Sampling and saving datasets to {args.path_out}...")
    
    # Create the config dictionary to pass to the data generator
    # It's populated from our script's arguments
    config_dict = {
        "PATH_OUT": args.path_out,
        "NB_SAMPLING_PER_EQ": args.nb_sampling_per_eq,
        "ORDER_OF_MAG_LIMIT": args.order_of_mag_limit,
        "NB_SAMPLE_PTS": args.nb_sample_pts,
        "NB_ZFILL": args.nb_zfill
    }
    
    count = process_and_save_datasets(config_dict, my_expr_uniq_seq)
    
    end_time = time.time()
    logger.info(f"Data generation complete. Created {count} datasets.")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
