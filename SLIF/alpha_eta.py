# Developed traditionally with the addition of AI assistance
# Author: Alan Hamm (pqn7)
# Date: April 2024

import logging 
from decimal import Decimal
import numpy as np

def calculate_numeric_alpha(alpha_str, num_topics):
    if alpha_str == 'symmetric':
        return Decimal('1.0') / num_topics
    elif alpha_str == 'asymmetric':
        return Decimal('1.0') / (num_topics + Decimal(num_topics).sqrt())
    else:
        # Use Decimal for arbitrary precision
        return Decimal(alpha_str)

def calculate_numeric_beta(beta_str, num_topics):
    if beta_str == 'symmetric':
        return Decimal('1.0') / num_topics
    else:
        # Use Decimal for arbitrary precision
        return Decimal(beta_str)

def validate_alpha_beta(alpha_str, beta_str):
    valid_strings = ['symmetric', 'asymmetric']
    if isinstance(alpha_str, str) and alpha_str not in valid_strings:
        logging.error(f"Invalid alpha_str value: {alpha_str}. Must be 'symmetric', 'asymmetric', or a numeric value.")
        raise ValueError(f"Invalid alpha_str value: {alpha_str}. Must be 'symmetric', 'asymmetric', or a numeric value.")
    if isinstance(beta_str, str) and beta_str not in valid_strings:
        logging.error(f"Invalid beta_str value: {beta_str}. Must be 'symmetric', or a numeric value.")
        raise ValueError(f"Invalid beta_str value: {beta_str}. Must be 'symmetric', or a numeric value.")
    
def calculate_alpha_beta(num_topics):
    """
    Calculate alpha and beta parameter values for Latent Dirichlet Allocation (LDA).
    
    Parameters:
    - num_topics: The number of topics in the LDA model.
    
    Returns:
    - alpha_values: List of alpha parameter values for LDA.
    - beta_values: List of beta parameter values for LDA.
    """
    # Calculate symmetric and asymmetric numeric values for alpha
    numeric_symmetric = 1.0 / num_topics
    numeric_asymmetric = 1.0 / (num_topics + np.sqrt(num_topics))
    
    # Generate alpha values
    alpha_values = ['symmetric', 'asymmetric']
    alpha_values += [numeric_symmetric, numeric_asymmetric] + np.arange(0.01, 1, 0.3).tolist()
    
    # Generate beta values
    beta_values = ['symmetric']
    beta_values += [numeric_symmetric] + np.arange(0.01, 1, 0.3).tolist()
    
    return alpha_values, beta_values