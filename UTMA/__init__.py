# __init__.py - Initialization for UTMA Package
# Author: Alan Hamm
# Date: October 2024

# Description:
# This __init__.py file initializes the Unified Topic Modeling and Analysis (UTMA) package.
# It exposes key functions and classes from various submodules to simplify access
# at the package level, providing a cohesive interface for the framework.

# Import essential functions and classes from submodules
from .utils import garbage_collection, exponential_backoff, convert_float32_to_float, get_file_size, download_from_url, process_local_file, clear_temp_files, periodic_cleanup
from .process_futures import process_completed_futures, futures_create_lda_datasets
from .topic_model_trainer import train_model_v2
from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta, validate_alpha_beta, calculate_alpha_beta
from .visualization import create_vis_pylda, create_vis_pcoa, process_visualizations, create_vis_pca
from .write_to_postgres import save_to_zip, create_dynamic_table_class, create_table_if_not_exists, add_model_data_to_database
from .yaml_loader import join, getenv, get_current_time
from .postgres_logging  import  PostgresLoggingHandler
from .mathstats import *
from .batch_estimation import estimate_futures_batches, estimate_futures_batches_large_docs

# Define __all__ to control what is imported with "from UTMA import *"
__all__ = [
    # mathstats
    'sample_coherence',
    'calculate_statistics',
    'sample_coherence_for_phase',
    'get_statistics',
    'calculate_perplexity_threshold',
    'coherence_score_decision',
    'replace_nan_with_interpolated',
    'replace_nan_with_high_precision',
    'calculate_perplexity',
    'compute_full_coherence_score',
    'calculate_convergence',
    'calculate_perplexity_score',

    # batch estimation
    'estimate_futures_batches_large_docs',
    'estimate_futures_batches',

    # utils
    'garbage_collection',
    'exponential_backoff',
    'convert_float32_to_float',
    'get_file_size',
    'download_from_url',
    'process_local_file',
    'clear_temp_files',
    'periodic_cleanup',

    #yaml_loader
    'join', 
    'getenv', 
    'get_current_time',

    #topic_model_trainer
    'train_model_v2',

    # alpha_eta
    'calculate_numeric_alpha',
    'calculate_numeric_beta',
    'validate_alpha_beta',
    'calculate_alpha_beta',

    # visualization
    'create_vis_pylda',
    'create_vis_pcoa',
    'create_vis_pca',
    'process_visualizations',

    # process_futures
    'process_completed_futures',
    'futures_create_lda_datasets',

    # writeToPostgres
    'save_to_zip', 
    'create_dynamic_table_class',
    'create_table_if_not_exists', 
    'add_model_data_to_database',
    
    #postgres_logging
    'PostgresLoggingHandler'
]
