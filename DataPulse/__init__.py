# __init__.py - Initialization for SpectraSync Package
# Author: Alan Hamm
# Date: October 2024

# Description:
# This __init__.py file initializes the Unified Topic Modeling and Analysis (SpectraSync) package.
# It exposes key functions and classes from various submodules to simplify access
# at the package level, providing a cohesive interface for the framework.

# Import essential functions and classes from submodules
from .utils import *
from .process_futures import *
from .topic_model_trainer import train_model_v2
from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta, validate_alpha_beta, calculate_alpha_beta
from .visualization import create_vis_pylda, create_vis_pcoa, process_visualizations, create_vis_pca, create_tsne_plot, get_document_topics
from .write_to_postgres import save_to_zip, create_dynamic_table_class, create_table_if_not_exists, add_model_data_to_database
from .yaml_loader import join, getenv, get_current_time
from .postgres_logging  import  PostgresLoggingHandler
from .mathstats import *
from .batch_estimation import *

# Define __all__ to control what is imported with "from SpectraSync import *"
__all__ = [
    # mathstats
    'init_sample_coherence',
    'calculate_statistics',
    'sample_coherence_for_phase',
    'get_statistics',
    'calculate_perplexity_threshold',
    'coherence_score_decision',
    'calculate_coherence_metrics',
    'calculate_perplexity',
    'compute_full_coherence_score',
    'calculate_convergence',
    'calculate_perplexity_score',
    'calculate_value',
    'cpu_simulate_coherence_scores_with_lln',
    'gpu_simulate_coherence_scores_with_lln',
    'simulate_coherence_scores_with_lln_optimized',

    # batch estimation
    'estimate_futures_batches_large_docs',
    'estimate_futures_batches',
    'estimate_futures_batches_large_docs_v2',
    'estimate_futures_batches_large_optimized',
    'estimate_futures_batches_large_optimized_v2',

    # utils
    'garbage_collection',
    'exponential_backoff',
    'convert_float32_to_float',
    'json_fallback_handler',
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
    'create_tsne_plot',
    'process_visualizations',
    'get_document_topics',
    'get_document_topics_batch',

    # process_futures
    'process_completed_futures',
    'futures_create_lda_datasets',
    'futures_create_lda_datasets_v2',
    'futures_create_lda_datasets_v3',
    'get_and_process_show_topics',
    'extract_topics_with_get_topic_terms',

    # writeToPostgres
    'save_to_zip', 
    'create_dynamic_table_class',
    'create_table_if_not_exists', 
    'add_model_data_to_database',
    
    #postgres_logging
    'PostgresLoggingHandler'
]
