# Developed traditionally with the addition of AI assistance
# Author: Alan Hamm (pqn7)
# Date: April 2024

# Import key functions and classes from submodules to allow easy access at the package level
from .data_io import get_num_records, futures_create_lda_datasets
from .utils import garbage_collection,  exponential_backoff, convert_float32_to_float
from .logging_helper import archive_log, close_logger, setup_logging
from .process_futures import retry_processing, process_completed_futures, handle_failed_future

from .model import train_model
from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta, validate_alpha_beta, calculate_alpha_beta
from .visualization import create_vis_pylda, create_vis_pcoa

from .write_to_postgres import save_to_zip, create_dynamic_table_class, create_table_if_not_exists, add_model_data_to_database

from .yaml_loader import join, getenv, get_current_time


# Define what should be imported with "from my_lda_library import *"
__all__ = [
    # data_io
    'get_num_records',
    'futures_create_lda_datasets',

    # utils
    'garbage_collection',
    'exponential_backoff',
    'convert_float32_to_float',

    #logging_helper
    'archive_log', 
    'close_logger', 
    'setup_logging',

    #yaml_loader
    'join', 
    'getenv', 
    'get_current_time',

    # model
    'train_model',

    # alpha_eta
    'calculate_numeric_alpha',
    'calculate_numeric_beta',
    'validate_alpha_beta',
    'calculate_alpha_beta',

    # visualization
    'create_vis_pylda',
    'create_vis_pcoa',

    # process_futures
    'retry_processing',
    'process_completed_futures',
    'handle_failed_future',

    # writeToPostgres
    'save_to_zip', 
    'create_dynamic_table_class',
    'create_table_if_not_exists', 
    'add_model_data_to_database'
]
