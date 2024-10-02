# writting with pair programming methodology

# Import key functions and classes from submodules to allow easy access at the package level
from .data_io import get_num_records, futures_create_lda_datasets, save_to_zip, add_model_data_to_metadata
from .utils import garbage_collection,  exponential_backoff
from .process_futures import retry_processing, process_completed_futures, handle_failed_future

from .model import train_model
from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta, validate_alpha_beta
from .visualization import create_vis


# Define what should be imported with "from my_lda_library import *"
__all__ = [
    # data_io
    'get_num_records',
    'futures_create_lda_datasets',
    'save_to_zip',
    'add_model_data_to_metadata',

    # utils
    'garbage_collection',
    'exponential_backoff',

    # model
    'train_model',

    # alpha_eta
    'calculate_numeric_alpha',
    'calculate_numeric_beta',
    'validate_alpha_beta',

    # visualization
    'create_vis',

    # process_futures
    'retry_processing',
    'process_completed_futures',
    'handle_failed_future'
]
