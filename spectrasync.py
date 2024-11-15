# spectrasync.py - SpectraSync: Neural Intelligence Meets Multi-Dimensional Topic Analysis
# Author: Alan Hamm
# Date: November 2024
#
# Description:
# SpectraSync is an advanced topic analysis engine, crafted for high-dimensional, large-scale document analysis.
# Fueled by Dask and Gensim, this script channels neural-inspired insights to map the evolving spectrum of topics within your corpus.
# SpectraSync’s precision algorithms uncover thematic flows, persisting topics, and semantic shifts within text—across time, genres, and realms.
#
# Usage:
# Run SpectraSync from the command line, passing your parameters to unlock new dimensions in topic modeling.
# Example: python spectrasync.py --input_dir=<path> --output_dir=<path>
#
# Dependencies:
# - PostgreSQL for robust database operations
# - Python libraries: Dask, Gensim, SQLAlchemy, and supporting libraries for SpectraSync’s multi-threaded infrastructure
#
# Developed with AI assistance and designed for those who push the boundaries of data exploration.

#%%
from SpectraSync import *

import argparse

from gensim.corpora import Dictionary
from dask.distributed import get_client
from dask.distributed import Client, LocalCluster, performance_report, wait
from distributed import Future
import dask
import threading
import socket
import tornado
import re

import yaml # used for logging configuration file
import logging
import logging.config
import logging.handlers
import gzip
from shutil import move

import sqlalchemy

import sys
import os

from datetime import datetime

from tqdm import tqdm
from time import time, sleep

import random
import math
import numpy as np
import pandas as pd

import itertools

import hashlib

import pickle

# Dask dashboard throws deprecation warnings w.r.t. Bokeh
import warnings
from bokeh.util.deprecation import BokehDeprecationWarning
from tornado.iostream import StreamClosedError

# pyLDAvis throws errors when using jc_PCoA(instead use MMD5)
from numpy import ComplexWarning



###################################
# BEGIN SCRIPT CONFIGURATION HERE #
###################################
def task_callback(future):
    if future.status == 'error':
        print(f"Task failed with exception: {future.exception()}")
    else:
        print("Task completed successfully")

def parse_args():
    """Parse command-line arguments for configuring the topic analysis script."""
    parser = argparse.ArgumentParser(description="Configure the topic analysis script using command-line arguments.")
    
    # Database Connection Arguments
    parser.add_argument("--username", type=str, help="Username for accessing the PostgreSQL database.")
    parser.add_argument("--password", type=str, help="Password for the specified PostgreSQL username.")
    parser.add_argument("--host", type=str, help="Hostname or IP address of the PostgreSQL server (e.g., 'localhost' or '192.168.1.1').")
    parser.add_argument("--port", type=int, help="Port number for the PostgreSQL server (default is 5432).")
    parser.add_argument("--database", type=str, help="Name of the PostgreSQL database to connect to.")
    
    # Corpus and Data Arguments
    parser.add_argument("--corpus_label", type=str, help="Unique label used to identify the corpus in outputs and logs. Must be suitable as a PostgreSQL table name.")
    parser.add_argument("--data_source", type=str, help="File path to the JSON file containing the data for analysis.")
    parser.add_argument("--train_ratio", type=float, help="Fraction of data to use for training (e.g., 0.8 for 80% training and 20% testing).")
    parser.add_argument("--validation_ratio", type=float, help="Fraction of data to use for validation.")

    # Topic Modeling Parameters
    parser.add_argument("--start_topics", type=int, help="Starting number of topics for evaluation.")
    parser.add_argument("--end_topics", type=int, help="Ending number of topics for evaluation.")
    parser.add_argument("--step_size", type=int, help="Incremental step size for increasing the topic count between start_topics and end_topics.")

    # System Resource Management
    parser.add_argument("--num_workers", type=int, help="Minimum number of CPU cores to utilize for parallel processing.")
    parser.add_argument("--max_workers", type=int, help="Maximum number of CPU cores allocated for parallel processing.")
    parser.add_argument("--num_threads", type=float, help="Maximum number of threads per core for efficient use of resources.")
    parser.add_argument("--max_memory", type=int, help="Maximum RAM (in GB) allowed per core for processing.")
    parser.add_argument("--mem_threshold", type=int, help="Memory usage threshold (in GB) to trigger data spill to disk.")
    parser.add_argument("--max_cpu", type=float, help="Maximum CPU utilization percentage to prevent overuse of resources.")
    parser.add_argument("--mem_spill", type=str, help="Directory for temporarily storing data when memory limits are exceeded.")

    # Gensim Model Settings
    parser.add_argument("--passes", type=int, help="Number of complete passes through the data for the Gensim topic model.")
    parser.add_argument("--iterations", type=int, help="Total number of iterations for the model to converge.")
    parser.add_argument("--update_every", type=int, help="Frequency (in number of documents) to update model parameters during training.")
    parser.add_argument("--eval_every", type=int, help="Frequency (in iterations) for evaluating model perplexity and logging progress.")
    parser.add_argument("--random_state", type=int, help="Seed value to ensure reproducibility of results.")
    parser.add_argument("--per_word_topics", type=bool, help="Whether to compute per-word topic probabilities (True/False).")

    # Batch Processing Parameters
    parser.add_argument("--futures_batches", type=int, help="Number of batches to process concurrently.")
    parser.add_argument("--base_batch_size", type=int, help="Initial number of documents processed in parallel in each batch.")
    parser.add_argument("--max_batch_size", type=int, help="Maximum batch size, representing the upper limit of documents processed in parallel.")
    parser.add_argument("--increase_factor", type=float, help="Percentage increase in batch size after successful processing.")
    parser.add_argument("--decrease_factor", type=float, help="Percentage decrease in batch size after failed processing.")
    parser.add_argument("--max_retries", type=int, help="Maximum attempts to retry failed batch processing.")
    parser.add_argument("--base_wait_time", type=float, help="Initial wait time in seconds for exponential backoff during retries.")

    # Directories and Logging
    parser.add_argument("--log_dir", type=str, help="Directory path for saving log files.")
    parser.add_argument("--root_dir", type=str, help="Root directory for saving project outputs, metadata, and temporary files.")

    args = parser.parse_args()

    # Validate corpus_label against PostgreSQL table naming conventions
    if args.corpus_label:
        if not re.match(r'^[a-z][a-z0-9_]{0,62}$', args.corpus_label):
            error_msg = "Invalid corpus_label: must start with a lowercase letter, can only contain lowercase letters, numbers, and underscores, and be up to 63 characters long."
            logging.error(error_msg)
            print(error_msg)
            sys.exit(1)

    return args

# Parse CLI arguments
args = parse_args()

# Define required arguments with corresponding error messages
required_args = {
    "username": "No value was entered for username",
    "password": "No value was entered for password",
    "database": "No value was entered for database",
    "corpus_label": "No value was entered for corpus_label",
    "data_source": "No value was entered for data_source",
    "end_topics": "No value was entered for end_topics",
    "step_size": "No value was entered for step_size",
    "max_memory": "No value was entered for max_memory",
    "mem_threshold": "No value was entered for mem_threshold",
    "futures_batches": "No value was entered for futures_batches",
}

# Check for required arguments and log error if missing
for arg, error_msg in required_args.items():
    if getattr(args, arg) is None:
        logging.error(error_msg)
        print(error_msg)
        sys.exit(1)

# Load and define parameters based on arguments or apply defaults
USERNAME = args.username
PASSWORD = args.password
HOST = args.host if args.host is not None else "localhost"
PORT = args.port if args.port is not None else 5432
DATABASE = args.database
CONNECTION_STRING = f"postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

CORPUS_LABEL = args.corpus_label
DATA_SOURCE = args.data_source

TRAIN_RATIO = args.train_ratio if args.train_ratio is not None else 0.70
VALIDATION_RATIO = args.validation_ratio if args.validation_ratio is not None else 0.15

START_TOPICS = args.start_topics if args.start_topics is not None else 1
END_TOPICS = args.end_topics
STEP_SIZE = args.step_size

CORES = args.num_workers if args.num_workers is not None else 1
MAXIMUM_CORES = args.max_workers if args.max_workers is not None else 1
THREADS_PER_CORE = args.num_threads if args.num_threads is not None else 1
# Convert max_memory to a string with "GB" suffix for compatibility with Dask LocalCluster() object
RAM_MEMORY_LIMIT = f"{args.max_memory}GB" if args.max_memory is not None else "4GB"  # Default to "4GB" if not provided
MEMORY_UTILIZATION_THRESHOLD = (args.mem_threshold * (1024 ** 3)) if args.mem_threshold else 4 * (1024 ** 3)
CPU_UTILIZATION_THRESHOLD = args.max_cpu if args.max_cpu is not None else 120
DASK_DIR = args.mem_spill if args.mem_spill else os.path.expanduser("~/temp/SpectraSync/max_spill")
os.makedirs(DASK_DIR, exist_ok=True)

# Model configurations
PASSES = args.passes if args.passes is not None else 15
ITERATIONS = args.iterations if args.iterations is not None else 100
UPDATE_EVERY = args.update_every if args.update_every is not None else 5
EVAL_EVERY = args.eval_every if args.eval_every is not None else 5
RANDOM_STATE = args.random_state if args.random_state is not None else 50
PER_WORD_TOPICS = args.per_word_topics if args.per_word_topics is not None else True

# Batch configurations
FUTURES_BATCH_SIZE = args.futures_batches # number of input docuemtns to read in batches
BATCH_SIZE = args.base_batch_size if args.base_batch_size is not None else FUTURES_BATCH_SIZE # number of documents used in each iteration of creating/training/saving 
MAX_BATCH_SIZE = args.max_batch_size if args.max_batch_size is not None else FUTURES_BATCH_SIZE * 10 # the maximum number of documents(ie batches) assigned depending upon sys performance
MIN_BATCH_SIZE = max(1, math.ceil(MAX_BATCH_SIZE * .10)) # the fewest number of docs(ie batches) to be processed if system is under stress

# Batch size adjustments and retry logic
INCREASE_FACTOR = args.increase_factor if args.increase_factor is not None else 1.05
DECREASE_FACTOR = args.decrease_factor if args.decrease_factor is not None else 0.10
MAX_RETRIES = args.max_retries if args.max_retries is not None else 5
BASE_WAIT_TIME = args.base_wait_time if args.base_wait_time is not None else 1.1

# Ensure required directories exist
ROOT_DIR = args.root_dir or os.path.expanduser("~/temp/SpectraSync/")
LOG_DIRECTORY = args.log_dir or os.path.join(ROOT_DIR, "log")
IMAGE_DIR = os.path.join(ROOT_DIR, "visuals")
PYLDA_DIR = os.path.join(IMAGE_DIR, 'pyLDAvis')
PCOA_DIR = os.path.join(IMAGE_DIR, 'PCoA')
#METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
TEXTS_ZIP_DIR = os.path.join(ROOT_DIR, "texts_zip")

for directory in [ROOT_DIR, LOG_DIRECTORY, IMAGE_DIR, PYLDA_DIR, PCOA_DIR, TEXTS_ZIP_DIR]: # METADATA_DIR
    os.makedirs(directory, exist_ok=True)

# Set JOBLIB_TEMP_FOLDER based on ROOT_DIR and CORPUS_LABEL
#JOBLIB_TEMP_FOLDER = os.path.join(ROOT_DIR, "log", "joblib") if CORPUS_LABEL else os.path.join(ROOT_DIR, "log", "joblib")
#os.makedirs(JOBLIB_TEMP_FOLDER, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = DASK_DIR


###############################
###############################
# DO NOT EDIT BELOW THIS LINE #
###############################
###############################

# to escape: distributed.nanny - WARNING - Worker process still alive after 4.0 seconds, killing
# https://github.com/dask/dask-jobqueue/issues/391
scheduler_options={"host":socket.gethostname()}

# Ensure the LOG_DIRECTORY exists
if args.log_dir: LOG_DIRECTORY = args.log_dir
if args.root_dir: ROOT_DIR = args.root_dir
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Define the top-level directory and subdirectories
LOG_DIR = os.path.join(ROOT_DIR, "log")
IMAGE_DIR = os.path.join(ROOT_DIR, "visuals")
PYLDA_DIR = os.path.join(IMAGE_DIR, 'pyLDAvis')
PCA_GPU_DIR = os.path.join(IMAGE_DIR, 'PCA_GPU')
PCOA_DIR = os.path.join(IMAGE_DIR, 'PCoA')
#METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
TEXTS_ZIP_DIR = os.path.join(ROOT_DIR, "texts_zip")

# Ensure that all necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PYLDA_DIR, exist_ok=True)
os.makedirs(PCOA_DIR, exist_ok=True)
os.makedirs(PCA_GPU_DIR, exist_ok=True)
#os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TEXTS_ZIP_DIR, exist_ok=True)

# Format the date and time as per your requirement
# Note: %w is the day of the week as a decimal (0=Sunday, 6=Saturday)
#       %Y is the four-digit year
#       %m is the two-digit month (01-12)
#       %H%M is the hour (00-23) followed by minute (00-59) in 24hr format
# Check if the environment variable is already set
if 'LOG_START_TIME' not in os.environ:
    os.environ['LOG_START_TIME'] = datetime.now().strftime('%w-%m-%Y-%H%M')

# Use the fixed timestamp from the environment variable
#log_filename = f"log-{os.environ['LOG_START_TIME']}.log"
log_filename = f"log-{os.environ['LOG_START_TIME']}.gz"
LOGFILE = os.path.join(LOG_DIR, log_filename)  # Directly join log_filename with LOG_DIRECTORY

# Database connection parameters
db_params = {
    'dbname': 'text_mining',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': 5432
}

# Custom filter to add extra field to log records
class CustomFieldFilter( logging.Filter):
    def filter(self, record):
        record.custom_field = f"{CORPUS_LABEL}"  # Replace with your actual value
        return True

# Initialize the PostgreSQL log handler with a formatter
postgres_handler = PostgresLoggingHandler(db_params)
formatter = logging.Formatter(' %(custom_field)s - %(asctime)s - %(levelname)s - %(message)s')
postgres_handler.setFormatter(formatter)

# Add the custom filter to the handler
custom_filter = CustomFieldFilter()
postgres_handler.addFilter(custom_filter)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[postgres_handler],
    format='%(custom_field)s - %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


##########################################
# Filter out the specific warning message
##########################################
# suppress RuntimeWarning: overflow encountered in exp2 globally across the entire script <- caused by ldamodel.py
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp2")

# Lib\site-packages\gensim\topic_coherence\direct_confirmation_measure.py:204
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

# Suppress ComplexWarnings generated in create_vis() function with pyLDAvis, note: this 
# is caused by using js_PCoA in the prepare() method call. Intsead of js_PCoA, MMDS is 
# implemented.
warnings.simplefilter('ignore', ComplexWarning)

# Get the logger for 'distributed' package
distributed_logger = logging.getLogger('distributed')

# Disable Bokeh deprecation warnings
logging.getLogger("bokeh").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
# Set the logging level for distributed.utils_perf to suppress warnings
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="distributed.utils_perf")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="distributed.worker")  # Adjust the module parameter as needed

# Suppress specific SettingWithCopyWarning from pyLDAvis internals
# line 299: A value is trying to be set on a copy of a slice from a DataFrame.
#   Try using .loc[row_indexer,col_indexer] = value instead
# line 300: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead
warnings.filterwarnings("ignore", category=Warning, module=r"pyLDAvis\._prepare")


# Suppress StreamClosedError warnings from Tornado
# \Lib\site-packages\distributed\comm\tcp.py", line 225, in read
#   frames_nosplit_nbytes_bin = await stream.read_bytes(fmt_size)
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class StreamClosedWarning(Warning):
    pass
warnings.filterwarnings("ignore", category=StreamClosedWarning)
logging.getLogger('tornado').setLevel(logging.ERROR)

# Get the logger for 'sqlalchemy.engine' which is used by SQLAlchemy to log SQL queries
sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')

# Remove all handlers associated with 'sqlalchemy.engine' (this includes StreamHandler)
for handler in sqlalchemy_logger.handlers[:]:
    sqlalchemy_logger.removeHandler(handler)

# Add a NullHandler to prevent default StreamHandler from being added later on
null_handler = logging.NullHandler()
sqlalchemy_logger.addHandler(null_handler)

# Optionally set a higher level if you want to ignore INFO logs from sqlalchemy.engine
# sqlalchemy_logger.setLevel(logging.WARNING)

# Suppress overflow warning from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp2")

# Enable serialization optimizations 
dask.config.set(scheduler='distributed', serialize=True)
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.scheduler.worker-ttl": '30m'})
dask.config.set({'distributed.worker.daemon': False})

#These settings disable automatic spilling but allow for pausing work when 80% of memory is consumed and terminating workers at 99%.
dask.config.set({'distributed.worker.memory.target': False,
                 'distributed.worker.memory.spill': False,
                 'distributed.worker.memory.pause': 0.8
                 ,'distributed.worker.memory.terminate': 0.99})



# https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
if __name__=="__main__":
    # Capture the start time
    start_time = pd.Timestamp.now()
    formatted_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"START TIME OF PROCESSING THE CORPUS: {formatted_start_time}")

    # Multiprocessing (processes=True): This mode creates multiple separate Python processes, 
    # each with its own Python interpreter and memory space. Since each process has its own GIL, 
    # they can execute CPU-bound tasks in true parallel on multiple cores without being affected 
    # by the GIL. This typically results in better utilization of multi-core CPUs for compute-intensive tasks.
    
    #Multithreading (processes=False): This mode uses threads within a single Python process for 
    # concurrent execution. While this is efficient for I/O-bound tasks due to low overhead in 
    # switching between threads, it's less effective for CPU-bound tasks because of Python's Global 
    # Interpreter Lock (GIL). The GIL prevents multiple native threads from executing Python bytecodes 
    # at once, which can limit the performance gains from multithreading when running CPU-intensive workloads.
    cluster = LocalCluster(
            n_workers=CORES,
            threads_per_worker=THREADS_PER_CORE,
            processes=True,
            memory_limit=RAM_MEMORY_LIMIT,
            local_directory=DASK_DIR,
            #dashboard_address=None,
            dashboard_address=":8787",
            protocol="tcp",
            death_timeout='1000s',  # Increase timeout before forced kill
    )


    # Create the distributed client
    client = Client(cluster, timeout='1000s')

    # set for adaptive scaling
    client.cluster.adapt(minimum=CORES, maximum=MAXIMUM_CORES)
    
    # Get information about workers from scheduler
    workers_info = client.scheduler_info()["workers"]

    # Iterate over workers and set their memory limits
    for worker_id, worker_info in workers_info.items():
        worker_info["memory_limit"] = RAM_MEMORY_LIMIT

    # Check if the Dask client is connected to a scheduler:
    if client.status == "running":
        logging.info("Dask client is connected to a scheduler.")
        # Scatter the embedding vectors across Dask workers
    else:
        logging.error("Dask client is not connected to a scheduler.")
        logging.error("The system is shutting down.")
        client.close()
        cluster.close()
        sys.exit()

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
        logging.info(f"{CORES} Dask workers are running.")
    else:
        logging.error("No Dask workers are running.")
        logging.error("The system is shutting down.")
        client.close()
        cluster.close()
        sys.exit()


    print("Creating training and evaluation samples...")

    started = time()
    
    scattered_train_data_futures = []
    scattered_validation_data_futures = []
    scattered_test_data_futures = []
    all_futures = []
    
    # Process each batch as it is generated
    #for batch_info in futures_create_lda_datasets(DATA_SOURCE, TRAIN_RATIO, VALIDATION_RATIO, FUTURES_BATCH_SIZE):
    for batch_info in futures_create_lda_datasets_v2(DATA_SOURCE):
        if batch_info['type'] == "dictionary":
            # Retrieve the dictionary
            unified_dictionary = batch_info['data']

        elif batch_info['type'] == "train":
            # Handle training data
            #print("We are inside the IF/ELSE block for producing TRAIN scatter.")
            try:
                scattered_future = client.scatter(batch_info['data'])
                #scattered_future.add_done_callback(task_callback)
                # After yielding each batch
                #print(f"Submitted {batch_info['type']} batch of size {len(batch_info['data'])} to Dask.")

                scattered_train_data_futures.append(scattered_future)
                
            except Exception as e:
                logging.error(f"There was an issue with creating the TRAIN scattered_future list: {e}")

        elif batch_info['type'] == 'validation':
            # Handle validation data
            try:
                scattered_future = client.scatter(batch_info['data'])
                #scattered_future.add_done_callback(task_callback)
                # After yielding each batch
                #print(f"Submitted {batch_info['type']} batch of size {len(batch_info['data'])} to Dask.")

                scattered_validation_data_futures.append(scattered_future)
            except Exception as e:
                logging.error(f"There was an issue with creating the VALIDATION scattererd_future list: {e}")

        elif batch_info['type'] == 'test':
            # Handle test data
            try:
                scattered_future = client.scatter(batch_info['data'])
                #scattered_future.add_done_callback(task_callback)
                # After yielding each batch
                #print(f"Submitted {batch_info['type']} batch of size {len(batch_info['data'])} to Dask.")

                scattered_test_data_futures.append(scattered_future)
            except Exception as e:
                logging.error(f"There was an issue with creating the TEST scattererd_future list: {e}")
        else:
            logging.error("There are documents not being scattered across the workers.")
        
    #print(f"Completed creation of train-validation-test split in {round((time() - started)/60,2)} minutes.\n")
    logging.info(f"Completed creation of train-validation-test split in {round((time() - started)/60,2)} minutes.\n")
    #print("Document scatter across workers complete...\n")
    logging.info("Document scatter across workers complete...")
    print(f"\nFinal count - Number of training batches: {len(scattered_train_data_futures)}, "
      f"Number of validation batches: {len(scattered_validation_data_futures)}, "
      f"Number of test batches: {len(scattered_test_data_futures)}\n")

    train_futures = []  # List to store futures for training
    validation_futures = []  # List to store futures for validation
    test_futures = []  # List to store futures for testing
   
    num_topics = len(range(START_TOPICS, END_TOPICS + 1, STEP_SIZE))

    #####################################################
    # PROCESS AND CONVERT ALPHA AND ETA PARAMETER VALUES
    #####################################################
    # Calculate numeric_alpha for symmetric prior
    numeric_symmetric = 1.0 / num_topics
    # Calculate numeric_alpha for asymmetric prior (using best judgment)
    numeric_asymmetric = 1.0 / (num_topics + np.sqrt(num_topics))
    # Create the list with numeric values
    numeric_alpha = [numeric_symmetric, numeric_asymmetric] + np.arange(0.01, 1, 0.3).tolist()
    numeric_beta = [numeric_symmetric] + np.arange(0.01, 1, 0.3).tolist()

    # The parameter `alpha` in Latent Dirichlet Allocation (LDA) represents the concentration parameter of the Dirichlet 
    # prior distribution for the topic-document distribution.
    # It controls the sparsity of the resulting document-topic distributions.
    # A lower value of `alpha` leads to sparser distributions, meaning that each document is likely to be associated
    # with fewer topics. Conversely, a higher value of `alpha` encourages documents to be associated with more
    # topics, resulting in denser distributions.

    # The choice of `alpha` affects the balance between topic diversity and document specificity in LDA modeling.
    alpha_values = ['symmetric', 'asymmetric']
    alpha_values += np.arange(0.01, 1, 0.3).tolist()

    # In Latent Dirichlet Allocation (LDA) topic analysis, the beta parameter represents the concentration 
    # parameter of the Dirichlet distribution used to model the topic-word distribution. It controls the 
    # sparsity of topics by influencing how likely a given word is to be assigned to a particular topic.
    # A higher value of beta encourages topics to have a more uniform distribution over words, resulting in more 
    # general and diverse topics. Conversely, a lower value of beta promotes sparser topics with fewer dominant words.

    # The choice of beta can impact the interpretability and granularity of the discovered topics in LDA.
    beta_values = ['symmetric']
    beta_values += np.arange(0.01, 1, 0.3).tolist()

    #################################################
    # CREATE PARAMETER COMBINATIONS FOR GRID SEARCH
    #################################################
    # Create a list of all combinations of n_topics, alpha_value, beta_value, and train_eval
    phases = ["train", "validation", "test"]
    combinations = list(itertools.product(range(START_TOPICS, END_TOPICS + 1, STEP_SIZE), alpha_values, beta_values, phases))

    # Define sample size for overall combinations if needed
    sample_fraction = 0.375
    # Ensure that `sample_size` doesn’t exceed the number of total combinations
    sample_size = min(max(1, int(len(combinations) * sample_fraction)), len(combinations))

    # Generate random combinations
    random_combinations = random.sample(combinations, sample_size)

    # Determine undrawn combinations
    undrawn_combinations = list(set(combinations) - set(random_combinations))

    # Randomly sample from the entire set of combinations
    random_combinations = random.sample(combinations, sample_size)

    # Determine undrawn combinations
    undrawn_combinations = list(set(combinations) - set(random_combinations))

    print(f"The random sample combinations contain {len(random_combinations)}. This leaves {len(undrawn_combinations)} undrawn combinations.\n")
    #for record in random_combinations:
    #    print("This is the random combination", record)
    for i, item in enumerate(random_combinations):
        if not isinstance(item, tuple) or len(item) != 4:
            print(f"Issue at index {i}: {item}")

    # Create empty lists to store all future objects for training and evaluation
    train_futures = []
    validation_futures = []
    test_futures = []

    # Start the cleanup in a background thread
    cleanup_thread = threading.Thread(target=periodic_cleanup, args=(DASK_DIR,), daemon=True) # 30 minute intervals. see utils script
    cleanup_thread.start()
    
    TOTAL_COMBINATIONS = len(random_combinations) * (len(scattered_train_data_futures) + len(scattered_validation_data_futures) + len(scattered_test_data_futures))
    progress_bar = tqdm(total=TOTAL_COMBINATIONS, desc="Creating and saving models", file=sys.stdout)

    # Custom sort order: train first, then validation, then test
    phase_order = {"train": 0, "validation": 1, "test": 2}
    sorted_combinations = sorted(
        random_combinations,
        key=lambda x: phase_order[x[3]]
    )
    # Initialize combined visualization lists outside the loop
    completed_pylda_vis, completed_pcoa_vis, completed_pca_gpu_vis = [], [], []
    train_models_dict, validation_models_dict, test_models_dict = {}, {}, {}
    completed_train_futures, completed_validation_futures, completed_test_futures = [], [], []

    # Create a unified dictionary for each (n_topics, alpha_value, beta_value) combination
    #computed_train_data  = client.gather(scattered_train_data_futures)
    #unified_train_data = [doc for scattered_data in computed_train_data for doc in scattered_data]
    #unified_dictionary = Dictionary(unified_train_data)  # Using the complete training dataset
    
    # Process sorted combinations by train, validation, and test phases
    for i, (n_topics, alpha_value, beta_value, train_eval_type) in enumerate(sorted_combinations):

        # Adaptive throttling logic remains here
        logging.info("Evaluating if adaptive throttling is necessary...")
        throttle_attempt = 0
        while throttle_attempt < MAX_RETRIES:
            scheduler_info = client.scheduler_info()
            all_workers_below_cpu_threshold = all(
                worker['metrics']['cpu'] < CPU_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values()
            )
            all_workers_below_memory_threshold = all(
                worker['metrics']['memory'] < MEMORY_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values()
            )
            if not (all_workers_below_cpu_threshold and all_workers_below_memory_threshold):
                logging.warning(f"Adaptive throttling (attempt {throttle_attempt + 1} of {MAX_RETRIES})")
                sleep(exponential_backoff(throttle_attempt, BASE_WAIT_TIME=BASE_WAIT_TIME))
                throttle_attempt += 1
            else:
                break
        if throttle_attempt == MAX_RETRIES:
            logging.warning("Maximum retries reached; proceeding despite resource usage.")

        num_workers = len(client.scheduler_info()["workers"])

        # Train Phase
        if train_eval_type == "train":
            try:
                # Train phase logic
                for scattered_data in scattered_train_data_futures:
                    model_key = (n_topics, alpha_value, beta_value)
                    future = client.submit(
                        train_model_v2, DATA_SOURCE, n_topics, alpha_value, beta_value, TEXTS_ZIP_DIR, PYLDA_DIR, PCOA_DIR, PCA_GPU_DIR, unified_dictionary, scattered_data, "train",
                        RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, num_workers, PER_WORD_TOPICS
                    )
                    train_futures.append(future)
                    progress_bar.update()
            except Exception as e:
                logging.error("Train phase error in SpectraSync.py with train_model_v2")
            try:
                # Wait for all training futures and then process results
                done_train, _ = wait(train_futures, timeout=None)
                completed_train_futures = [done.result() for done in done_train]
                if len(completed_train_futures) == 0:
                    logging.error("No results were output from '_v2' for writing to SSD. Number of completed train futures: {len(completed_train_futures)}")
            except Exception as e:
                logging.error(f"Train phase error with WAIT: {e}")
                print(f"Train phase error with WAIT: {e}")
                sys.exit()
            try:
                # Gather model results and visualize
                for train_result in completed_train_futures:
                    model_key = (train_result['topics'], str(train_result['alpha_str'][0]), str(train_result['beta_str'][0]))
                    train_models_dict[model_key] = train_result['lda_model']

                    # Compute delayed objects before using them
                    #for key, value in train_result.items():
                    #    if isinstance(value, dask.delayed.Delayed):
                    #        logging.debug(f"train_result['{key}'] is a Delayed object and needs to be computed.")
                    #sys.exit()

                    # Compute delayed objects before using them
                    #lda_model = pickle.loads(train_result['lda_model']).compute() if isinstance(train_result['lda_model'], dask.delayed.Delayed) else train_result['lda_model']
                    #corpus = pickle.loads(train_result['corpus']).compute() if isinstance(train_result['corpus'], dask.delayed.Delayed) else train_result['corpus']
                    #dictionary = pickle.loads(train_result['dictionary']).compute() if isinstance(train_result['dictionary'], dask.delayed.Delayed) else train_result['dictionary']
                    
                    try:
                        # Visualization tasks
                        train_pcoa_vis = create_vis_pca(
                            pickle.loads(train_result['lda_model']),
                            pickle.loads(train_result['corpus']),
                            n_topics, "TRAIN", train_result['text_md5'],
                            train_result['time_key'], PCOA_DIR
                        )
                    except Exception as e:
                        logging.error(f"Visualization/SpectaSync/Train Phase/create_vis_pca: {e}")
                    
                    try:
                        train_pca_gpu_vis = create_pca_plot_gpu(
                            train_result['validation_result'], 
                            train_result['topics_words'],
                            train_result['perplexity_threshold'],
                                        "TRAIN",
                                        train_result['num_word'], 
                                        n_topics, train_result['text_md5'],
                                        train_result['time_key'], PCA_GPU_DIR
                        )
                    except Exception as e:
                        logging.error(f"Visualization/SpectaSync/Train Phase/create_pca_plot_gpu: {e}")

                    try:
                        train_pylda_vis = create_vis_pylda(
                            pickle.loads(train_result['lda_model']),
                            pickle.loads(train_result['corpus']),
                            pickle.loads(train_result['dictionary']),
                            n_topics, "TRAIN", train_result['text_md5'], CORES,
                            train_result['time_key'], PYLDA_DIR
                        )
                    except Exception as e:
                        logging.error(f"Visualization/SpectaSync/Train Phase/create_vis_pylda: {e}")

                    # Compute visualization results
                    #completed_pylda_vis.append(train_pylda_vis.compute())
                    #completed_pcoa_vis.append(train_pcoa_vis.compute())
                    completed_pylda_vis.append(train_pylda_vis)
                    completed_pcoa_vis.append(train_pcoa_vis)
                    completed_pca_gpu_vis.append(train_pca_gpu_vis)
            except Exception as e:
                logging.error(f"Error in visualization train phase: {e}")
                sys.exit()
                continue

            # After processing all train phases in the sorted combinations
            try:
                if completed_train_futures or completed_validation_futures or completed_test_futures:
                    process_completed_futures("TRAIN",
                        CONNECTION_STRING, CORPUS_LABEL,
                        completed_train_futures, completed_validation_futures, completed_test_futures,
                        len(completed_train_futures),
                        num_workers, BATCH_SIZE, TEXTS_ZIP_DIR, 
                        vis_pylda=completed_pylda_vis, vis_pcoa=completed_pcoa_vis, vis_pca=completed_pca_gpu_vis
                    )
            except Exception as e:
                logging.error(f"Error processing TRAIN completed futures: {e}")
                print(f"Error processing TRAIN completed futures: {e}")
                sys.exit()

        # Validation Phase
        try:
            for scattered_data in scattered_validation_data_futures:
                model_key = (n_topics, alpha_value, beta_value)
                ldamodel = pickle.loads(train_models_dict[model_key])
                future = client.submit(
                    train_model_v2, DATA_SOURCE, n_topics, alpha_value, beta_value, TEXTS_ZIP_DIR, PYLDA_DIR, PCOA_DIR, PCA_GPU_DIR, unified_dictionary, scattered_data, "validation",
                    RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, num_workers, PER_WORD_TOPICS, ldamodel=ldamodel
                )
                validation_futures.append(future)
                progress_bar.update()

                # Wait for validation futures and update progress
                done_validation, _ = wait(validation_futures, timeout=None)
                completed_validation_futures = [done.result() for done in done_validation]
                
                # Gather model results and visualize
                for validation_result in completed_validation_futures:
                    model_key = (validation_result['topics'], str(validation_result['alpha_str'][0]), str(validation_result['beta_str'][0]))
                    validation_models_dict[model_key] = validation_result['lda_model']

                    # Visualization tasks
                    validation_pcoa_vis = create_vis_pca(
                        pickle.loads(validation_result['lda_model']),
                        pickle.loads(validation_result['corpus']),
                        n_topics, "TRAIN", validation_result['text_md5'],
                        validation_result['time_key'], PCOA_DIR
                    )
                    validation_pca_gpu_vis = create_pca_plot_gpu(
                        pickle.loads(validation_result['validation_result']), 
                        pickle.loads(validation_result['topic_labels']),
                                     "VALIDATION",
                                     validation_result['num_words'], 
                                     n_topics, validation_result['text_md5'],
                                     validation_result['time_key'], PCA_GPU_DIR,
                                     title="PCA GPU Topic Distribution"
                    )
                    validation_pylda_vis = create_vis_pylda(
                        pickle.loads(validation_result['lda_model']),
                        pickle.loads(validation_result['corpus']),
                        pickle.loads(validation_result['dictionary']),
                        n_topics, "TRAIN", validation_result['text_md5'], CORES,
                        validation_result['time_key'], PYLDA_DIR
                    )

                    # Compute visualization results
                    completed_pylda_vis.append(validation_pylda_vis.compute())
                    completed_pcoa_vis.append(validation_pcoa_vis.compute())
                    completed_pca_gpu_vis.append(validation_pca_gpu_vis.compute())
        except Exception as e:
            logging.error(f"Error in validation phase: {e}")
            continue

        # After processing all train phases in the sorted combinations
        try:
            if completed_train_futures or completed_validation_futures or completed_test_futures:
                process_completed_futures("VALIDATION",
                        CONNECTION_STRING, CORPUS_LABEL,
                        completed_train_futures, completed_validation_futures, completed_test_futures,
                        len(completed_validation_futures),
                        num_workers, BATCH_SIZE, TEXTS_ZIP_DIR, vis_pylda=completed_pylda_vis, vis_pcoa=completed_pcoa_vis
                )
        except Exception as e:
            logging.error(f"Error processing VALIDATION completed futures: {e}")

        # Test Phase
        try:
            for scattered_data in scattered_test_data_futures:
                model_key = (n_topics, alpha_value, beta_value)
                ldamodel = pickle.loads(test_models_dict[model_key])
                future = client.submit(
                    train_model_v2, DATA_SOURCE, n_topics, alpha_value, beta_value, TEXTS_ZIP_DIR, PYLDA_DIR, PCOA_DIR, PCA_GPU_DIR, unified_dictionary, scattered_data, "test",
                    RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, num_workers, PER_WORD_TOPICS, ldamodel=ldamodel
                )
                test_futures.append(future)
                progress_bar.update()

            # Wait for test futures and update progress
            done_test, _ = wait(test_futures, timeout=None)
            completed_test_futures = [done.result() for done in done_test]

            # Gather model results and visualize
            for test_resut in completed_test_futures:
                model_key = (test_resut['topics'], str(test_resut['alpha_str'][0]), str(test_resut['beta_str'][0]))
                test_models_dict[model_key] = test_resut['lda_model']

                # Visualization tasks
                test_pcoa_vis = create_vis_pca(
                    pickle.loads(test_resut['lda_model']),
                    pickle.loads(test_resut['corpus']),
                    n_topics, "TRAIN", test_resut['text_md5'],
                    test_resut['time_key'], PCOA_DIR
                )
                test_pca_gpu_vis = create_pca_plot_gpu(
                    pickle.loads(test_resut['validation_result']), 
                    pickle.loads(test_resut['topic_labels']),
                    "VALIDATION",
                    test_resut['num_words'], 
                    n_topics, test_resut['text_md5'],
                    test_resut['time_key'], PCA_GPU_DIR,
                    title="PCA GPU Topic Distribution"
                )
                test_pylda_vis = create_vis_pylda(
                    pickle.loads(test_resut['lda_model']),
                    pickle.loads(test_resut['corpus']),
                    pickle.loads(test_resut['dictionary']),
                    n_topics, "TRAIN", test_resut['text_md5'], CORES,
                    test_resut['time_key'], PYLDA_DIR
                )

                # Compute visualization results
                completed_pylda_vis.append(test_pylda_vis.compute())
                completed_pcoa_vis.append(test_pcoa_vis.compute())
                completed_pcoa_vis.append(test_pca_gpu_vis.compute())
        except Exception as e:
            logging.error(f"Error in test phase: {e}")
            continue

        # After processing all train phases in the sorted combinations
        try:
            if completed_train_futures or completed_validation_futures or completed_test_futures:
                    process_completed_futures("TEST",
                        CONNECTION_STRING, CORPUS_LABEL,
                        completed_train_futures, completed_validation_futures, completed_test_futures,
                        len(completed_test_futures),
                        num_workers, BATCH_SIZE, TEXTS_ZIP_DIR, vis_pylda=completed_pylda_vis, vis_pcoa=completed_pcoa_vis
                    )
        except Exception as e:
            logging.error(f"Error processing TEST completed futures: {e}")

        # Log the processing time
        elapsed_time = round(((time() - started) / 60), 2)
        logging.info(f"Finished processing futures to disk in {elapsed_time} minutes")
    
        completed_train_futures.clear()
        completed_validation_futures.clear()
        completed_test_futures.clear()
        test_futures.clear()
        validation_futures.clear()
        train_futures.clear()
        client.rebalance()

    # Capture the end time
    end_time = pd.Timestamp.now()
    formatted_end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"END TIME OF PROCESSING THE CORPUS: {formatted_end_time}")

    # Calculate and log the time difference
    time_difference = end_time - start_time
    days = time_difference.days
    hours, remainder = divmod(time_difference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logging.info(f"TOTAL PROCESSING TIME: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"TOTAL PROCESSING TIME: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
    progress_bar.close()        
    client.close()
    cluster.close()