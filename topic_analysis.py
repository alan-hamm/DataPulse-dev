# Developed traditionally with the addition of AI assistance
# Author: Alan Hamm (pqn7)
# Date: April 2024

#%%
from SLIF import *

import argparse

from dask.distributed import Client, LocalCluster, performance_report, wait
from distributed import Future
import dask
import socket
import tornado
import re

import yaml # used for logging configuration file
import logging
import logging.config
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

# pyLDAvis throws errors when using jc_PCoA(instead use MMD5)
from numpy import ComplexWarning

#import multiprocessing

###################################
# BEGIN SCRIPT CONFIGURATION HERE #
###################################

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

    # Topic Modeling Parameters
    parser.add_argument("--start_topics", type=int, help="Starting number of topics for evaluation.")
    parser.add_argument("--end_topics", type=int, help="Ending number of topics for evaluation.")
    parser.add_argument("--step_size", type=int, help="Incremental step size for increasing the topic count between start_topics and end_topics.")

    # System Resource Management
    parser.add_argument("--num_workers", type=int, help="Minimum number of CPU cores to utilize for parallel processing.")
    parser.add_argument("--max_workers", type=int, help="Maximum number of CPU cores allocated for parallel processing.")
    parser.add_argument("--num_threads", type=int, help="Maximum number of threads per core for efficient use of resources.")
    parser.add_argument("--max_memory", type=int, help="Maximum RAM (in GB) allowed per core for processing.")
    parser.add_argument("--mem_threshold", type=int, help="Memory usage threshold (in GB) to trigger data spill to disk.")
    parser.add_argument("--max_cpu", type=int, help="Maximum CPU utilization percentage to prevent overuse of resources.")
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
    parser.add_argument("--base_wait_time", type=int, help="Initial wait time in seconds for exponential backoff during retries.")

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
TRAIN_RATIO = args.train_ratio if args.train_ratio is not None else 0.80
START_TOPICS = args.start_topics if args.start_topics is not None else 1
END_TOPICS = args.end_topics
STEP_SIZE = args.step_size
CORES = args.num_workers if args.num_workers is not None else 1
MAXIMUM_CORES = args.max_workers if args.max_workers is not None else 1
THREADS_PER_CORE = args.num_threads if args.num_threads is not None else 1
# Convert max_memory to a string with "GB" suffix for compatibility with Dask LocalCluster() object
RAM_MEMORY_LIMIT = f"{args.max_memory}GB" if args.max_memory is not None else "4GB"  # Default to "4GB" if not provided
MEMORY_UTILIZATION_THRESHOLD = (args.mem_threshold * (1024 ** 3)) if args.mem_threshold else 4 * (1024 ** 3)
CPU_UTILIZATION_THRESHOLD = args.max_cpu if args.max_cpu is not None else 100
DASK_DIR = args.mem_spill if args.mem_spill else os.path.expanduser("~/temp/slif/max_spill")
os.makedirs(DASK_DIR, exist_ok=True)

# Model configurations
PASSES = args.passes if args.passes is not None else 15
ITERATIONS = args.iterations if args.iterations is not None else 100
UPDATE_EVERY = args.update_every if args.update_every is not None else 5
EVAL_EVERY = args.eval_every if args.eval_every is not None else 5
RANDOM_STATE = args.random_state if args.random_state is not None else 50
PER_WORD_TOPICS = args.per_word_topics if args.per_word_topics is not None else True

# Batch configurations
FUTURES_BATCH_SIZE = args.futures_batches
BATCH_SIZE = args.base_batch_size if args.base_batch_size is not None else FUTURES_BATCH_SIZE
MAX_BATCH_SIZE = args.max_batch_size if args.max_batch_size is not None else FUTURES_BATCH_SIZE * 10
MIN_BATCH_SIZE = math.ceil(FUTURES_BATCH_SIZE * 1.01)

# Batch size adjustments and retry logic
INCREASE_FACTOR = args.increase_factor if args.increase_factor is not None else 1.05
DECREASE_FACTOR = args.decrease_factor if args.decrease_factor is not None else 0.10
MAX_RETRIES = args.max_retries if args.max_retries is not None else 5
BASE_WAIT_TIME = args.base_wait_time if args.base_wait_time is not None else 30

# Ensure required directories exist
ROOT_DIR = args.root_dir or os.path.expanduser("~/temp/slif/")
LOG_DIRECTORY = args.log_dir or os.path.join(ROOT_DIR, "log")
IMAGE_DIR = os.path.join(ROOT_DIR, "visuals")
PYLDA_DIR = os.path.join(IMAGE_DIR, 'pyLDAvis')
PCOA_DIR = os.path.join(IMAGE_DIR, 'PCoA')
METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
TEXTS_ZIP_DIR = os.path.join(ROOT_DIR, "texts_zip")

for directory in [ROOT_DIR, LOG_DIRECTORY, IMAGE_DIR, PYLDA_DIR, PCOA_DIR, METADATA_DIR, TEXTS_ZIP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set JOBLIB_TEMP_FOLDER based on ROOT_DIR and CORPUS_LABEL
JOBLIB_TEMP_FOLDER = os.path.join(ROOT_DIR, "log", "joblib") if CORPUS_LABEL else os.path.join(ROOT_DIR, "log", "joblib")
os.makedirs(JOBLIB_TEMP_FOLDER, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP_FOLDER


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


#print("Script started successfully.")
#sys.exit(0)

# Define the top-level directory and subdirectories
LOG_DIR = os.path.join(ROOT_DIR, "log")
IMAGE_DIR = os.path.join(ROOT_DIR, "visuals")
PYLDA_DIR = os.path.join(IMAGE_DIR, 'pyLDAvis')
PCOA_DIR = os.path.join(IMAGE_DIR, 'PCoA')
METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
TEXTS_ZIP_DIR = os.path.join(ROOT_DIR, "texts_zip")

# Ensure that all necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PYLDA_DIR, exist_ok=True)
os.makedirs(PCOA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TEXTS_ZIP_DIR, exist_ok=True)

# Redirect stderr to the file
#sys.stderr = open(f"{LOG_DIR}/stderr.log", "w")

# Get the current date and time for log filename
#now = datetime.now()

# Format the date and time as per your requirement
# Note: %w is the day of the week as a decimal (0=Sunday, 6=Saturday)
#       %Y is the four-digit year
#       %m is the two-digit month (01-12)
#       %H%M is the hour (00-23) followed by minute (00-59) in 24hr format
#log_filename = now.strftime('log-%w-%m-%Y-%H%M.log')
#log_filename = 'log-0250.log'
# Check if the environment variable is already set
if 'LOG_START_TIME' not in os.environ:
    os.environ['LOG_START_TIME'] = datetime.now().strftime('%w-%m-%Y-%H%M')

# Use the fixed timestamp from the environment variable
log_filename = f"log-{os.environ['LOG_START_TIME']}.log"
LOG_FILENAME = os.path.join(LOG_DIRECTORY, log_filename)
LOGFILE = os.path.join(LOG_DIRECTORY,LOG_FILENAME)

# Configure logging to write to a file with this name
logging.basicConfig(
    filename=LOGFILE,
    filemode='a',  # Append mode if you want to keep adding to the same file during the day
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

##########################################
# Filter out the specific warning message
##########################################
# Suppress ComplexWarnings generated in create_vis() function with pyLDAvis, note: this 
# is caused by using js_PCoA in the prepare() method call. Intsead of js_PCoA, MMDS is 
# implemented.
warnings.simplefilter('ignore', ComplexWarning)

# Get the logger for 'distributed' package
distributed_logger = logging.getLogger('distributed')

# Disable Bokeh deprecation warnings
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
#tornado.iostream.StreamClosedError: Stream is closed
#warnings.filterwarnings("ignore", category=tornado.iostream.StreamClosedError)
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

# Archive log only once if running in main process
#if multiprocessing.current_process().name == 'MainProcess':
#archive_log(logger, LOGFILE, LOG_DIR)

# Enable serialization optimizations 
dask.config.set(scheduler='distributed', serialize=True) #note: could this be causing the pyLDAvis creation problems??
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.scheduler.worker-ttl": None})
dask.config.set({'distributed.worker.daemon': False})

#These settings disable automatic spilling but allow for pausing work when 80% of memory is consumed and terminating workers at 99%.
dask.config.set({'distributed.worker.memory.target': False,
                 'distributed.worker.memory.spill': False,
                 'distributed.worker.memory.pause': 0.8
                 ,'distributed.worker.memory.terminate': 0.99})



# https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
if __name__=="__main__":
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
            death_timeout='300s',  # Increase timeout before forced kill
    )


    # Create the distributed client
    client = Client(cluster)

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
    scattered_eval_data_futures = []

    # Process each batch as it is generated
    for batch_info in futures_create_lda_datasets(DATA_SOURCE, TRAIN_RATIO, FUTURES_BATCH_SIZE):
        #print(f"Received batch: {batch_info['type']}")  # Debugging output
        if batch_info['type'] == 'train':
            # Handle training data
            #print("We are inside the IF/ELSE block for producing TRAIN scatter.")
            try:
                scattered_future = client.scatter(batch_info['data'])
                scattered_train_data_futures.append(scattered_future)
                #print(f"Appended to train futures: {len(scattered_train_data_futures)}") # Debugging output
            except Exception as e:
                logging.error(f"There was an issue with creating the TRAIN scattered_future list: {e}")
                
            #if whole_train_dataset is None:
            #    whole_train_dataset = batch_info['whole_dataset']
        elif batch_info['type'] == 'eval':
            # Handle evaluation data
            #print("We are inside the IF/ELSE block for producing EVAL scatter.")
            try:
                scattered_future = client.scatter(batch_info['data'])
                scattered_eval_data_futures.append(scattered_future)
                #print(f"Appended to eval futures: {len(scattered_eval_data_futures)}")  # Debugging output
            except Exception as e:
                logging.error(f"There was an issue with creating the EVAL scattererd_future list: {e}")

                
            #if whole_eval_dataset is None:
            #    whole_eval_dataset = batch_info['whole_dataset']
        else:
            print("There are documents not being scattered across the workers.")

    print(f"Completed creation of training and evaluation documents in {round((time() - started)/60,2)} minutes.\n")
    print("Data scatter complete...\n")


    train_futures = []  # List to store futures for training
    eval_futures = []  # List to store futures for evaluation
   
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
    combinations = list(itertools.product(range(START_TOPICS, END_TOPICS + 1, STEP_SIZE), alpha_values, beta_values, ['eval', 'train']))

    # Separate the combinations into two lists based on 'train' and 'eval' for debugging
    train_combinations = [combo for combo in combinations if combo[-1] == 'train']
    eval_combinations = [combo for combo in combinations if combo[-1] == 'eval']

    # Calculate the sample size for each category
    sample_size = min(len(train_combinations), len(eval_combinations))

    # Select random combinations from each category
    random_train_combinations = random.sample(train_combinations, sample_size)
    random_eval_combinations = random.sample(eval_combinations, sample_size)

    # Combine the randomly selected train and eval combinations
    random_combinations = random_eval_combinations+ random_train_combinations
    sample_size = max(1, int(len(combinations) * 0.375))

    # Select random_combinations conditionally
    random_combinations = random.sample(combinations, sample_size) if sample_size < len(combinations) else combinations

    # Determine which combinations were not drawn by using set difference
    undrawn_combinations = list(set(combinations) - set(random_combinations))

    print(f"The random sample combinations contains {len(random_combinations)}. This leaves {len(undrawn_combinations)} undrawn combinations.\n")

    # clear utility vars
    del train_combinations, eval_combinations, random_train_combinations, random_eval_combinations


    # Create empty lists to store all future objects for training and evaluation
    train_futures = []
    eval_futures = []

    # List to store parameters of models that failed to complete even after a retry
    failed_model_params = []
    # Mapping from futures to their corresponding parameters (n_topics, alpha_value, beta_value)
    future_to_params = {}

    TOTAL_COMBINATIONS = len(random_combinations) * (len(scattered_train_data_futures) + len(scattered_eval_data_futures) )
    progress_bar = tqdm(total=TOTAL_COMBINATIONS, desc="Creating and saving models", file=sys.stdout)
    # Iterate over the combinations and submit tasks
    num_iter = len(random_combinations)
    for n_topics, alpha_value, beta_value, train_eval_type in random_combinations:
        #print(f"this is the number of for loop iterations: {num_iter}")
        num_iter-=1
        #print(f"We are on iteration number: {num_iter}")
        # determine if throttling is needed
        logging.info("Evaluating if adaptive throttling is necessary (method exponential backoff)...")
        started, throttle_attempt = time(), 0

        # https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
        while throttle_attempt < MAX_RETRIES:
            scheduler_info = client.scheduler_info()
            all_workers_below_cpu_threshold = all(worker['metrics']['cpu'] < CPU_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values())
            all_workers_below_memory_threshold = all(worker['metrics']['memory'] < MEMORY_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values())

            if not (all_workers_below_cpu_threshold and all_workers_below_memory_threshold):
                logging.info(f"Adaptive throttling (attempt {throttle_attempt} of {MAX_RETRIES-1})")
                # Uncomment the next line if you want to log hyperparameters information as well.
                #logging.info(f"for LdaModel hyperparameters combination -- type: {train_eval_type}, topic: {n_topics}, ALPHA: {alpha_value} and ETA {beta_value}")
                sleep(exponential_backoff(throttle_attempt, BASE_WAIT_TIME=BASE_WAIT_TIME))
                throttle_attempt += 1
            else:
                break

        if throttle_attempt == MAX_RETRIES:
            logging.error("Maximum retries reached. The workers are still above the CPU or Memory threshold.")
            #garbage_collection(False, 'Max Retries - throttling attempt')
        else:
            logging.info("Proceeding with workload as workers are below the CPU and Memory thresholds.")

        # Submit a future for each scattered data object in the training list
        num_workers = len(client.scheduler_info()["workers"])
        for scattered_data in scattered_train_data_futures:
            try:
                future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'train',
                                            RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, num_workers, PER_WORD_TOPICS)
                train_futures.append(future)
                logging.info(f"The training value is being appended to the train_futures list. Size: {len(train_futures)}")
                pass
            except Exception as e:
                logging.error("An error occurred in train_model() Dask operation")
                logging.error(f"TYPE: train -- n_topics: {n_topics}, alpha: {alpha_value}, beta: {beta_value}")

        # Submit a future for each scattered data object in the evaluation list
        #if train_eval_type == 'eval':
        for scattered_data in scattered_eval_data_futures:
            try:
                future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'eval',
                                        RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, num_workers, PER_WORD_TOPICS)
                eval_futures.append(future)
                logging.info(f"The evaluation value is being appended to the eval_futures list. Size: {len(eval_futures)}")
                pass
            except Exception as e:
                logging.error("An error occurred in train_model() Dask operation")
                logging.error(f"TYPE: eval -- n_topics: {n_topics}, alpha: {alpha_value}, beta: {beta_value}")              
        #garbage_collection(False, 'client.submit(train_model(...) train and eval)')


        # Check if it's time to process futures based on BATCH_SIZE
        if (len(train_futures) + len(eval_futures)) >= BATCH_SIZE or num_iter == 0:
            time_of_vis_call = pd.to_datetime('now')
            time_of_vis_call = time_of_vis_call.strftime('%Y%m%d%H%M%S%f')
            PERFORMANCE_TRAIN_LOG = os.path.join(LOG_DIR, f"train_perf_{time_of_vis_call}.html")
            del time_of_vis_call

            with performance_report(filename=PERFORMANCE_TRAIN_LOG):
                logging.info("In holding pattern until WAIT completes.")
                started = time()

                # Wait for completion of eval_futures
                try:
                    done_eval, not_done_eval = wait(eval_futures, timeout=None)  # return_when='FIRST_COMPLETED'
                    logging.info(f"This is the size of the done_eval list: {len(done_eval)} and this is the size of the not_done_eval list: {len(not_done_eval)}")
                except Exception as e:
                    distributed_logger.exception("EVAL_FUTURES wait:  An error occurred while gathering results from train_model() Dask operation")

                try:
                    # Wait for completion of train_futures
                    done_train, not_done_train = wait(train_futures, timeout=None)  # return_when='FIRST_COMPLETED'
                    logging.info(f"This is the size of the done_train list: {len(done_train)} and this is the size of the not_done_train list: {len(not_done_train)}")
                except Exception as e:
                    distributed_logger.exception("TRAIN_FUTURES wait:  An error occurred while gathering results from train_model() Dask operation")


                done = done_train.union(done_eval)
                not_done = not_done_eval.union(not_done_train)
                    
                elapsed_time = round(((time() - started) / 60), 2)
                logging.info(f"WAIT completed in {elapsed_time} minutes")


                # Now clear references to these completed futures by filtering them out of your lists
                train_futures = [f for f in train_futures if f not in done_train]
                eval_futures = [f for f in eval_futures if f not in done_eval]
                
                completed_train_futures = [f for f in done_train]
                completed_eval_futures = [f for f in done_eval]

            # Handle failed futures using the previously defined function
            for future in not_done:
                failed_future_timer = time()
                logging.error("Handling of failed WAIT method has been initiated.")
                handle_failed_future(future, future_to_params, train_futures,  eval_futures, client)
                elapsed_time = round(((time() - started) / 60), 2)
                logging.error(f"It took {elapsed_time} minutes to handle {len(train_futures)} train futures and {len(eval_futures)} evaluation futures the failed future.")

        
            ########################
            # PROCESS VISUALIZATIONS
            ########################
            time_of_vis_call = pd.to_datetime('now')
            time_of_vis_call = time_of_vis_call.strftime('%Y%m%d%H%M%S%f')
            PERFORMANCE_TRAIN_LOG = os.path.join(IMAGE_DIR, f"vis_perf_{time_of_vis_call}.html")
            del time_of_vis_call
            with performance_report(filename=PERFORMANCE_TRAIN_LOG):
                logging.info("In holding pattern until process TRAIN and EVAL visualizations completes.")
                started = time()
                # To get the results from the completed futures
                logging.info("Gathering TRAIN and EVAL futures.")
                results_train_len = len(done_train)
                results_eval_len = len(done_eval)
                results_train = [d.result() for d in done_train if isinstance(d, Future)]
                results_eval = [d.result() for d in done_eval if isinstance(d, Future)]
                if results_train_len != len(done_train):
                    missing = results_train_len - len(done_train)
                    missing = abs(missing)
                    logging.error(f"There are {missing} removed TRAIN visualizations.")
                if results_eval_len != len(done_eval):
                    missing = results_eval_len - len(done_eval)
                    missing = abs(missing)
                    logging.error(f"There are {missing} removed TRAIN visualizations.")
                
                # results_train and results_eval are lists of lists of dictionaries
                for result_list in results_train:
                    for result_dict in result_list:
                        result_dict['type'] = 'train'

                for result_list in results_eval:
                    for result_dict in result_list:
                        result_dict['type'] = 'eval'

                # combine results with additional key 'type'
                results = [result_dict for sublist in (results_train + results_eval) for result_dict in sublist]

                logging.info(f"Completed TRAIN and EVAL gathering {len(results)} futures.")
                
                # Now you can process these results and submit new tasks based on them
                visualization_futures_pylda = []
                visualization_futures_pcoa = []
                
                processed_results = set()  # Use a set to track processed result hashes
                
                for result_dict in results:
                    unique_id = result_dict['time_key']
                    
                    if unique_id not in processed_results:
                        processed_results.add(unique_id)
                            
                        try:
                            vis_future_pylda = client.submit(create_vis_pylda,
                                                            result_dict['lda_model'],
                                                            result_dict['corpus'],
                                                            result_dict['dictionary'],
                                                            result_dict['topics'],
                                                            unique_id,  # filename
                                                            CORES,  
                                                            result_dict['text_md5'], # vis_root
                                                            PYLDA_DIR)
                            visualization_futures_pylda.append(vis_future_pylda)
                        except Exception as e:
                                    logging.error(f"An error occurred in create_vis_pylda() Dask operation: {e}")
                                    logging.error(f"TYPE: pyLDA -- MD5: {result_dict['text_md5']}")

                        try:
                            vis_future_pcoa = client.submit(create_vis_pcoa,
                                                            result_dict['lda_model'],
                                                            result_dict['corpus'],
                                                            result_dict['topics'], # set f'number_of_topics-{topics}'
                                                            unique_id, # filename
                                                            result_dict['text_md5'], #vis_root
                                                            PCOA_DIR)
                            visualization_futures_pcoa.append(vis_future_pcoa)
                        except Exception as e:
                                    logging.error(f"An error occurred in create_vis_pcoa() Dask operation: {e}")
                                    logging.error(f"TYPE: PCoA -- MD5{result_dict['text_md5']}")

                logging.info(f"Executing WAIT on TRAIN and EVAL pyLDA create_visualizations {len(visualization_futures_pylda)} futures.")
                logging.info(f"Executing WAIT on TRAIN and EVAL PCoA create_visualizations {len(visualization_futures_pcoa)} futures.")

                # Wait for all visualization tasks to complete
                done_viz_futures_pylda, not_done_viz_futures_pylda = wait(visualization_futures_pylda)
                done_viz_futures_pcoa, not_done_viz_futures_pcoa = wait(visualization_futures_pcoa)
                #done_visualizations, not_done_visualizations = wait(visualization_futures_pylda + visualization_futures_pcoa)
                if len(not_done_viz_futures_pylda) > 0:
                    logging.error(f"All TRAIN and EVAL pyLDA visualizations couldn't be generated. There were {len(not_done_viz_futures_pylda)} not created.")
                if len(not_done_viz_futures_pcoa) > 0:
                    logging.error(f"All TRAIN and EVAL PCoA visualizations couldn't be generated. There were {len(not_done_viz_futures_pcoa)} not created.")

                # Gather the results from the completed visualization tasks
                logging.info("Gathering TRAIN and EVAL completed visualization results futures.")
                completed_pylda_vis = [future.result() for future in done_viz_futures_pylda]
                completed_pcoa_vis = [future.result() for future in done_viz_futures_pcoa]

                #del completed_visualization_results
                logging.info(f"Completed gathering {len(completed_pylda_vis)+len(completed_pcoa_vis)} TRAIN and EVAL visualization results futures.")

                elapsed_time = round(((time() - started) / 60), 2)
                logging.info(f"Create visualizations for TRAIN and EVAL data completed in {elapsed_time} minutes")
                #print(f"Create visualizations for TRAIN and EVAL data completed in {elapsed_time} minutes")
            # close performance report encapsulation of visualization performance analysis
            #############################
            # END PROCESS VISUALIZATIONS
            #############################            
            
            started = time()
            num_workers = len(client.scheduler_info()["workers"])
            logging.info(f"Writing processed completed futures to disk.")
            completed_eval_futures, completed_train_futures = process_completed_futures(CONNECTION_STRING, \
                                                                                    CORPUS_LABEL, \
                                                                                    completed_train_futures, \
                                                                                        completed_eval_futures, \
                                                                                        (len(completed_eval_futures)+len(completed_train_futures)), \
                                                                                        num_workers, \
                                                                                        BATCH_SIZE, \
                                                                                        TEXTS_ZIP_DIR, \
                                                                                        METADATA_DIR, \
                                                                                        vis_pylda=completed_pylda_vis,
                                                                                        vis_pcoa=completed_pcoa_vis)
            
            elapsed_time = round(((time() - started) / 60), 2)
            logging.info(f"Finished write processed completed futures to disk in  {elapsed_time} minutes")

            progress_bar.update(len(done))
            
            # monitor system resource usage and adjust batch size accordingly
            scheduler_info = client.scheduler_info()
            all_workers_below_cpu_threshold = all(worker['metrics']['cpu'] < CPU_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values())
            all_workers_below_memory_threshold = all(worker['metrics']['memory'] < MEMORY_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values())
     
            if (all_workers_below_cpu_threshold and all_workers_below_memory_threshold):
                BATCH_SIZE = int(math.ceil(BATCH_SIZE * INCREASE_FACTOR)) if int(math.ceil(BATCH_SIZE * INCREASE_FACTOR)) <= MAX_BATCH_SIZE else MAX_BATCH_SIZE
                logging.info(f"Increasing batch size to {BATCH_SIZE}")
            else:
                BATCH_SIZE = max(MIN_BATCH_SIZE, int(BATCH_SIZE * (1-DECREASE_FACTOR)))
                logging.info(f"Decreasing batch size to {BATCH_SIZE}")
                #garbage_collection(False, 'Batch Size Decrease')

            eval_futures.clear()
            train_futures.clear()
            client.rebalance()
         
    #garbage_collection(False, "Cleaning WAIT -> done, not_done")     
    progress_bar.close()
            
    client.close()
    cluster.close()