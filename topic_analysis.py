# developed traditionally in with addition of AI assistance
# author: alan hamm(pqn7)
# date apr 2024

#%%
from SLIF import *

import argparse

from dask.distributed import Client, LocalCluster, performance_report, wait
from distributed import Future
import dask
import socket
import tornado

import yaml # used for logging configuration file
import logging
import logging.config

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


###################################
# BEGIN SCRIPT CONFIGURATION HERE #
###################################
"""
# windows cmd
python topic_analysis.py --time_period "2015-2019" --data_source "C:/topic-modeling/data/tokenized-sentences/2015-2019/2015-2019_min_six_word-w-bigrams.json" --start_topics 20 --end_topics 100 --step_size 5 --num_workers 8 --max_workers 14 --num_threads 8 --max_memory "5GB" --mem_threshold 4 --max_cpu 110 --futures_batches 100 --base_batch_size 100 --max_batch_size 130 --log_dir "C:/topic-modeling/data/lda-models/2015-2019/log/" --root_dir "C:/topic-modeling/data/lda-models/2015-2019/" 2>"C:/topic-modeling/data/lda-models/2015-2019/log/terminal_output.txt"
python topic_analysis.py --time_period "moby-dick" --data_source "C:/topic-modeling/data/tokenized-sentences/moby-dick/moby-dick-w-bigrams.json" --start_topics 20 --end_topics 100 --step_size 5 --num_workers 8 --max_workers 14 --num_threads 8 --max_memory "5GB" --mem_threshold 4 --max_cpu 110 --futures_batches 100 --base_batch_size 100 --max_batch_size 130 --log_dir "C:/topic-modeling/data/lda-models/moby-dick/log" --root_dir "C:/topic-modeling/data/lda-models/moby-dick/" 2>"C:/topic-modeling/data/lda-models/moby-dick/log/terminal_output.txt"
python topic_analysis.py --time_period "proust" --data_source "C:/topic-modeling/data/tokenized-sentences/proust/In-Search-of-Lost-Time-w-bigrams.json" --start_topics 100 --end_topics 1200 --step_size 50 --num_workers 8 --max_workers 14 --num_threads 8 --max_memory "5GB" --mem_threshold 4 --max_cpu 100 --futures_batches 100 --base_batch_size 100 --max_batch_size 130 --log_dir "C:/topic-modeling/data/lda-models/proust/log" --root_dir "C:/topic-modeling/data/lda-models/proust/" 2>"C:/topic-modeling/data/lda-models/proust/log/terminal_output.txt"
python topic_analysis.py --time_period "war-and-peace" --data_source "C:/topic-modeling/data/tokenized-sentences/war-and-peace/war-and-peace-w-bigrams.json" --start_topics 20 --end_topics 100 --step_size 5 --num_workers 8 --max_workers 14 --num_threads 8 --max_memory "5GB" --mem_threshold 4 --max_cpu 100 --futures_batches 100 --base_batch_size 50 --max_batch_size 100 --log_dir "C:/topic-modeling/data/lda-models/war-and-peace/log" --root_dir "C:/topic-modeling/data/lda-models/war-and-peace/" 2>"C:/topic-modeling/data/lda-models/war-and-peace/log/terminal_output.txt"
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Script configuration via CLI")
    parser.add_argument("--time_period",    type=str,           help="Decade to process") #no default
    parser.add_argument("--data_source",    type=str,           help="Path to data source JSON file") #no default
    parser.add_argument("--train_ratio",    type=float,     default=0.80,help="Train ratio for test-train split")
    parser.add_argument("--start_topics",   type=int,       default=1,   help="Minimum number of topics.")
    parser.add_argument("--end_topics",     type=int,           help="Maximum number of topics.") #no default
    parser.add_argument("--step_size",      type=int,           help="Value to determine how start_topic increases to end_topic, exlusive") #no default
    parser.add_argument("--num_workers",    type=int,       default=1,  help="The minimum number of cores to be utilized")
    parser.add_argument("--max_workers",    type=int,       default=1,  help="The maximum number of cores to be utilized")
    parser.add_argument("--num_threads",    type=int,       default=1,  help="The maximum number of threads to be utilized")
    parser.add_argument("--max_memory",     type=str,           help="The maximum amount of RAM(in GB) assigned to each core") #no default
    parser.add_argument("--mem_threshold",  type=int,           help="The memory threshold") #no default
    parser.add_argument("--max_cpu",        type=int,       default=100,    help="The maximum CPU utilization threshold")
    parser.add_argument("--mem_spill",      type=str,       default="c:/temp/slif/max_spill",    help="Directory to be used when RAM exceeds threshold")
    parser.add_argument("--passes",         type=int,       default=15,     help="Number of passes for Gensim model")
    parser.add_argument("--iterations",     type=int,       default= 100,   help="Number of iterations")
    parser.add_argument("--update_every",   type=int,       default=5,      help="Number of documents to be iterated through for each update. Default is 5. Set to 0 for batch learning, > 1 for online iterative learning.")
    parser.add_argument("--eval_every",     type=int,       default=5,      help="Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.")
    parser.add_argument("--random_state",   type=int,       default=50,     help="Either a randomState object or a seed to generate one. Useful for reproducibility.")
    parser.add_argument("--per_word_topics", type=bool,     default=True,   help=" If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).")
    parser.add_argument("--futures_batches", type=int,      default=1,    help="The number of futures in a batch")
    parser.add_argument("--base_batch_size", type=int,      default=1,    help="The number of documents to be processed in parallel")
    parser.add_argument("--max_batch_size",  type=int,      default=1,    help="The maximum number of documents to be processed in parallel")
    parser.add_argument("--increase_factor", type=float,    default=1.05,   help="Increase document batch size by p-percent")
    parser.add_argument("--decrease_factor", type=float,    default=.10,    help="Decrese batch size by p-percent upon failure or timeout")
    parser.add_argument("--max_retries",    type=int,       default=5,      help="Maximum numbewr of times to attempt to process a future")
    parser.add_argument("--base_wait_time", type=int,       default=30,     help="Base wait time in seconds for exponential backoff")
    parser.add_argument("--log_dir",        type=str,       default=os.path.expanduser("~/temp/slif/log"), help="Log directory output")
    parser.add_argument("--root_dir",       type=str,       default=os.path.expanduser("~/temp/slif/"),    help="Root directory")

    # Add more arguments as needed
    return parser.parse_args()

# Parse CLI arguments
args = parse_args()

# Load define time span to process
if args.time_period: DECADE_TO_PROCESS = args.time_period
else:
    logging.error(f"No value was entered for decade_to_process")
    print(f"No value was entered for decade_to_process")
    sys.exit
# Load data from the JSON file
if args.data_source:DATA_SOURCE = args.data_source
else:
    logging.error(f"No value was entered for data_source")
    print(f"No value was entered for data_source")
    sys.exit
# Define training ratio
if args.train_ratio: TRAIN_RATIO = args.train_ratio
# Define starting number of topics
if args.start_topics: START_TOPICS = args.start_topics
# Define starting number of topics
if args.end_topics: END_TOPICS = args.end_topics
else:
    logging.error(f"No value was entered for end_topics")
    print(f"No value was entered for end_topics")
    sys.exit
# Amount used to determine number of topics
if args.step_size: STEP_SIZE = args.step_size
# Number of cores
if args.num_workers: CORES = args.num_workers
# maximum number of cores
if args.max_workers: MAXIMUM_CORES = args.max_workers
# max amount of RAM per worker
if args.max_memory: RAM_MEMORY_LIMIT = args.max_memory
# Number of threads per core
if args.num_threads: THREADS_PER_CORE = args.num_threads
# max RAM assigned to each core
if args.mem_threshold: MEMORY_UTILIZATION_THRESHOLD = args.mem_threshold * (1024 ** 3)
# max CUP utilization
if args.max_cpu: CPU_UTILIZATION_THRESHOLD = args.max_cpu
# memory to spill into HDD/SDD
if args.mem_spill: 
    DASK_DIR = args.mem_spill
    os.makedirs(DASK_DIR, exist_ok=True)
# specify the number of passes for Gensim LdaModel
if args.passes: PASSES = args.passes
# specify the number of iterationx
if args.iterations: ITERATIONS = args.iterations
# Number of documents to be iterated through for each update. 
# Set to 0 for batch learning, > 1 for online iterative learning.
if args.update_every: UPDATE_EVERY = args.update_every
# Log perplexity is estimated every that many updates. 
# Setting this to one slows down training by ~2x.
if args.eval_every: EVAL_EVERY = args.eval_every
if args.random_state: RANDOM_STATE = args.random_state
if args.per_word_topics: PER_WORD_TOPICS = args.per_word_topics
# the number of documents( defined as HTML extract of <p>...</p> ) to read from the JSON source file per batch
if args.futures_batches: FUTURES_BATCH_SIZE = args.futures_batches
else:
    logging.error(f"No value was entered for futures_batches")
    print(f"No value was entered for futures_batches")
    sys.exit
# number of documents, value should be greater than FUTURES_BATCH_SIZE
if args.base_batch_size: BATCH_SIZE = args.base_batch_size
# maximum number of documents to be processed
if args.max_batch_size: MAX_BATCH_SIZE = args.max_batch_size
# minimum size of batch if resources are strained
if args.futures_batches: MIN_BATCH_SIZE = math.ceil(args.futures_batches * 1.01)
else:
    logging.error(f"min_batch_size parameter could not be assigned value")
    print(f"min_batch_size parameter could not be assigned value")
    sys.exit
# Increase batch size by p% upon success
if args.increase_factor: INCREASE_FACTOR = args.increase_factor
# decrease batch size by p% upon success
if args.decrease_factor: DECREASE_FACTOR = args.decrease_factor
# Maximum number of retries per task
if args.max_retries: MAX_RETRIES = args.max_retries
# Base wait time in seconds for exponential backoff
if args.base_wait_time: BASE_WAIT_TIME = args.base_wait_time


####################################################
# MUST BE DONE VIA TERMINAL WITH ADMIN PRIVILEGES  #
####################################################
# Set the JOBLIB_TEMP_FOLDER environment variable to your desired folder path
# By setting a custom temporary folder, you have more control over where joblib stores its 
# data and can avoid issues related to permissions or automatic cleanup of system temporary 
# directories. Remember to clean up this directory periodically if joblib does not do so 
# automatically, as it may accumulate large amounts of data over time.
# for joblib -- via CLI to only allow writing but not deleting
# icacls "C:\Temp\joblib" /grant "pqn7":(OI)(CI)W 

# for joblib --  Modify permissions to allow deletion:
# icacls "C:\path\to\directory" /grant "pqn7":M
# delete folders
# del "C:\path\to\directory\*"

custom_temp_folder = f"C:/topic-modeling/data/lda-models/{DECADE_TO_PROCESS}/log/joblib"
os.makedirs(custom_temp_folder, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = custom_temp_folder


###############################
###############################
# DO NOT EDIT BELOW THIS LINE #
###############################
###############################

# to escape: distributed.nanny - WARNING - Worker process still alive after 4.0 seconds, killing
# https://github.com/dask/dask-jobqueue/issues/391
scheduler_options={"host":socket.gethostname()}

# write Dask terminal output to file: C:/topic-modeling/data/lda-models/log_all/dask_log.log
#with open(r'C:\Users\pqn7\OneDrive - CDC\git-a-aitch\topic-modeling-pkg\dask.yaml', 'r') as f:
#    config = yaml.safe_load(f.read())
#    logging.config.dictConfig(config)

# Ensure the LOG_DIRECTORY exists
if args.log_dir: LOG_DIRECTORY = args.log_dir
if args.root_dir: ROOT_DIR = args.root_dir
#LOG_DIRECTORY = f"C:/topic-modeling/data/lda-models/{DECADE_TO_PROCESS}/log/"
#ROOT_DIR = f"C:/topic-modeling/data/lda-models/{DECADE_TO_PROCESS}"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

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


# Get the current date and time for log filename
now = datetime.now()

# Format the date and time as per your requirement
# Note: %w is the day of the week as a decimal (0=Sunday, 6=Saturday)
#       %Y is the four-digit year
#       %m is the two-digit month (01-12)
#       %H%M is the hour (00-23) followed by minute (00-59) in 24hr format
#log_filename = now.strftime('log-%w-%m-%Y-%H%M.log')
log_filename = 'log-0250.log'
LOGFILE = os.path.join(LOG_DIRECTORY,log_filename)

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

# Suppress specific SettingWithCopyWarning from pyLDAvis
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

# Enable serialization optimizations 
dask.config.set(scheduler='distributed', serialize=True) #note: could this be causing the pyLDAvis creation problems??
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.scheduler.worker-ttl": None})
dask.config.set({'distributed.worker.daemon': False})

#These settings disable automatic spilling but allow for pausing work when 80% of memory is consumed and terminating workers at 95%.
dask.config.set({'distributed.worker.memory.target': False,
                 'distributed.worker.memory.spill': False,
                 'distributed.worker.memory.pause': 0.8
                 ,'distributed.worker.memory.terminate': 0.99})

class NoCreatingAndSavingFilter(logging.Filter):
    def filter(self, record):
        return 'Creating and saving models' not in record.getMessage()
# Set up file handler
file_handler = logging.FileHandler(f'{LOG_DIR}/dask_distributed_err.log')
file_handler.setLevel(logging.INFO)

# Set up stream handler with filter
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.addFilter(NoCreatingAndSavingFilter())

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

    # Verify that memory limits have been set correctly
    #for worker_id, worker_info in workers_info.items():
    #    print(f"Worker {worker_id}: Memory Limit - {worker_info['memory_limit']}")

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

    #whole_train_dataset = None
    #whole_eval_dataset = None

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

        # Update the progress bar with the cumulative count of samples processed
        #pbar.update(batch_info['cumulative_count'] - pbar.n)
        #pbar.update(len(batch_info['data']))
    
    #pbar.close()  # Ensure closure of the progress bar

    print(f"Completed creation of training and evaluation documents in {round((time() - started)/60,2)} minutes.\n")
    #print(f"The size of the TRAIN scatter: {len(scattered_train_data_futures)}.")
    #print(f"The size of the EVAL scatter: {len(scattered_eval_data_futures)}.")
    print("Data scatter complete...\n")
    #garbage_collection(False, 'scattering training and eval data')
    #del scattered_future
    #del whole_train_dataset, whole_eval_dataset # these variables are not used at all

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

        #logging.info(f"for LdaModel hyperparameters combination -- type: {train_eval_type}, topic: {n_topics}, ALPHA: {alpha_value} and ETA {beta_value}")
        # Submit a future for each scattered data object in the training list
        #if train_eval_type == 'train':
        # Submit a future for each scattered data object in the training list
        for scattered_data in scattered_train_data_futures:
            try:
                future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'train',
                                            RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, CORES, PER_WORD_TOPICS)
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
                                        RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, CORES, PER_WORD_TOPICS)
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
                #print(f"We have completed the TRAIN list comprehension. The size is {len(completed_train_futures)}")
                #print(f"This is the length of the TRAIN completed_train_futures var {len(completed_train_futures)}")
                
                completed_eval_futures = [f for f in done_eval]
                #print(f"We have completed the EVAL list comprehension. The size is {len(completed_eval_futures)}")
                #print(f"This is the length of the EVAL completed_eval_futures var {len(completed_eval_futures)}")


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
                results_train = [d.result() for d in done_train if isinstance(d, Future)]
                results_eval = [d.result() for d in done_eval if isinstance(d, Future)]
                
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
                    unique_id = hashlib.md5(result_dict['time'].strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
                    
                    if unique_id not in processed_results:
                        processed_results.add(unique_id)
                            
                        try:
                            vis_future_pylda = client.submit(create_vis_pylda,
                                                            result_dict['lda_model'],
                                                            result_dict['corpus'],
                                                            result_dict['dictionary'],
                                                            unique_id,
                                                            CORES,
                                                            result_dict['text_md5'],
                                                            PYLDA_DIR)
                            visualization_futures_pylda.append(vis_future_pylda)
                        except Exception as e:
                                    logging.error(f"An error occurred in create_vis_pylda() Dask operation: {e}")
                                    logging.error(f"TYPE: pyLDA -- MD5: {result_dict['text_md5']}")

                        try:
                            vis_future_pcoa = client.submit(create_vis_pcoa,
                                                            result_dict['lda_model'],
                                                            result_dict['corpus'],
                                                            unique_id,
                                                            result_dict['text_md5'],
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
            completed_eval_futures, completed_train_futures = process_completed_futures(completed_train_futures, \
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


            for f in completed_eval_futures: client.cancel(f)
            for f in completed_train_futures: client.cancel(f)
            for f in completed_pcoa_vis: client.cancel(f)
            for f in completed_pylda_vis: client.cancel(f)
            del completed_eval_futures, completed_train_futures, completed_pcoa_vis, completed_pylda_vis
            
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

    # After all loops have finished running...
    if len(train_futures) > 0 or len(eval_futures) > 0:
        print("we are in the first IF statement for retry_processing()")
        failed_model_params = retry_processing(train_futures, eval_futures, "10 minutes")
    
    #defensive programming to ensure WAIT output list of futures are empty
    #for f in train_futures: client.cancel(f)
    #for f in eval_futures: client.cancel(f)

    # Now give one more chance with extended timeout only to those that were incomplete previously
    if len(failed_model_params) > 0:
        print("Retrying incomplete models with extended timeout...")
        
        # Create new lists for retrying futures
        retry_train_futures = []
        retry_eval_futures = []

        # Resubmit tasks only for those that failed in the first attempt
        for params in failed_model_params:
            n_topics, alpha_value, beta_value = params
            
            #with performance_report(filename=PERFORMANCE_TRAIN_LOG):
            try:
                future_train_retry = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_train_data_futures, 'train')
                future_eval_retry = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_eval_data_futures, 'eval')
                pass
            except Exception as e:
                distributed_logger.exception("An error occurred in Dask operation")

            retry_train_futures.append(future_train_retry)
            retry_eval_futures.append(future_eval_retry)

            # Keep track of these new futures as well
            #future_to_params[future_train_retry] = params
            #future_to_params[future_eval_retry] = params

        # Clear the list of failed model parameters before reattempting
        failed_model_params.clear()

        # Wait for all reattempted futures with an extended timeout (e.g., 120 seconds)
        done, not_done = wait(retry_train_futures + retry_eval_futures, timeout=None) #, timeout=EXTENDED_TIMEOUT)

        # Process completed ones after reattempting
        num_workers = len(client.scheduler_info()["workers"])
        completed_eval_futures, completed_train_futures = process_completed_futures(completed_train_futures, \
                                                                                        completed_eval_futures, \
                                                                                        (len(completed_eval_futures)+len(completed_train_futures)), \
                                                                                        num_workers, \
                                                                                        BATCH_SIZE, \
                                                                                        TEXTS_ZIP_DIR, \
                                                                                        METADATA_DIR, \
                                                                                        vis_pylda=completed_pylda_vis,
                                                                                        vis_pcoa=completed_pcoa_vis)

        # Record parameters of still incomplete futures after reattempting for later review
        #for future in not_done:
        #    failed_model_params.append(future_to_params[future])

        # At this point `failed_model_params` contains the parameters of all models that didn't complete even after a retry

    #client.close()
    print("The training and evaluation loop has completed.")
    logging.info("The training and evaluation loop has completed.")

    if len(failed_model_params) > 0:
        # You can now review `failed_model_params` to see which models did not complete successfully.
        logging.error("The following model parameters did not complete even after a second attempt:")
    #    perf_logger.info("The following model parameters did not complete even after a second attempt:")
        for params in failed_model_params:
            logging.error(params)
    #        perf_logger.info(params)
            
    client.close()
    cluster.close()

