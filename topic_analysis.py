# developed traditionally in addition to pair programming
# author: alan hamm(pqn7)
# date apr 2024

#%%
from SLIF import *
from dask.distributed import Client, LocalCluster, performance_report, wait
from distributed import Future
import dask
import os
import logging
from datetime import datetime
from tqdm import tqdm
import math
import sys
from time import time, sleep
import numpy as np
import itertools
import random
import pandas as pd
import hashlib
import pickle

# Dask dashboard throws deprecation warnings w.r.t. Bokeh
import warnings
from bokeh.util.deprecation import BokehDeprecationWarning
from numpy import ComplexWarning


###################################
# BEGIN SCRIPT CONFIGURATION HERE #
###################################
DECADE_TO_PROCESS ='2015-2019'

# Load data from the JSON file
DATA_SOURCE = f"C:/topic-modeling/data/tokenized-sentences/{DECADE_TO_PROCESS}/{DECADE_TO_PROCESS}_min_six_word-w-bigrams.json"
# test-train split
TRAIN_RATIO = .80

# Define the range of number of topics for LDA and step size
START_TOPICS = 20
END_TOPICS = 120
STEP_SIZE = 5

# define the decade that is being modelled 
DECADE = DECADE_TO_PROCESS

# In the case of this machine, since it has an Intel Core i9 processor with 8 physical cores (16 threads with Hyper-Threading), 
# it would be appropriate to set the number of workers in Dask Distributed LocalCluster to 8 or slightly lower to allow some CPU 
# resources for other tasks running on your system.
# https://www.intel.com/content/www/us/en/products/sku/228439/intel-core-i912950hx-processor-30m-cache-up-to-5-00-ghz/specifications.html
CORES = 10
MAXIMUM_CORES = 12
THREADS_PER_CORE = 2
RAM_MEMORY_LIMIT = "10GB" # Dask diagnostics significantly overestimates RAM usage
CPU_UTILIZATION_THRESHOLD = 110 # eg 85%
MEMORY_UTILIZATION_THRESHOLD = 9 * (1024 ** 3)  # Convert GB to bytes
# Specify the local directory path, spilling will be written here
DASK_DIR = '/topic-modeling/dask-spill'

# specify the number of passes for Gensim LdaModel
PASSES = 15
# specify the number of iterations
ITERATIONS = 50
# Number of documents to be iterated through for each update. 
# Set to 0 for batch learning, > 1 for online iterative learning.
UPDATE_EVERY = 5
# Log perplexity is estimated every that many updates. 
# Setting this to one slows down training by ~2x.
EVAL_EVERY = 10
RANDOM_STATE = 75
PER_WORD_TOPICS = True


# the number of documents( defined as HTML extract of <p>...</p> ) to read from the JSON source file per batch
FUTURES_BATCH_SIZE = 50

# Constants for adaptive batching and retries
# Number of futures to process per iteration
BATCH_SIZE = 50 # number of documents, value should be greater than FUTURES_BATCH_SIZE
MAX_BATCH_SIZE = 60 # maximum number of documents to be processed
MIN_BATCH_SIZE = math.ceil(FUTURES_BATCH_SIZE * 1.01) # minimum size of batch if resources are strained
INCREASE_FACTOR = 1.05  # Increase batch size by p% upon success
DECREASE_FACTOR = .10 # Decrease batch size by p% upon failure or timeout
MAX_RETRIES = 5        # Maximum number of retries per task
BASE_WAIT_TIME = 30     # Base wait time in seconds for exponential backoff


# timeout for Dask wait method
TIMEOUT = None #"90 minutes"
# timeout for Dask wait method in retry_processing() method
EXTENDED_TIMEOUT = None #"120 minutes"


# Ensure the LOG_DIRECTORY exists
LOG_DIRECTORY = f"C:/topic-modeling/data/lda-models/{DECADE_TO_PROCESS}/log/"
ROOT_DIR = f"C:/topic-modeling/data/lda-models/{DECADE_TO_PROCESS}"
os.makedirs(LOG_DIRECTORY, exist_ok=True)


###############################
###############################
# DO NOT EDIT BELOW THIS LINE #
###############################
###############################

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
log_filename = 'log-1600.log'
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

# Disable Bokeh deprecation warnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
# Set the logging level for distributed.utils_perf to suppress warnings
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="distributed.utils_perf")


# Enable serialization optimizations 
dask.config.set(scheduler='distributed', serialize=True) #note: could this be causing the pyLDAvis creation problems??
#These settings disable automatic spilling but allow for pausing work when 80% of memory is consumed and terminating workers at 95%.
dask.config.set({'distributed.worker.memory.target': False,
                 'distributed.worker.memory.spill': False,
                 'distributed.worker.memory.pause': 0.8
                 ,'distributed.worker.memory.terminate': 0.99})
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.scheduler.worker-ttl": None})


# https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
if __name__=="__main__":
    cluster = LocalCluster(
            n_workers=CORES,
            threads_per_worker=THREADS_PER_CORE,
            processes=False,
            memory_limit=RAM_MEMORY_LIMIT,
            local_directory=DASK_DIR,
            #dashboard_address=None,
            dashboard_address=":8787",
            protocol="tcp",
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
        print("Dask client is connected to a scheduler.")
        # Scatter the embedding vectors across Dask workers
    else:
        print("Dask client is not connected to a scheduler.")
        print("The system is shutting down.")
        client.close()
        cluster.close()
        sys.exit()

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
        print(f"{CORES} Dask workers are running.")
    else:
        print("No Dask workers are running.")
        print("The system is shutting down.")
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
    progress_bar = tqdm(total=TOTAL_COMBINATIONS, desc="Creating and saving models")
    # Iterate over the combinations and submit tasks
    num_iter = 0
    for n_topics, alpha_value, beta_value, train_eval_type in random_combinations:
        #print(f"this is the number of for loop iterations: {num_iter}")
        num_iter+=1
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
            garbage_collection(False, 'Max Retries - throttling attempt')
        else:
            logging.info("Proceeding with workload as workers are below the CPU and Memory thresholds.")

        #logging.info(f"for LdaModel hyperparameters combination -- type: {train_eval_type}, topic: {n_topics}, ALPHA: {alpha_value} and ETA {beta_value}")
        # Submit a future for each scattered data object in the training list
        #if train_eval_type == 'train':
        # Submit a future for each scattered data object in the training list
        for scattered_data in scattered_train_data_futures:
            future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'train',
                                    RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, CORES, PER_WORD_TOPICS)
            train_futures.append(future)
            logging.info(f"The training value is being appended to the train_futures list. Size: {len(train_futures)}")

        # Submit a future for each scattered data object in the evaluation list
        #if train_eval_type == 'eval':
        for scattered_data in scattered_eval_data_futures:
            future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'eval',
                                    RANDOM_STATE, PASSES, ITERATIONS, UPDATE_EVERY, EVAL_EVERY, CORES, PER_WORD_TOPICS)
            eval_futures.append(future)
            logging.info(f"The evaluation value is being appended to the eval_futures list. Size: {len(eval_futures)}")
        #garbage_collection(False, 'client.submit(train_model(...) train and eval)')


        # Check if it's time to process futures based on BATCH_SIZE
        if (len(train_futures) + len(eval_futures)) >= BATCH_SIZE:
            time_of_vis_call = pd.to_datetime('now')
            time_of_vis_call = time_of_vis_call.strftime('%Y%m%d%H%M%S%f')
            PERFORMANCE_TRAIN_LOG = os.path.join(LOG_DIR, f"train_perf_{time_of_vis_call}.html")
            del time_of_vis_call

            with performance_report(filename=PERFORMANCE_TRAIN_LOG):
                logging.info("In holding pattern until WAIT completes.")
                started = time()

                # Wait for completion of eval_futures
                done_eval, not_done_eval = wait(eval_futures, timeout=None)  # return_when='FIRST_COMPLETED'
                logging.info(f"This is the size of the done_eval list: {len(done_eval)} and this is the size of the not_done_eval list: {len(not_done_eval)}")

                # Wait for completion of train_futures
                done_train, not_done_train = wait(train_futures, timeout=None)  # return_when='FIRST_COMPLETED'
                logging.info(f"This is the size of the done_train list: {len(done_train)} and this is the size of the not_done_train list: {len(not_done_train)}")

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
                results_train = [d.result() for d in done_train if  isinstance(d, Future)]       
                results_eval = [d.result() for d in done_eval if  isinstance(d, Future)]       
                results = results_train + results_eval
                logging.info(f"Completed TRAIN and EVAL gathering {len(results)} futures.") 
                if len(results) != (len(done_train) + len(done_eval)):
                    logging.error(f"All DONE({len(results)}) futures could not be resolved.")

                # Now you can process these results and submit new tasks based on them
                visualization_futures = []
                #results = completed_train_futures + completed_eval_futures
                for r in results:
                    for result in r:
                        # Process your result here and define a new task based on it
                        vis_future = client.submit(create_vis, pickle.loads(result['lda_model']), \
                                                        pickle.loads(result['corpus']), \
                                                        pickle.loads(result['dictionary']),
                                                        hashlib.md5(result['time'].strftime('%Y%m%d%H%M%S%f').encode()).hexdigest(), \
                                                        CORES, PYLDA_DIR, PCOA_DIR  )
                        visualization_futures.append(vis_future)

                logging.info(f"Executing WAIT on TRAIN and EVAL create_visualizations {len(visualization_futures)} futures.")
                # Wait for all visualization tasks to complete
                done_visualizations, not_done_visualizations = wait(visualization_futures)
                if len(not_done_visualizations) > 0:
                    logging.error(f"All TRAIN and EVAL visualizations couldn't be generated. There were {len(not_done_visualizations)} not created.")

                # Gather the results from the completed visualization tasks
                logging.info("Gathering TRAIN and EVAL completed visualization results futures.")
                completed_visualization_results = [future.result() for future in done_visualizations]
                #del completed_visualization_results
                logging.info(f"Completed gathering {len(completed_visualization_results)} TRAIN and EVAL visualization results futures.")

                elapsed_time = round(((time() - started) / 60), 2)
                logging.info(f"Create visualizations for TRAIN and EVAL data completed in {elapsed_time} minutes")
                #print(f"Create visualizations for TRAIN and EVAL data completed in {elapsed_time} minutes")
            # close performance report encapsulation of visualization performance analysis
            #############################
            # END PROCESS VISUALIZATIONS
            #############################            
            

            num_workers = len(client.scheduler_info()["workers"])
            completed_eval_futures, completed_train_futures = process_completed_futures(completed_train_futures, \
                                                                                        completed_eval_futures, \
                                                                                        (len(completed_eval_futures)+len(completed_train_futures)), \
                                                                                        num_workers, \
                                                                                        BATCH_SIZE, \
                                                                                        TEXTS_ZIP_DIR, \
                                                                                        METADATA_DIR, \
                                                                                        visualization_results=completed_visualization_results)
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
                garbage_collection(False, 'Batch Size Decrease')


            #defensive programming to ensure WAIT output list of futures are cancelled to clear memory
            for f in done: client.cancel(f)
            for f in done_train: client.cancel(f)
            for f in completed_visualization_results: client.cancel(f)
            for f in visualization_futures: client.cancel(f)
            for f in done_visualizations: client.cancel(f)
            for f in not_done_visualizations: client.cancel(f)

            
            del done, not_done, done_train, done_eval, not_done_eval, not_done_train
            del visualization_futures, done_visualizations, not_done_visualizations
            garbage_collection(False,'End of a batch being processed.')
            client.rebalance()
         
    #garbage_collection(False, "Cleaning WAIT -> done, not_done")     
    progress_bar.close()

    # After all loops have finished running...
    if len(train_futures) > 0 or len(eval_futures) > 0:
        print("we are in the first IF statement for retry_processing()")
        failed_model_params = retry_processing(train_futures, eval_futures, TIMEOUT)
    
    #defensive programming to ensure WAIT output list of futures are empty
    for f in train_futures: client.cancel(f)
    for f in eval_futures: client.cancel(f)

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
            future_train_retry = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_train_data_futures, 'train')
            future_eval_retry = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_eval_data_futures, 'eval')

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
        completed_eval_futures, completed_train_futures = process_completed_futures(retry_train_futures, \
                                                                                    retry_eval_futures, \
                                                                                    (len(retry_train_futures)+len(retry_eval_futures)), \
                                                                                    num_workers, \
                                                                                    BATCH_SIZE, \
                                                                                    TEXTS_ZIP_DIR, \
                                                                                    METADATA_DIR, \
                                                                                    visualization_results=completed_visualization_results)

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

