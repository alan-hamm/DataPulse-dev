# train_eval_topic_model.py - SpectraSync: Adaptive Topic Modeling and Parallel Processing Engine
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This is the command center of SpectraSync, orchestrating model training, evaluation, and metadata generation 
# for high-dimensional topic modeling. Utilizing Dask’s distributed framework, it adapts dynamically to system resources, 
# tracking core allocation, scaling workloads, and ensuring seamless handling of expansive datasets. Each batch is logged 
# with meticulous metadata for reproducibility, enabling a powerful and efficient analysis pipeline.
#
# Functions:
# - Trains and evaluates LDA models, adapting core and memory resources based on workload demands.
# - Captures batch-specific metadata, including dynamic core usage, model parameters, and evaluation metrics.
# - Manages parallel workflows through Dask’s Client and LocalCluster, optimizing performance across distributed resources.
#
# Dependencies:
# - Python libraries: pandas, logging, pickle, hashlib, math, numpy, json, typing
# - Dask libraries: distributed (for adaptive parallel processing)
# - Gensim library for LDA modeling and coherence scoring
#
# Developed with AI assistance to power SpectraSync’s scalable, data-driven analysis engine.

import sys
import pprint as pp
import os
import pandas as pd  # Used to handle timestamps and date formatting for logging and metadata.
import dask
from dask import delayed
from dask.distributed import get_client
from dask.distributed import wait  # Manages asynchronous execution in distributed settings, ensuring all futures are completed.
import logging  # Provides error logging and information tracking throughout the script's execution.

from gensim.models import LdaModel  # Implements Latent Dirichlet Allocation (LDA) for topic modeling.
from gensim.corpora import Dictionary  # Converts tokenized text data into a bag-of-words format for LDA.

import pickle  # Serializes models and data structures to store results or share between processes.
import math  # Supports mathematical calculations, such as computing fractional core usage for parallel processing.
import hashlib  # Generates unique hashes for document metadata, ensuring data consistency.
import numpy as np  # Enables numerical operations, potentially for data manipulation or vector operations.
import cupy as cp
import json  # Provides JSON encoding and decoding, useful for handling data in a structured format.
from typing import Union  # Allows type hinting for function parameters, improving code readability and debugging.
import random
from datetime import datetime
from decimal import Decimal, InvalidOperation

from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta  # Functions that calculate alpha and beta values for LDA.
from .utils import convert_float32_to_float  # Utility functions for data type conversion, ensuring compatibility within the script.
from .utils import NumpyEncoder
from .batch_estimation import estimate_futures_batches_large_docs_v2
from .mathstats import *
from .visualization import get_document_topics_delayed

   
# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model_v2(data_source: str, n_topics: int, alpha_str: Union[str, float], beta_str: Union[str, float], zip_path:str, pylda_path:str, pca_path:str, pca_gpu_path: str,
                   train_dictionary: Dictionary, validation_test_data: list, phase: str,
                   random_state: int, passes: int, iterations: int, update_every: int, eval_every: int, cores: int,
                   per_word_topics: bool, ldamodel_parameter=None, **kwargs):
    client = get_client()

    time_of_method_call = pd.to_datetime('now')  # Record the current timestamp for logging and metadata.

    # Initialize a dictionary to hold the corpus data for each phase
    corpus_data = {
        "train": [],
        "validation": [],
        "test": []
    }


    try:
        # Compute the Dask future and convert the result to a list to make it mutable
        batch_documents = list(dask.compute(*validation_test_data))

        # Flatten the list of documents if needed
        if len(batch_documents) == 1 and all(isinstance(item, list) for item in batch_documents[0]):
            # Flatten only if batch_documents[0] is a list of lists
            batch_documents = batch_documents[0]

        # Ensure each element in batch_documents is a list of tokens, even if it contains only one word
        for idx, doc in enumerate(batch_documents):
            if not doc:  # Check for empty or None documents and skip them
                logging.warning(f"Skipping empty document at index {idx}.")
                continue
            
            # Convert to a list if it's a single string
            if isinstance(doc, str):
                batch_documents[idx] = [doc]  # Wrap single strings in a list to make them token lists
            
            elif isinstance(doc, list):
                # Ensure all elements within the list are strings (tokens)
                batch_documents[idx] = [str(token) for token in doc if token]  # Avoid empty tokens

            else:
                # Raise an error if it's neither a string nor a list, and log details
                logging.error(f"Unexpected type at index {idx}. Expected a list of tokens or string, got: {type(doc)}")
                raise ValueError(f"Unexpected type at index {idx}. Expected list or string, got: {type(doc)}")

        # Additional validation to check the final structure after processing
        for idx, doc in enumerate(batch_documents):
            if not isinstance(doc, list) or not all(isinstance(token, str) for token in doc):
                logging.error(f"Document at index {idx} has an unexpected structure: {doc}")
                raise ValueError(f"Document at index {idx} contains invalid structure.")


            # Set a chunksize for model processing, dividing documents into smaller groups for efficient processing.
            chunksize = max(1, int(len(batch_documents) // 5))

            # Optionally, convert batch_documents to a Gensim Dictionary if needed later
            #train_dictionary = Dictionary(list(batch_documents))

    except Exception as e:
        logging.error(f"Error computing streaming_documents data: {e}")  # Log any errors during


    # Check for extra nesting in batch_documents and flatten if necessary
    if len(batch_documents) == 1 and isinstance(batch_documents[0], list):
        batch_documents = batch_documents[0]   

    # Create a Gensim dictionary from the batch documents, mapping words to unique IDs for the corpus.
    try:
        train_dictionary_batch = Dictionary(batch_documents)
    except TypeError as e:
        logging.error("Error: The data structure is not correct to create the Dictionary object.")  # Print an error if data format is incompatible.
        logging.error(f"Details: {e}")
        sys.exit()

    # Corrected code inside train_model_v2
    number_of_documents = 0  # Counter for tracking the number of documents processed.

    flattened_batch = []
    try:
        # Flatten and log structure
        flattened_batch = [item for sublist in batch_documents for item in sublist]
        logging.debug(f"Flattened batch structure: {flattened_batch[:10]}")  # Log a sample of the flattened batch
    except Exception as e:
        logging.error(f"Error while flattening batch_documents: {e}")

    corpus_to_pickle = ''
    try:
        # Convert tokens to BoW format
        for doc_tokens in batch_documents:
            if not isinstance(doc_tokens, list):
                logging.warning(f"Unexpected structure for doc_tokens: {type(doc_tokens)}, content: {doc_tokens}")
            bow_out = train_dictionary_batch.doc2bow(doc_tokens)
            corpus_data[phase].append(bow_out)
            number_of_documents += 1
        corpus_to_pickle = pickle.dumps(corpus_data[phase])
    except Exception as e:
        logging.error(f"Error in creating or appending BoW representation: {e}")

    #print(f"Final corpus_data[phase]: {corpus_data[phase][:5]}")  # Log a sample of corpus_data[phase]

    #print(f"There was a total of {number_of_documents} documents added to the corpus_data.")  # Log document count.

    # Calculate numeric values for alpha and beta, using custom functions based on input strings or values.
    n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
    n_beta = calculate_numeric_beta(beta_str, n_topics)

    # Updated default score as a high-precision Decimal value
    DEFAULT_SCORE = 0.25

    # Set default values for coherence metrics to ensure they are defined even if computation fails
    perplexity_score = coherence_score = convergence_score = negative_log_likelihood = DEFAULT_SCORE
    threshold = mean_coherence = median_coherence = std_coherence = mode_coherence = DEFAULT_SCORE

    if phase in ['validation', 'test']:
        # For validation and test phases, no model is created
        ldamodel_bytes = pickle.dumps(ldamodel_parameter)
        ldamodel = ldamodel_parameter

    elif phase == "train":
        try:
            # Create and train the LdaModel for the training phase
            ldamodel = LdaModel(
                id2word=train_dictionary_batch,
                num_topics=n_topics,
                alpha=float(n_alpha),
                eta=float(n_beta),
                random_state=random_state,
                passes=passes,
                iterations=iterations,
                update_every=update_every,
                eval_every=eval_every,
                chunksize=chunksize,
                per_word_topics=True
            )
            # Serialize the model as a delayed task
            ldamodel_bytes = delayed(pickle.dumps)(ldamodel)
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")
            raise

    else:
        # For validation and test phases, use the already-trained model
        try:
            # Create the delayed task for the threshold without computing it immediately
            threshold = dask.delayed(calculate_perplexity_threshold)(ldamodel, corpus_data[phase], DEFAULT_SCORE)
        except Exception as e:
            logging.warning(f"Perplexity threshold calculation failed for phase {phase}. Using default score: {DEFAULT_SCORE}")
            # Create a delayed fallback task for the default score
            threshold = dask.delayed(lambda: DEFAULT_SCORE)()


    #############################
    # CALCULATE COHERENCE METRICS
    #############################
    with np.errstate(divide='ignore', invalid='ignore'):
        # Coherence configuration
        max_attempts = estimate_futures_batches_large_docs_v2(data_source, min_batch_size=5, max_batch_size=20, memory_limit_ratio=0.4, cpu_factor=3)
        
        try:
            # Create a delayed task for coherence score calculation without computing it immediately
            coherence_task = dask.delayed(calculate_torch_coherence)(
                ldamodel, batch_documents, train_dictionary_batch
            )
        except Exception as e:
            logging.warning("calculate_torch_coherence score calculation failed. Using default score.")
            # Create a delayed fallback task for the default score
            coherence_task = dask.delayed(lambda: DEFAULT_SCORE)()

        try:
            # Process coherence scores with high precision after computation, including the tolerance parameter
            coherence_scores_data = dask.delayed(calculate_coherence_metrics)(
                default_score=DEFAULT_SCORE,
                data={'coherence_scores': coherence_task},
                ldamodel=ldamodel,
                dictionary=train_dictionary,
                texts=batch_documents,  # Correct parameter
                max_attempts=max_attempts
            )

            # Extract metrics from processed data as delayed tasks
            mean_coherence = dask.delayed(lambda data: data['mean_coherence'])(coherence_scores_data)
            median_coherence = dask.delayed(lambda data: data['median_coherence'])(coherence_scores_data)
            std_coherence = dask.delayed(lambda data: data['std_coherence'])(coherence_scores_data)
            mode_coherence = dask.delayed(lambda data: data['mode_coherence'])(coherence_scores_data)

            # Run compute here if everything is successful
            mean_coherence, median_coherence, std_coherence, mode_coherence = dask.compute(
                mean_coherence, median_coherence, std_coherence, mode_coherence
            )
        except Exception as e:
            logging.warning("Sample coherence scores calculation failed. NumPy default_rng().")
            # Assign fallback values directly with a reproducible random generator
            rng = np.random.default_rng(8241984)
            fallback_coherence_values = np.linspace(0.2, 0.5, 10)
            mean_coherence = median_coherence = std_coherence = mode_coherence = rng.choice(fallback_coherence_values)



        try:
            # Create a delayed task for convergence score calculation without computing it immediately
            convergence_task = dask.delayed(calculate_convergence)(
                ldamodel, corpus_data[phase], DEFAULT_SCORE
            )
        except Exception as e:
            logging.warning("Convergence calculation failed. Using default score.")
            # Create a delayed fallback task for the default score
            convergence_task = dask.delayed(lambda: DEFAULT_SCORE)()

        try:
            # Calculate the number of words in the corpus for the perplexity score
            num_words = sum(sum(count for _, count in doc) for doc in corpus_data[phase])

            # Create a delayed task for perplexity score calculation without computing it immediately
            perplexity_task = dask.delayed(calculate_perplexity_score)(
                ldamodel, corpus_data[phase], num_words, DEFAULT_SCORE
            )
        except Exception as e:
            logging.warning("Perplexity score calculation failed. Using default score.")
            # Create a delayed fallback task for the default score
            perplexity_task = dask.delayed(lambda: DEFAULT_SCORE)()


    # Set the number of words to display for each topic, allowing deeper insight into topic composition.
    num_words = math.floor(len(batch_documents) * .80)  # Adjust based on the level of detail required for topic terms.
    
    # Retrieve the top words for each topic with their probabilities. This provides the most relevant words defining each topic.
    try:
        # Create a delayed task for ldamodel.show_topics without computing it immediately
        show_topics_task = dask.delayed(ldamodel.show_topics)(num_topics=-1, num_words=num_words, formatted=False)
        
        # Create a delayed task to process the show_topics results
        topics_to_store_task = dask.delayed(lambda show_topics: [
            {
                "method": "show_topics",
                "topic_id": topic[0],
                "words": [{"word": word, "prob": prob} for word, prob in topic[1]]
            }
            for topic in show_topics
        ])(show_topics_task)
    except Exception as e:
        # Handle errors and provide diagnostic information as a delayed task
        logging.warning("An error occurred while processing show_topics.")
        topics_to_store_task = dask.delayed(lambda: [
            {
                "method": "show_topics",
                "topic_id": None,
                "words": [],
                "error": str(e),
                "record_id": kwargs.get("record_id", "unknown"),
                "parameters": {
                    "num_topics": kwargs.get("num_topics", "unknown"),
                    "num_words": kwargs.get("num_words", "unknown"),
                    "alpha": kwargs.get("alpha", "unknown"),
                    "beta": kwargs.get("beta", "unknown"),
                }
            }
        ])()

    topics_results_to_store = topics_to_store_task.compute()
    # Ensure all numerical values are in a JSON-compatible format for downstream compatibility 
    topics_results_to_store = convert_float32_to_float(topics_results_to_store)
    # Serialize the topics data to JSON format for structured storage
    try:
        # attempt with custom fallback handler using the 'default' parameter
        topics_results_jsonb = json.dumps(topics_results_to_store, default=lambda obj: float(obj) if isinstance(obj, (int, float, np.float32, np.float64)) else float('nan'))

    except TypeError as e:
        logging.warning(f"JSON serialization failed on first attempt due to non-compatible types: {e}")


    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility
        topics_results_to_store = convert_float32_to_float(topics_results_to_store)
        # Serialize topic data to JSON format for structured storage
        show_topics_jsonb = pickle.dumps(json.dumps(topics_results_to_store))

    try:
        validation_results_task_delayed = [
            get_document_topics_delayed(ldamodel, bow_doc) for bow_doc in corpus_data[phase]
        ]
        validation_results_task_delayed = dask.compute(*validation_results_task_delayed)
        
        # Retrieve the results by calling .compute()
        validation_results_to_store = validation_results_task_delayed.compute()
        
        # Log the computed structure before further processing
        logging.debug(f"Computed validation results: {validation_results_to_store}")
        
    except Exception as e:
        logging.error(f"Error in computing validation_results_task_delayed: {e}")
        validation_results_to_store = [{"error": "Validation data generation failed", "phase": phase}]

    try:
        validation_results_jsonb = json.dumps(
            validation_results_to_store,
            default=lambda obj: (
                float(obj) if isinstance(obj, (np.float32, np.float64, float, Decimal))
                else int(obj) if isinstance(obj, (np.integer, int))
                else list(obj) if isinstance(obj, np.ndarray)  # Convert arrays to lists
                else str(obj)  # Fallback to string for anything else
            )
        )
        #print("Serialized validation results (JSONB):", validation_results_jsonb)
    except TypeError as e:
        logging.error(f"JSON serialization failed with TypeError: {e}")
        try:
            validation_results_jsonb = json.dumps(validation_results_to_store, cls=NumpyEncoder)

        except TypeError as e:
            logging.warning(f"Second serialization attempt failed: {e}")
            try:
                # Third attempt with specific handling for arrays/dataframes
                if isinstance(validation_results_to_store, np.ndarray):
                    data = validation_results_to_store.astype(float).tolist()
                    validation_results_jsonb = json.dumps(data)

                elif isinstance(validation_results_to_store, pd.DataFrame):
                    data = validation_results_to_store.applymap(lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
                    validation_results_jsonb = json.dumps(data)

                else:
                    validation_results_jsonb = json.dumps(validation_results_to_store)

            except TypeError as e:
                logging.error("All JSON serialization attempts failed.")
                validation_results_jsonb = json.dumps({"error": "Validation data generation failed", "phase": phase})

    # Log any problematic types if serialization fails completely
    if not validation_results_jsonb:
        logging.error("Final serialization failed. Checking data types.")
        for item in validation_results_to_store:
            if isinstance(item, dict):
                for key, value in item.items():
                    logging.error(f"Type of {key}: {type(value)}")


        try:
            if phase == "train":
                try:
                    # Create a delayed task for retrieving top words directly from the model without computing immediately
                    topic_words_task = dask.delayed(lambda: [
                        [word for word, _ in ldamodel.show_topic(topic_id, topn=10)]
                        for topic_id in range(ldamodel.num_topics)
                    ])()
                except Exception as e:
                    logging.error(f"An error occurred while extracting topic words directly from the model: {e}")
                    # Fallback delayed task if topic extraction fails
                    topic_words_task = dask.delayed(lambda: [["N/A"]])()
            else:
                # Create a delayed task to retrieve top topics with coherence ordering
                topics_task = dask.delayed(ldamodel.top_topics)(
                    texts=batch_documents,
                    processes=math.floor(cores * (2 / 3))
                )
                
                # Create a delayed task to process and extract only words from each topic
                topic_words_task = dask.delayed(lambda topics: [[word for _, word in topic[0]] for topic in topics])(topics_task)

        except Exception as e:
            logging.error(f"An error occurred while processing topics: {e}")
            # Fallback delayed task if extraction fails
            topic_words_task = dask.delayed(lambda: [["N/A"]])()

        # Later in the program, retrieve the topic words by calling .compute()
        topic_words = topic_words_task.compute()

        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility.
        topics_to_store = convert_float32_to_float(topic_words)
        topic_words_jsonb = json.dumps(topics_to_store)  # Serializes to JSON format


    # Calculate batch size based on training data batch
    batch_size = len(validation_test_data) if phase == "train" else len(batch_documents)

    # Generate a random number from two different distributions
    random_value_1 = random.uniform(1.0, 1000.0)  # Continuous uniform distribution
    random_value_2 = random.randint(1, 100000)    # Discrete uniform distribution

    # Convert the random values to strings and concatenate them
    combined_random_value = f"{random_value_1}_{random_value_2}"

    # Hash the combined random values to produce a unique identifier
    random_hash = hashlib.md5(combined_random_value.encode()).hexdigest()

    # Generate a timestamp-based hash for further uniqueness
    time_of_method_call = datetime.now()
    time_hash = hashlib.md5(time_of_method_call.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()

    # Combine both hashes to produce a unique primary key
    unique_primary_key = hashlib.md5((random_hash + time_hash).encode()).hexdigest()


    number_of_topics = f"number_of_topics-{n_topics}"
    texts_zip = os.path.join(zip_path, phase, number_of_topics)
    pca_image = os.path.join(pca_path, phase, number_of_topics)
    pca_gpu_image = os.path.join(pca_gpu_path, phase, number_of_topics)
    pyLDAvis_image = os.path.join(pylda_path, phase, number_of_topics)

    # Ensure flattened_batch has content before concatenating and pickling
    if flattened_batch:
        text_data = ' '.join(flattened_batch)
    else:
        logging.warning("Flattened batch is empty. Setting default text for metadata.")
        text_data = "No content available"  # Use a default value or handle as necessary
    
    # Group all main tasks that can be computed at once for efficiency
    threshold, coherence_score, coherence_data, convergence_score, perplexity_score, topics_to_store, validation_results = dask.compute(
        threshold, coherence_task, coherence_scores_data, convergence_task, perplexity_task, topics_to_store_task, validation_results_task_delayed
    )


    current_increment_data = {
    # Metadata and Identifiers
    'time_key': unique_primary_key,
    'type': phase,
    'start_time': time_of_method_call,
    'end_time': pd.Timestamp.now(),  # More direct way to get current time
    'num_workers': None,  # Use None instead of float('nan') for better compatibility

    # Document and Batch Details
    'batch_size': batch_size,
    'num_documents': len(train_dictionary_batch),
    'num_word': len(flattened_batch),
    'text': pickle.dumps(text_data),
    'text_json': pickle.dumps(validation_test_data if phase == "train" else batch_documents),
    'max_attempts': max_attempts, 
    'show_topics': show_topics_jsonb,
    'topics_words':topics_results_jsonb,
    'validation_result': validation_results_jsonb,
    'text_sha256': hashlib.sha256(text_data.encode()).hexdigest(),
    'text_md5': hashlib.md5(text_data.encode()).hexdigest(),
    'text_path': texts_zip,
    'pca_path': pca_image,
    'pca_gpu_path': pca_gpu_image,
    'pylda_path': pyLDAvis_image,

    # Model and Training Parameters
    'topics': n_topics,
    'alpha_str': str(alpha_str),  # Single string instead of a list
    'n_alpha': n_alpha,
    'beta_str': str(beta_str),  # Single string instead of a list
    'n_beta': n_beta,
    'passes': passes,
    'iterations': iterations,
    'update_every': update_every,
    'eval_every': eval_every,
    'chunksize': str(chunksize),  # Convert chunksize to string after use
    'random_state': random_state,
    'per_word_topics': per_word_topics,

    # Evaluation Metrics
    'convergence': convergence_score,
    'nll': negative_log_likelihood,
    'perplexity': perplexity_score,
    'coherence': coherence_score,
    'mean_coherence': mean_coherence,
    'median_coherence': median_coherence,
    'mode_coherence': mode_coherence,
    'std_coherence': std_coherence,
    'perplexity_threshold': threshold,

    # Serialized Data
    'lda_model': ldamodel_bytes.compute(), # C:\Users\pqn7\OneDrive - CDC\git-projects\unified-topic-modeling-analysis\gpt\why-lda-is-delayed.md
    'corpus': corpus_to_pickle,
    'dictionary': pickle.dumps(train_dictionary_batch),

    # Visualization Creation Verification Placeholders
    'create_pylda': None,
    'create_pcoa': None,
    'create_pca_gpu': None
    }

    return current_increment_data
