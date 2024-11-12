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


import os
import pandas as pd  # Used to handle timestamps and date formatting for logging and metadata.
import dask
from dask import delayed
from dask.distributed import get_client
from dask.distributed import wait  # Manages asynchronous execution in distributed settings, ensuring all futures are completed.
import logging  # Provides error logging and information tracking throughout the script's execution.

from gensim.models import LdaModel  # Implements Latent Dirichlet Allocation (LDA) for topic modeling.
from gensim.corpora import Dictionary  # Converts tokenized text data into a bag-of-words format for LDA.
from gensim.models import CoherenceModel  # Evaluates topic model coherence to measure topic interpretability.

import pickle  # Serializes models and data structures to store results or share between processes.
import math  # Supports mathematical calculations, such as computing fractional core usage for parallel processing.
import hashlib  # Generates unique hashes for document metadata, ensuring data consistency.
import numpy as np  # Enables numerical operations, potentially for data manipulation or vector operations.
import json  # Provides JSON encoding and decoding, useful for handling data in a structured format.
from typing import Union  # Allows type hinting for function parameters, improving code readability and debugging.
import random
from datetime import datetime
from decimal import Decimal, InvalidOperation

from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta  # Functions that calculate alpha and beta values for LDA.
from .utils import convert_float32_to_float  # Utility function for data type conversion, ensuring compatibility within the script.
from .mathstats import *
from .visualization import create_vis_pca

    
# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model_v2(n_topics: int, alpha_str: Union[str, float], beta_str: Union[str, float], zip_path:str, pylda_path:str, pca_path:str, train_dictionary: Dictionary, validation_test_data: list, phase: str,
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
        batch_documents = dask.compute(*validation_test_data)
        # Set a chunksize for model processing, dividing documents into smaller groups for efficient processing.
        chunksize = max(1, int(len(batch_documents) // 5))
    except Exception as e:
        logging.error(f"Error computing streaming_documents data: {e}")  # Log any errors during Dask computation.
        raise  # Re-raise the exception to stop execution if data computation fails.

    # Check for extra nesting in batch_documents and flatten if necessary
    if len(batch_documents) == 1 and isinstance(batch_documents[0], list):
        batch_documents = batch_documents[0]   

    # Create a Gensim dictionary from the batch documents, mapping words to unique IDs for the corpus.
    try:
        train_dictionary_batch = train_dictionary
    except TypeError as e:
        print("Error: The data structure is not correct to create the Dictionary object.")  # Print an error if data format is incompatible.
        print(f"Details: {e}")

    # Corrected code inside train_model_v2
    number_of_documents = 0  # Counter for tracking the number of documents processed.

    flattened_batch = []
    # Flatten the list of documents, converting each sublist of tokens into a single list for metadata.
    flattened_batch = [item for sublist in batch_documents for item in sublist]
    for doc_tokens in batch_documents:
        bow_out = train_dictionary_batch.doc2bow(doc_tokens)  # Convert tokens to BoW format using training dictionary
        corpus_data[phase].append(bow_out)  # Append the bag-of-words representation to the appropriate phase corpus
        number_of_documents += 1  # Increment the document counter
    corpus_to_pickle = corpus_data[phase]

    logging.info(f"There was a total of {number_of_documents} documents added to the corpus_data.")  # Log document count.

    # Calculate numeric values for alpha and beta, using custom functions based on input strings or values.
    n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
    n_beta = calculate_numeric_beta(beta_str, n_topics)

    # Updated default score as a high-precision Decimal value
    DEFAULT_SCORE = Decimal('0')

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
            # Calculate negative_log_likelihood threshold using the BoW corpus
            threshold = delayed(calculate_perplexity_threshold)(ldamodel, corpus_data["train"], DEFAULT_SCORE)
            # Serialize the model as a delayed task
            ldamodel_bytes = delayed(pickle.dumps)(ldamodel)
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")
            raise

    else:
        # For validation and test phases, use the already-trained model
        #ldamodel_bytes = pickle.dumps(ldamodel)

        try:
            if phase == "validation":
                # Create the delayed task for the threshold without computing it immediately
                threshold = dask.delayed(calculate_perplexity_threshold)(ldamodel, corpus_data["validation"], DEFAULT_SCORE)
            elif phase == "test":
                # Create the delayed task for the threshold without computing it immediately
                threshold = dask.delayed(calculate_perplexity_threshold)(ldamodel, corpus_data["test"], DEFAULT_SCORE)
        except Exception as e:
            logging.warning(f"Perplexity threshold calculation failed. Using default score: {DEFAULT_SCORE}")
            # Create a delayed fallback task for the default score
            threshold = dask.delayed(lambda: DEFAULT_SCORE)()

    # Calculate coherence, convergence, and other metrics
    # Coherence, convergence, and other metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            # Create a delayed task for coherence score calculation without computing it immediately
            coherence_task = dask.delayed(compute_full_coherence_score)(
                ldamodel, train_dictionary_batch, batch_documents, cores
            )
        except Exception as e:
            logging.warning("Coherence score calculation failed. Using default score.")
            # Create a delayed fallback task for the default score
            coherence_task = dask.delayed(lambda: DEFAULT_SCORE)()

        try:
            # Create a delayed task for sample coherence scores calculation
            coherence_scores_task = dask.delayed(sample_coherence_for_phase)(
                ldamodel, batch_documents, train_dictionary_batch, sample_ratio=0.1
            )
            
            # Process coherence scores with high precision after computation, including the tolerance parameter
            coherence_scores_data = dask.delayed(replace_nan_with_high_precision)(
                DEFAULT_SCORE, {'coherence_scores': coherence_scores_task}, tolerance=1e-5  # Example tolerance value
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
            logging.warning("Sample coherence scores calculation failed. Using default scores.")
            # Assign fallback values directly
            mean_coherence = median_coherence = std_coherence = mode_coherence = DEFAULT_SCORE

            

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
    num_words = 30  # Adjust based on the level of detail required for topic terms.
    
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

    # Retrieve results later by calling .compute()
    topics_to_store = topics_to_store_task.compute()

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility.
        topics_to_store = convert_float32_to_float(topics_to_store)
        # Serialize the topic data to JSON format for structured storage, such as in a database.
        show_topics_jsonb = json.dumps(topics_to_store)

        try:
            # Create a delayed task for document topic extraction
            validation_results_task_delayed = dask.delayed(lambda: [
                ldamodel.get_document_topics(bow_doc, minimum_probability=0.01) for bow_doc in corpus_data[phase]
            ])()
        except Exception as e:
            # Create a delayed error handling task
            validation_results_task_delayed = dask.delayed(lambda: {
                "error": "Validation data generation failed",
                "phase": phase
            })()


        # Create a delayed task without computing it immediately
        validation_results_task = dask.delayed(validation_results_task_delayed)

        # Later in the code, retrieve the results by calling .compute()
        validation_results_to_store = validation_results_task.compute()

        # Log the computed structure before conversion for debugging purposes
        logging.debug(f"Validation results before type conversion: {validation_results_to_store}")
        
        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility
        validation_results_to_store = convert_float32_to_float(validation_results_to_store)

        # Serialize the validation data to JSON format for structured storage
        try:
            validation_results_jsonb = json.dumps(validation_results_to_store)
        except TypeError as e:
            logging.error(f"JSON serialization failed due to non-compatible types: {e}")
            # Log the problematic types for debugging
            for item in validation_results_to_store:
                if isinstance(item, dict):
                    for key, value in item.items():
                        logging.error(f"Type of {key}: {type(value)}")
            validation_results_jsonb = json.dumps({"error": "Validation data generation failed", "phase": phase})
                    

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
        topic_words_jsonb = json.dumps(topic_words)  # Serializes to JSON format


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
    'primary_key': unique_primary_key,  # Unique identifier for this batch, based on concatenated hash.
    'type': phase,  # Indicates whether this batch is for 'train', 'validation', or 'test'.
    'start_time': time_of_method_call,  # Start time of this batch's processing.
    'end_time': pd.to_datetime('now'),  # End time of this batch's processing.
    'num_workers': float('nan'),  # Placeholder for adaptive core count used for this batch.
    
    # Document and Batch Details
    'batch_size': batch_size,  # Number of documents in the current training batch
    'num_documents': len(train_dictionary_batch) if phase == "train" else float('nan'),
    'text': pickle.dumps(text_data),
    'text_json': pickle.dumps(validation_test_data if phase == "train" else batch_documents),
    'show_topics': show_topics_jsonb,
    'top_words': topic_words_jsonb,
    'validation_result': validation_results_jsonb,
    'text_sha256': hashlib.sha256(text_data.encode()).hexdigest(),
    'text_md5': hashlib.md5(text_data.encode()).hexdigest(),
    'text_path': texts_zip,
    'pca_path': pca_image,
    'pylda_path': pyLDAvis_image,
    
    # Model and Training Parameters
    'topics': n_topics,  # Number of topics generated in the model.
    'alpha_str': [str(alpha_str)],  # Alpha parameter as a string, before conversion.
    'n_alpha': n_alpha,  # Numeric value of the alpha parameter.
    'beta_str': [str(beta_str)],  # Beta parameter as a string, before conversion.
    'n_beta': n_beta,  # Numeric value of the beta parameter.
    'passes': passes,  # Number of passes through the corpus.
    'iterations': iterations,  # Number of iterations within each pass.
    'update_every': update_every,  # Number of documents before each model update.
    'eval_every': eval_every,  # Frequency of model evaluation.
    'chunksize': chunksize,  # Number of documents processed in each chunk.
    'random_state': random_state,  # Random seed for reproducibility.
    'per_word_topics': per_word_topics,  # Boolean flag for per-word topics.
    
    # Evaluation Metrics
    'convergence': convergence_score,  # Convergence score for evaluating model stability.
    'nll': negative_log_likelihood,  # negative log likelihood
    'perplexity': perplexity_score, # Perplexity score to assess model fit.
    'coherence': coherence_score,  # Coherence score to measure topic interpretability.
    'mean_coherence': mean_coherence,
    'median_coherence': median_coherence,
    'mode_coherence': mode_coherence,
    'std_coherence': std_coherence,
    'threshold': threshold,
    
    # Serialized Data
    'lda_model': ldamodel_bytes,  # Serialized LDA model, if trained in this batch.
    'corpus': pickle.dumps(corpus_to_pickle),  # Serialized corpus used for training.
    'dictionary': pickle.dumps(train_dictionary_batch),  # Serialized dictionary for topic modeling.
    
    # Visualization Creation Verification Placeholders
    'create_pylda': None,  # Placeholder for pyLDA verification of visualization creation.
    'create_pcoa': None  # Placeholder for PCoA verification of visualization creation.
    }

  # Use `replace_nan_with_high_precision` to handle any NaN values with calculated replacements
    metrics_data = replace_nan_with_high_precision(default_score=DEFAULT_SCORE, data=current_increment_data)  # Replace `default_score=nan` with your desired default if needed
    return metrics_data
