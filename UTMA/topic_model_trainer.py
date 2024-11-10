# train_eval_topic_model.py - Adaptive Topic Modeling and Parallel Processing for SLIF
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This script manages model training, evaluation, and metadata generation for topic modeling tasks within the
# Unified Topic Modeling and Analysis (UTMA). It leverages Dask for distributed, adaptive parallel processing
# to efficiently handle large-scale data and variable resource availability. This includes tracking the dynamic
# allocation of cores used in each batch, adaptive scaling, and comprehensive metadata storage for reproducibility.
#
# Functions:
# - Trains and evaluates LDA models for topic modeling based on dynamic, adaptive resource allocation
# - Tracks batch-specific metadata, including dynamic core count, model parameters, and evaluation scores
# - Manages parallelized workflows and efficient data processing using Dask's Client and LocalCluster
#
# Dependencies:
# - Python libraries: pandas, logging, pickle, hashlib, math, numpy, json, typing
# - Dask libraries: distributed
# - Gensim library for LDA modeling and coherence scoring
#
# Developed with AI assistance.


import pandas as pd  # Used to handle timestamps and date formatting for logging and metadata.
import dask
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
from .mathstats import calculate_perplexity_threshold, coherence_score_decision, replace_nan_with_high_precision, calculate_perplexity


    
# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model_v2(n_topics: int, alpha_str: Union[str, float], beta_str: Union[str, float], train_data: list, data: list, phase: str,
                   random_state: int, passes: int, iterations: int, update_every: int, eval_every: int, cores: int,
                   per_word_topics: bool, ldamodel=None, **kwargs):

    time_of_method_call = pd.to_datetime('now')  # Record the current timestamp for logging and metadata.

    coherence_score_list = []  # Initialize a list to store coherence scores for evaluation.

    # Initialize a dictionary to hold the corpus data for each phase
    corpus_data = {
        "train": [],
        "validation": [],
        "test": []
    }

    try:
        # Compute the Dask collections in `data`, resolving all delayed computations at once.
        train_batch_documents = dask.compute(*train_data)
        batch_documents = dask.compute(*data)
        # Set a chunksize for model processing, dividing documents into smaller groups for efficient processing.
        chunksize = max(1, int(len(train_batch_documents) // 5))
    except Exception as e:
        logging.error(f"Error computing streaming_documents data: {e}")  # Log any errors during Dask computation.
        raise  # Re-raise the exception to stop execution if data computation fails.

    # Create a Gensim dictionary from the batch documents, mapping words to unique IDs for the corpus.
    try:
        train_dictionary_batch = Dictionary(list(train_batch_documents))
    except TypeError:
        print("Error: The data structure is not correct to create the Dictionary object.")  # Print an error if data format is incompatible.


    # Corrected code inside train_model_v2
    number_of_documents = 0  # Counter for tracking the number of documents processed.

    if phase == "train":
        # Flatten the list of documents, converting each sublist of tokens into a single list for metadata.
        flattened_batch = [item for sublist in train_batch_documents for item in sublist]
        for doc_tokens in train_batch_documents:
            bow_out = train_dictionary_batch.doc2bow(doc_tokens)  # Convert tokens to BoW format using training dictionary
            corpus_data['train'].append(bow_out)  # Add the bag-of-words representation to the training corpus batch
            number_of_documents += 1  # Increment the document counter
    else:
        # Flatten the list of documents, converting each sublist of tokens into a single list for metadata.
        flattened_batch = [item for sublist in batch_documents for item in sublist]
        for doc_tokens in batch_documents:
            bow_out = train_dictionary_batch.doc2bow(doc_tokens)  # Convert tokens to BoW format using training dictionary
            corpus_data[phase].append(bow_out)  # Append the bag-of-words representation to the appropriate phase corpus
            number_of_documents += 1  # Increment the document counter

    logging.info(f"There was a total of {number_of_documents} documents added to the corpus_data.")  # Log document count.

    # Calculate numeric values for alpha and beta, using custom functions based on input strings or values.
    n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
    n_beta = calculate_numeric_beta(beta_str, n_topics)

    # Updated default score as a high-precision Decimal value
    DEFAULT_SCORE = Decimal('0')

    if phase in ['validation', 'test']:
        # For validation and test phases, no model is created
        ldamodel_bytes = pickle.dumps(ldamodel)
        coherence_score = convergence_score = negative_log_likelihood = DEFAULT_SCORE

    elif phase == "train":
        try:
            # Create and train the LdaModel for the training phase
            ldamodel = LdaModel(
                corpus=corpus_data["train"],
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
            threshold = calculate_perplexity_threshold(ldamodel, corpus_data["train"])
            ldamodel_bytes = pickle.dumps(ldamodel)
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")
            raise

    else:
        # For validation and test phases, use the already-trained model
        ldamodel_bytes = pickle.dumps(ldamodel)
    # Calculate negative_log_likelihood and convergence for each phase

    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate coherence metrics across all phases
        coherence_score, mean_coherence, median_coherence, mode_coherence, std_coherence, threshold = coherence_score_decision(
            ldamodel, batch_documents if phase != "train" else train_batch_documents, train_dictionary_batch, initial_sample_ratio=0.1
        )

        # Calculate full coherence score without threshold check for debugging
        coherence_model_lda = CoherenceModel(
            model=ldamodel, dictionary=train_dictionary_batch, texts=train_batch_documents if phase == "train" else batch_documents, coherence='c_v',
            processes=math.floor(cores * (1/3))
        )
        coherence_score = coherence_model_lda.get_coherence()

        try:
            convergence_score = ldamodel.bound(corpus_data[phase])
        except Exception as e:
            logging.error(f"Issue calculating convergence score: {e}. Value '{DEFAULT_SCORE}' assigned.")
            convergence_score = DEFAULT_SCORE

        try:
            negative_log_likelihood = ldamodel.log_perplexity(corpus_data[phase])
                
            # Calculate the total number of words for perplexity calculation
            num_words = sum(sum(count for _, count in doc) for doc in corpus_data[phase])
                
            # Calculate perplexity using the calculate_perplexity function
            perplexity_score = calculate_perplexity(negative_log_likelihood, num_words)
        except RuntimeWarning as e:
            logging.info(f"Issue calculating perplexity score: {e}. Value '{DEFAULT_SCORE}' assigned.")
            negative_log_likelihood = perplexity_score = DEFAULT_SCORE



    # Set the number of words to display for each topic, allowing deeper insight into topic composition.
    num_words = 30  # Adjust based on the level of detail required for topic terms.
    
    # Retrieve the top words for each topic with their probabilities. This provides the most relevant words defining each topic.
    try:
        show_topics_results = ldamodel.show_topics(num_topics=-1, num_words=num_words, formatted=False)
        topics_to_store = [
            {
                "method": "show_topics",
                "topic_id": topic[0],
                "words": [{"word": word, "prob": prob} for word, prob in topic[1]]
            }
            for topic in show_topics_results
        ]
    except Exception as e:
        # Fallback values with diagnostic information
        topics_to_store = [
            {
                "method": "show_topics",
                "topic_id": None,  # Set to None since topic_id is unavailable
                "words": [],       # Empty list for words since there was a failure
                "error": str(e),   # Store the exception message
                "record_id": kwargs.get("record_id", "unknown"),  # ID or key to identify the record
                "parameters": {     # Add any parameters that were relevant to the failure
                    "num_topics": kwargs.get("num_topics", "unknown"),
                    "num_words": kwargs.get("num_words", "unknown"),
                    "alpha": kwargs.get("alpha", "unknown"),
                    "beta": kwargs.get("beta", "unknown"),
                }
            }
        ]

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility.
        topics_to_store = convert_float32_to_float(topics_to_store)
        # Serialize the topic data to JSON format for structured storage, such as in a database.
        show_topics_jsonb = json.dumps(topics_to_store)
        
        # Get the topic distribution for each document in the validation or test corpus
        try:
            if phase in ['validation', 'test']:
                validation_results_to_store = [
                    ldamodel.get_document_topics(bow_doc, minimum_probability=0.01)
                    for bow_doc in corpus_data[phase]
                ]
            else: 
                validation_results_to_store = ['N/A']
        except Exception as e:
            logging.error(f"The validation data could not be generated: {e}")
            validation_results_to_store = {"error": "Validation data generation failed", "phase": phase}

        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility
        validation_results_to_store = convert_float32_to_float(validation_results_to_store)

        # Serialize the validation data to JSON format for structured storage
        try:
            validation_results_jsonb = json.dumps(validation_results_to_store)
        except TypeError as e:
            logging.error(f"JSON serialization failed due to non-compatible types: {e}")
            validation_results_jsonb = json.dumps({"error": "Validation data generation failed", "phase": phase})

        try:
            if phase == "train":
            # Retrieve the top topics based on coherence scores, which assess how interpretable or meaningful each topic is.
                topics = ldamodel.top_topics(texts=train_batch_documents, processes=math.floor(cores * (2/3)))
            else:
                topics = ldamodel.top_topics(texts=batch_documents, processes=math.floor(cores * (2/3)))

            # Extract only the words from each topic, removing the associated scores to provide a simplified view of topic terms.
            topic_words = []
            for topic in topics:
                topic_representation = topic[0]  # Access the list of top words with their coherence-based ordering.
                words = [word for _, word in topic_representation]  # Isolate just the words for easy reference.
                topic_words.append(words)  # Store the list of words for each topic.
        except Exception as e:
            logging.error(f"An error occurred while extracting topic words: {e}")
            # If topics extraction fails, provide a single "N/A" list as a fallback
            topic_words = [["N/A"]]  # Fallback with one list containing "N/A" to indicate failure

        # Ensure all numerical values are in a JSON-compatible format for downstream compatibility.
        topics_to_store = convert_float32_to_float(topic_words)
        topic_words_jsonb = json.dumps(topic_words)  # Serializes to JSON format

    # Generate unique time-based key with document text hash
    time_of_method_call = datetime.now()
    time_hash = time_of_method_call.strftime('%Y%m%d%H%M%S%f')
    random_suffix = f"{random.randint(100, 999)}"
    unique_time_key = hashlib.md5((time_hash + random_suffix).encode()).hexdigest()
    text_hash = hashlib.md5(' '.join(flattened_batch).encode()).hexdigest()

    # Concatenate document hash and unique time hash to form the final key
    string_time = (text_hash + unique_time_key).strip()  # `strip()` is optional here
    # When creating time_key
    logging.info(f"Generated time_key: {string_time}")

    current_increment_data = {
    # Metadata and Identifiers
    'time_key': string_time,  # Unique identifier for this batch, based on concatenated hash.
    'type': phase,  # Indicates whether this batch is for 'train', 'validation', or 'test'.
    'start_time': time_of_method_call,  # Start time of this batch's processing.
    'end_time': pd.to_datetime('now'),  # End time of this batch's processing.
    'num_workers': float('nan'),  # Placeholder for adaptive core count used for this batch.
    
    # Document and Batch Details
    'batch_size': len(batch_documents),  # Number of documents processed in this batch.
    'num_documents': float('nan'),  # Placeholder for the total document count.
    'text': pickle.dumps([' '.join(flattened_batch)]),  # Concatenated text of the batch for metadata/logging.
    'text_json': pickle.dumps(batch_documents),  # Serialized batch documents for reference.
    'show_topics': show_topics_jsonb, # Serialized top terms per topic for analysis
    'top_words': topic_words_jsonb, # Serialized most coherent words across topics for comparative analysis
    'validation_result': validation_results_jsonb, 
    'text_sha256': hashlib.sha256(' '.join(flattened_batch).encode()).hexdigest(),  # SHA-256 hash of text for integrity.
    'text_md5': hashlib.md5(' '.join(flattened_batch).encode()).hexdigest(),  # MD5 hash for quick lookups.
    
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
    'corpus': pickle.dumps(corpus_data[phase]),  # Serialized corpus used for training.
    'dictionary': pickle.dumps(train_dictionary_batch),  # Serialized dictionary for topic modeling.
    
    # Visualization Creation Verification Placeholders
    'create_pylda': None,  # Placeholder for pyLDA verification of visualization creation.
    'create_pcoa': None  # Placeholder for PCoA verification of visualization creation.
    }

  # Use `replace_nan_with_high_precision` to handle any NaN values with calculated replacements
    metrics_data = replace_nan_with_high_precision(default_score=DEFAULT_SCORE, data=current_increment_data)  # Replace `default_score=nan` with your desired default if needed
    return metrics_data
