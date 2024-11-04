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

from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta  # Functions that calculate alpha and beta values for LDA.
from .utils import convert_float32_to_float  # Utility function for data type conversion, ensuring compatibility within the script.

    
# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model_v2(n_topics: int, alpha_str: Union[str, float], beta_str: Union[str, float], data: list, phase: str,
                   random_state: int, passes: int, iterations: int, update_every: int, eval_every: int, cores: int,
                   per_word_topics: bool, ldamodel=None):

    coherence_score_list = []  # Initialize a list to store coherence scores for evaluation.
    corpus_batch = []  # Initialize an empty list to hold the corpus in bag-of-words format.
    time_of_method_call = pd.to_datetime('now')  # Record the current timestamp for logging and metadata.

    try:
        # Compute the Dask collections in `data`, resolving all delayed computations at once.
        batch_documents = dask.compute(*data)
        # Set a chunksize for model processing, dividing documents into smaller groups for efficient processing.
        chunksize = max(1, int(len(batch_documents) // 5))
    except Exception as e:
        logging.error(f"Error computing streaming_documents data: {e}")  # Log any errors during Dask computation.
        raise  # Re-raise the exception to stop execution if data computation fails.

    # Create a Gensim dictionary from the batch documents, mapping words to unique IDs for the corpus.
    try:
        dictionary_batch = Dictionary(list(batch_documents))
    except TypeError:
        print("Error: The data structure is not correct.")  # Print an error if data format is incompatible.

    # Flatten the list of documents, converting each sublist of tokens into a single list for metadata.
    flattened_batch = [item for sublist in batch_documents for item in sublist]

    # Convert each document to bag-of-words format (list of word IDs and counts), creating a corpus.
    number_of_documents = 0  # Counter for tracking the number of documents processed.
    for doc_tokens in batch_documents:
        bow_out = dictionary_batch.doc2bow(doc_tokens)  # Convert tokens to bag-of-words format.
        corpus_batch.append(bow_out)  # Add the bag-of-words representation to the corpus batch.
        number_of_documents += 1  # Increment the document counter.

    logging.info(f"There was a total of {number_of_documents} documents added to the corpus_batch.")  # Log document count.

    # Calculate numeric values for alpha and beta, using custom functions based on input strings or values.
    n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
    n_beta = calculate_numeric_beta(beta_str, n_topics)


    # Only create and train the LdaModel if phase is "train"
    if phase == "train":
        try:
            # Initialize the LdaModel with specified parameters, using the prepared corpus and dictionary.
            ldamodel = LdaModel(
                corpus=corpus_batch,  # Bag-of-words corpus representation for training.
                id2word=dictionary_batch,  # Dictionary that maps word IDs to words.
                num_topics=n_topics,  # Number of topics to be identified.
                alpha=float(n_alpha),  # Alpha parameter controlling topic sparsity.
                eta=float(n_beta),  # Eta parameter affecting word distribution in topics.
                random_state=random_state,  # Random seed for reproducibility.
                passes=passes,  # Number of full passes through the corpus.
                iterations=iterations,  # Maximum number of iterations in each pass.
                update_every=update_every,  # Number of documents to accumulate before updating.
                eval_every=eval_every,  # Evaluate model every n updates (for large datasets).
                chunksize=chunksize,  # Number of documents processed at once.
                per_word_topics=True  # Flag to compute topics for each word.
            )
            ldamodel_bytes = pickle.dumps(ldamodel)  # Serialize the trained model for later use and storage.
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")  # Log any errors in training.
            raise  # Stop execution if model creation fails.

        # Calculate scores for training phase
        try:
            # Compute coherence score, assessing topic interpretability on a subset of processes.
            coherence_model_lda = CoherenceModel(
                model=ldamodel,  # Trained LDA model.
                processes=math.floor(cores * (1/3)),  # Use a third of available cores for evaluation.
                dictionary=dictionary_batch,  # Dictionary used for coherence computation.
                texts=batch_documents,  # Original texts for calculating coherence.
                coherence='c_v'  # Measure of how coherent the topics are.
            )
            coherence_score = coherence_model_lda.get_coherence()  # Get coherence score for topics.
            coherence_score_list.append(coherence_score)  # Store the score for logging or analysis.
        except Exception as e:
            logging.error("Issue calculating coherence score. Value '-Inf' assigned.")  # Log error in coherence calculation.
            coherence_score = float('-inf')  # Assign default score if calculation fails.
            coherence_score_list.append(coherence_score)

        try:
            convergence_score = ldamodel.bound(corpus_batch)  # Calculate model convergence score.
        except Exception as e:
            logging.error("Issue calculating convergence score. Value '-Inf' assigned.")  # Log error in convergence score.
            convergence_score = float('-inf')  # Assign default score if calculation fails.

        try:
            perplexity_score = ldamodel.log_perplexity(corpus_batch)  # Calculate model perplexity score.
        except RuntimeWarning as e:
            logging.info("Issue calculating perplexity score. Value '-Inf' assigned.")  # Log any warnings in perplexity calculation.
            perplexity_score = float('-inf')  # Assign default score if calculation fails.
    elif phase in ['validation', 'test']:
        # For validation and test phases, no model is created
        ldamodel_bytes = None  # No model is serialized or stored.
        coherence_score = float('-inf')  # Default coherence score for non-training phases.
        convergence_score = float('-inf')  # Default convergence score for non-training phases.
        perplexity_score = float('-inf')  # Default perplexity score for non-training phases.
  


    # Set the number of words to display for each topic, allowing deeper insight into topic composition.
    num_words = 30  # Adjust based on the level of detail required for topic terms.
    
    # Retrieve the top words for each topic with their probabilities. This provides the most relevant words defining each topic.
    show_topics_results = ldamodel.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    
    # Format the top words per topic as a list of dictionaries, making the data easier to store and access.
    topics_to_store = [
        {
            "method": "show_topics",           # Indicates the method used for topic representation
            "topic_id": topic[0],              # Unique identifier for each topic
            "words": [{"word": word, "prob": prob} for word, prob in topic[1]]  # Words with their probabilities for this topic
        }
        for topic in show_topics_results
    ]

    # Ensure all numerical values are in a JSON-compatible format for downstream compatibility.
    topics_to_store = convert_float32_to_float(topics_to_store)

    # Serialize the topic data to JSON format for structured storage, such as in a database.
    show_topics_jsonb = json.dumps(topics_to_store)

    # Retrieve the top topics based on coherence scores, which assess how interpretable or meaningful each topic is.
    topics = ldamodel.top_topics(texts=batch_documents, processes=math.floor(cores * (1/3)))
    
    # Extract only the words from each topic, removing the associated scores to provide a simplified view of topic terms.
    topic_words = []
    for topic in topics:
        topic_representation = topic[0]  # Access the list of top words with their coherence-based ordering.
        words = [word for _, word in topic_representation]  # Isolate just the words for easy reference.
        topic_words.append(words)  # Store the list of words for each topic.


    # Generate unique time-based hashes for identifying and logging model runs.
    time_hash = hashlib.md5(time_of_method_call.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()  # Hash based on start time.
    text_hash = hashlib.md5(pd.to_datetime('now').strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()  # Hash based on current time.
    string_time = text_hash.strip() + time_hash.strip()  # Concatenate hashes for unique time key.

    current_increment_data = {
    # Metadata and Identifiers
    'time_key': string_time,  # Unique identifier for this batch, based on concatenated hash.
    'type': phase,  # Indicates whether this batch is for 'train', 'validation', or 'test'.
    'time': time_of_method_call,  # Start time of this batch's processing.
    'end_time': pd.to_datetime('now'),  # End time of this batch's processing.
    'num_workers': -1, #len(client.scheduler_info()["workers"]),  # Adaptive core count used for this batch.
    
    # Document and Batch Details
    'batch_size': len(batch_documents),  # Number of documents processed in this batch.
    'num_documents': float('-inf'),  # Placeholder for the total document count.
    'text': [' '.join(flattened_batch)],  # Concatenated text of the batch for metadata/logging.
    'text_json': pickle.dumps(batch_documents),  # Serialized batch documents for reference.
    'show_topics': show_topics_jsonb, # Serialized top terms per topic for analysis
    'top_words': topic_words, # Most coherent words across topics for comparative analysis
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
    'perplexity': perplexity_score,  # Perplexity score to assess model fit.
    'coherence': coherence_score,  # Coherence score to measure topic interpretability.
    
    # Serialized Data
    'lda_model': ldamodel_bytes,  # Serialized LDA model, if trained in this batch.
    'corpus': pickle.dumps(corpus_batch),  # Serialized corpus used for training.
    'dictionary': pickle.dumps(dictionary_batch),  # Serialized dictionary for topic modeling.
    
    # Visualization Placeholders
    'create_pylda': None,  # Placeholder for pyLDA visualization.
    'create_pcoa': None  # Placeholder for PCoA visualization.
    }

    return current_increment_data
