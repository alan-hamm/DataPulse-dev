# model.py - Topic Modeling and Parallel Processing for SLIF
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This script handles model training and distributed parallel processing for topic analysis tasks within the
# Scalable LDA Insights Framework (SLIF). It leverages Dask for managing distributed computations and provides
# utilities for processing large-scale data efficiently.
#
# Functions:
# - Manages parallelized workflows and data processing for topic modeling
# - Utilizes Dask's Client and LocalCluster for scalable computing
#
# Dependencies:
# - Python libraries: pandas, logging
# - Dask libraries: distributed, diagnostics
#
# Developed with AI assistance.

import pandas as pd
from dask.distributed import as_completed
import dask   # Parallel computing library that scales Python workflows across multiple cores or machines 
from dask.distributed import Client, LocalCluster, wait   # Distributed computing framework that extends Dask functionality 
from dask.diagnostics import ProgressBar   # Visualizes progress of Dask computations
from dask.distributed import progress
from distributed import Future
from dask.delayed import Delayed # Decorator for creating delayed objects in Dask computations
#from dask.distributed import as_completed
from dask.bag import Bag
from dask import delayed
from dask import persist
import dask.config
from dask.distributed import performance_report, wait, as_completed #,print
from distributed import get_worker
import logging
from gensim.models import LdaModel  # Implements LDA for topic modeling using the Gensim library
from gensim.corpora import Dictionary  # Represents a collection of text documents as a bag-of-words corpus
from gensim.models import CoherenceModel  # Computes coherence scores for topic models
import pickle
import math
import hashlib
import numpy as np
import json
from typing import Union

from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta
from .utils import convert_float32_to_float
    
# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model_v2(n_topics: int, alpha_str: Union[str, float], beta_str: Union[str, float], data: list, phase: str,
                   random_state: int, passes: int, iterations: int, update_every: int, eval_every: int, cores: int,
                   per_word_topics: bool, ldamodel=None):
    models_data = []
    coherence_score_list = []
    corpus_batch = []
    time_of_method_call = pd.to_datetime('now')

    try:
        # Compute several dask collections at once
        streaming_documents = dask.compute(*data)
        chunksize = max(1, int(len(streaming_documents) // 5))
    except Exception as e:
        logging.error(f"Error computing streaming_documents data: {e}")
        raise

    batch_documents = streaming_documents

    # Create a new Gensim Dictionary for the current batch
    try:
        dictionary_batch = Dictionary(list(batch_documents))
    except TypeError:
        print("Error: The data structure is not correct.")

    flattened_batch = [item for sublist in batch_documents for item in sublist]

    # Create corpus for training
    number_of_documents = 0
    for doc_tokens in batch_documents:
        bow_out = dictionary_batch.doc2bow(doc_tokens)
        corpus_batch.append(bow_out)
        number_of_documents += 1

    logging.info(f"There was a total of {number_of_documents} documents added to the corpus_batch.")

    n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
    n_beta = calculate_numeric_beta(beta_str, n_topics)

    # Only create and train the LdaModel if phase is "train"
    if phase == "train":
        try:
            ldamodel = LdaModel(
                corpus=corpus_batch,
                id2word=dictionary_batch,
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
            ldamodel_bytes = pickle.dumps(ldamodel)
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")
            raise

        # Calculate scores for training phase
        try:
            coherence_model_lda = CoherenceModel(model=ldamodel, processes=math.floor(cores * (1/3)),
                                                 dictionary=dictionary_batch, texts=batch_documents, coherence='c_v')
            coherence_score = coherence_model_lda.get_coherence()
            coherence_score_list.append(coherence_score)
        except Exception as e:
            logging.error("Issue calculating coherence score. Value '-Inf' assigned.")
            coherence_score = float('-inf')
            coherence_score_list.append(coherence_score)

        try:
            convergence_score = ldamodel.bound(corpus_batch)
        except Exception as e:
            logging.error("Issue calculating convergence score. Value '-Inf' assigned.")
            convergence_score = float('-inf')

        try:
            perplexity_score = ldamodel.log_perplexity(corpus_batch)
        except RuntimeWarning as e:
            logging.info("Issue calculating perplexity score. Value '-Inf' assigned.")
            perplexity_score = float('-inf')
    else:
        # For validation and test phases, no model is created
        ldamodel_bytes = None  # No model is serialized
        coherence_score = float('-inf')
        convergence_score = float('-inf')
        perplexity_score = float('-inf')

    # Other parts of your code for formatting results and generating keys
    time_hash = hashlib.md5(time_of_method_call.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
    text_hash = hashlib.md5(pd.to_datetime('now').strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
    string_time = text_hash.strip() + time_hash.strip()

    current_increment_data = {
        'time_key': string_time,
        'type': phase,
        'num_workers': 0,  # Placeholder; update later
        'batch_size': -1,
        'num_documents': -1,
        'text': [' '.join(flattened_batch)],
        'text_json': pickle.dumps(batch_documents),
        'text_sha256': hashlib.sha256(' '.join(flattened_batch).encode()).hexdigest(),
        'text_md5': hashlib.md5(' '.join(flattened_batch).encode()).hexdigest(),
        'convergence': convergence_score,
        'perplexity': perplexity_score,
        'coherence': coherence_score,
        'topics': n_topics,
        'alpha_str': [str(alpha_str)],
        'n_alpha': n_alpha,
        'beta_str': [str(beta_str)],
        'n_beta': n_beta,
        'passes': passes,
        'iterations': iterations,
        'update_every': update_every,
        'eval_every': eval_every,
        'chunksize': chunksize,
        'random_state': random_state,
        'per_word_topics': per_word_topics,
        'show_topics': None,  # Skip topic storage for validation/test
        'lda_model': ldamodel_bytes,
        'corpus': pickle.dumps(corpus_batch),
        'dictionary': pickle.dumps(dictionary_batch),
        'create_pylda': None,
        'create_pcoa': None,
        'time': time_of_method_call,
        'end_time': pd.to_datetime('now')
    }

    return current_increment_data
