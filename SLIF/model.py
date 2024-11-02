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

from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta
from .utils import convert_float32_to_float
    
# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model(n_topics: int, alpha_str: list, beta_str: list, data: list, train_eval: str, 
                random_state: int, passes: int, iterations: int, update_every: int, eval_every: int, cores: int,
                per_word_topics: bool):
        models_data = []
        coherehce_score_list = []
        corpus_batch = []
        time_of_method_call = pd.to_datetime('now')

        try:
            # Compute several dask collections at once.
            streaming_documents = dask.compute(*data)
            chunksize = max(1,int(len(streaming_documents) // 5))
        except Exception as e:
            logging.error(f"Error computing streaming_documents data: {e}")
            raise

        # Select documents for current batch
        batch_documents = streaming_documents
        
        # Create a new Gensim Dictionary for the current batch
        try:
            dictionary_batch = Dictionary(list(batch_documents))
            #print("The dictionary was cretaed.")
        except TypeError:
            print("Error: The data structure is not correct.")

        flattened_batch = [item for sublist in batch_documents for item in sublist]

        # Iterate over each document in batch_documents
        number_of_documents = 0
        for doc_tokens in batch_documents:
            # Create the bag-of-words representation for the current document using the dictionary
            bow_out = dictionary_batch.doc2bow(doc_tokens)
            # Append this representation to the corpus
            corpus_batch.append(bow_out)
            number_of_documents += 1
        logging.info(f"There was a total of {number_of_documents} documents added to the corpus_batch.")
              
        n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
        n_beta = calculate_numeric_beta(beta_str, n_topics)
        try:
            lda_model_gensim = LdaModel(corpus=corpus_batch,
                                        id2word=dictionary_batch,
                                        num_topics=n_topics,
                                        alpha= float(n_alpha),
                                        eta= float(n_beta),
                                        random_state=random_state,
                                        passes=passes,
                                        iterations=iterations,
                                        update_every=update_every,
                                        eval_every=eval_every,
                                        chunksize=chunksize,
                                        per_word_topics=True)
                                          
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")
            raise  # Optionally re-raise the exception if you want it to propagate further      

        # convert lda model to pickle for storage in output dictionary
        ldamodel_bytes = pickle.dumps(lda_model_gensim)

        #coherence_score = None  # Assign a default value
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                #coherence_model_lda = CoherenceModel(model=lda_model_gensim, processes=math.floor(CORES*(2/3)), dictionary=dictionary_batch, texts=batch_documents[0], coherence='c_v') 
                coherence_model_lda = CoherenceModel(model=lda_model_gensim, processes=math.floor(cores*(1/3)), dictionary=dictionary_batch, texts=batch_documents, coherence='c_v') 
                coherence_score = coherence_model_lda.get_coherence()
                coherehce_score_list.append(coherence_score)
            except Exception as e:
                logging.error("there was an issue calculating coherence score. Value '-Inf' has been assigned.\n")
                coherence_score = float('-inf')
                coherehce_score_list.append(coherence_score)
                #sys.exit()

            try:
                convergence_score = lda_model_gensim.bound(corpus_batch)
            except Exception as e:
                logging.error("there was an issue calculating convergence score. value '-Inf' has been assigned.\n")
                convergence_score = float('-inf')
                        
            try:
                perplexity_score = lda_model_gensim.log_perplexity(corpus_batch)
            except RuntimeWarning as e:
                logging.info("there was an issue calculating perplexity score. value '-Inf' has been assigned.\n")
                perplexity_score = float('-inf')
                #sys.exit()

        # Set the number of words per topic to capture more context
        num_words = 30  # Increase as needed for more comprehensive topics
        show_topics_results = lda_model_gensim.show_topics(num_topics=-1, num_words=num_words, formatted=False)
        # Convert to a list of dictionaries for easy storage and later access
        topics_to_store = [
            {
                "method": "show_topics",
                "topic_id": topic[0],               # Topic ID
                "words": [{"word": word, "prob": prob} for word, prob in topic[1]]  # List of word-probability dictionaries
            }
            for topic in show_topics_results
        ]

        # Convert topics_to_store to ensure JSON compatibility
        topics_to_store = convert_float32_to_float(topics_to_store)

        # Convert to JSON format for PostgreSQL
        show_topics_jsonb = json.dumps(topics_to_store)

        # Get top topics with their coherence scores
        topics = lda_model_gensim.top_topics(texts=batch_documents, processes=math.floor(cores*(1/3)))
        # Extract the words as strings from each topic representation
        topic_words = []
        for topic in topics:
            topic_representation = topic[0]
            words = [word for _, word in topic_representation]
            topic_words.append(words)

        # transform list of tokens comprising the doc into a single string
        string_result = ' '.join(map(str, flattened_batch))

        # Convert numeric beta value to string if necessary
        if isinstance(beta_str, float):
            beta_str = str(beta_str)
                
        # Convert numeric alpha value to string if necessary
        if isinstance(alpha_str, float):
            alpha_str = str(alpha_str)   

        # add key for MD5 of json file(same as you did with text_md5)
        time_hash = hashlib.md5(time_of_method_call.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
        text_hash = hashlib.md5(pd.to_datetime('now').strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
        string_time = text_hash.strip() + time_hash.strip()
        current_increment_data = {
                'time_key': string_time,
                'type': train_eval,
                'num_workers': 0, # this value is set to 0 which will signify an error in assignment of adaptive-scaling worker count assigned in process_completed()
                'batch_size': -1,
                'num_documents': -1, 
                'text': [string_result],
                'text_json': pickle.dumps(batch_documents),
                'text_sha256': hashlib.sha256(string_result.encode()).hexdigest(),
                'text_md5': hashlib.md5(string_result.encode()).hexdigest(),
                'convergence': convergence_score,
                'perplexity': perplexity_score,
                'coherence': coherence_score,
                'topics': n_topics,
                'alpha_str': [alpha_str],
                'n_alpha': calculate_numeric_alpha(alpha_str, n_topics),
                'beta_str': [beta_str],
                'n_beta': calculate_numeric_beta(beta_str, n_topics),
                'passes': passes,
                'iterations': iterations,
                'update_every': update_every,
                'eval_every': eval_every,
                'chunksize': chunksize,
                'random_state': random_state,
                'per_word_topics': per_word_topics,
                'show_topics': show_topics_jsonb, 
                'top_words': topic_words,
                'lda_model': ldamodel_bytes,
                'corpus': pickle.dumps(corpus_batch),
                'dictionary': pickle.dumps(dictionary_batch),
                'create_pylda': None, 
                'create_pcoa': None, 
                'time': time_of_method_call,
                'end_time': pd.to_datetime('now'),
        }

        models_data.append(current_increment_data)

        return models_data