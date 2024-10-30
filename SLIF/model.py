# developed traditionally in with addition of AI assistance
from .utils import garbage_collection
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

from .alpha_eta import calculate_numeric_alpha, calculate_numeric_beta

# https://examples.dask.org/applications/embarrassingly-parallel.html
def train_model(n_topics: int, alpha_str: list, beta_str: list, data: list, train_eval: str, 
                random_state: int, passes: int, iterations: int, update_every: int, eval_every: int, cores: int,
                per_word_topics: bool):
        models_data = []
        coherehce_score_list = []
        corpus_batch = []
        time_of_method_call = pd.to_datetime('now')

        #print("this is an investigation into the full datafile")
        #pp.pprint(full_datafile)
        try:
            # Compute several dask collections at once.
            streaming_documents = dask.compute(*data)
            chunksize = max(1,int(len(streaming_documents) // 5))
            #print("these are the streaming documents")
            #print(streaming_documents)
            #garbage_collection(False, 'train_model(): streaming_documents = dask.compute(*data)')
        except Exception as e:
            logging.error(f"Error computing streaming_documents data: {e}")
            raise
        #print(f"This is the dtype for 'streaming_documents' {type(streaming_documents)}.\n")  # Should output <class 'tuple'>
        #print(streaming_documents[0][0])     # Check the first element to see if it's as expected

        # Select documents for current batch
        batch_documents = streaming_documents
        
        # Create a new Gensim Dictionary for the current batch
        try:
            dictionary_batch = Dictionary(list(batch_documents))
            #print("The dictionary was cretaed.")
        except TypeError:
            print("Error: The data structure is not correct.")
        #else:
        #    print("Dictionary created successfully!")

        #if isinstance(batch_documents[0], list) and all(isinstance(doc, list) for doc in batch_documents[0]):
        #bow_out = dictionary_batch.doc2bow(batch_documents[0])
        flattened_batch = [item for sublist in batch_documents for item in sublist]
        #bow_out = dictionary_batch.doc2bow(flattened_batch)
        #else:
        #    raise ValueError(f"Expected batch_documents[0] to be a list of token lists. Instead received {type(batch_documents[0])} with value {batch_documents[0]}\n")

        # Iterate over each document in batch_documents
        number_of_documents = 0
        for doc_tokens in batch_documents:
            # Create the bag-of-words representation for the current document using the dictionary
            bow_out = dictionary_batch.doc2bow(doc_tokens)
            # Append this representation to the corpus
            corpus_batch.append(bow_out)
            number_of_documents += 1
        logging.info(f"There was a total of {number_of_documents} documents added to the corpus_batch.")
            
        #logger.info(f"HERE IS THE TEXT for corpus_batch using LOGGER: {corpus_batch}\n")
        #except Exception as e:
        #    logger.error(f"An unexpected error occurred with BOW_OUT: {e}")
                
        #if isinstance(texts_out[0], list):
        #    texts_batch.append(texts_out[0])
        #else:
        #    logging.error("Expected texts_out to be a list of strings (words), got:", texts_out[0])
        #    raise ValueError("Expected texts_out to be a list of strings (words), got:", texts_out[0])
                
        n_alpha = calculate_numeric_alpha(alpha_str, n_topics)
        n_beta = calculate_numeric_beta(beta_str, n_topics)
        try:
            #logger.info("we are inside the try block at the beginning")
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
            #logger.info("we are inside the try block after the constructor")

                                          
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

        # Get top topics with their coherence scores
        #topics_as_word_lists=[]
        topics = lda_model_gensim.top_topics(texts=batch_documents, processes=math.floor(cores*(1/3)))
        # Extract the words as strings from each topic representation
        topic_words = []
        for topic in topics:
            topic_representation = topic[0]
            words = [word for _, word in topic_representation]
            topic_words.append(words)
            
            # Append this list of words for current topic to the main list
            #topics_as_word_lists.append(topic_words)
            
        #print(f"type: {train_eval}, coherence: {coherence_score}, n_topics: {n_topics}, n_alpha: {n_alpha}, alpha_str: {alpha_str}, n_beta: {n_beta}, beta_str: {beta_str}")
        #logging.info(f"type: {train_eval}, coherence: {coherence_score}, n_topics: {n_topics}, alpha_str: {alpha_str}, beta_str: {beta_str}, batch documents: {batch_documents}")     

        # transform list of tokens comprising the doc into a single string
        string_result = ' '.join(map(str, flattened_batch))

        # Convert numeric beta value to string if necessary
        if isinstance(beta_str, float):
            beta_str = str(beta_str)
                
        # Convert numeric alpha value to string if necessary
        if isinstance(alpha_str, float):
            alpha_str = str(alpha_str)   

        # get time to complete function
        #time_to_complete = (datetime.now() - time_of_method_call).total_seconds()
        #formatted_time = time_to_complete.strftime("%H:%M:%S.%f")

        # add key for MD5 of json file(same as you did with text_md5)
        current_increment_data = {
                'time_key': hashlib.md5(time_of_method_call.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest(),
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
        #garbage_collection(False, 'train_model(...)')
        #del batch_documents, streaming_documents, lda_model_gensim, dictionary_batch, current_increment_data #, vis, success

        return models_data