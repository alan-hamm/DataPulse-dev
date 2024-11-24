# process_futures.py - SpectraSync: Distributed Future Management for Topic Modeling
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This module orchestrates the distributed processing of futures in SpectraSync, ensuring that data flows smoothly 
# through LDA preparation, error handling, and database storage. It empowers SpectraSync to manage high-dimensional 
# datasets, with adaptive retries and resource management that keep the analysis engine running at peak efficiency.
#
# Functions:
# - futures_create_lda_datasets: Constructs LDA-ready datasets for training and validation phases from raw input.
# - Database utilities: Dynamic PostgreSQL utilities for creating and updating tables to store analysis results.
# - Error handling: Implements exponential backoff for failed tasks, with garbage collection to optimize memory usage.
#
# Dependencies:
# - Python libraries: time, os, json, random, pandas, logging
# - Dask libraries: distributed (for parallel processing)
#
# Developed with AI assistance to maximize SpectraSync’s distributed processing capabilities.

import sys
from .utils import exponential_backoff, garbage_collection
from .write_to_postgres import add_model_data_to_database, create_dynamic_table_class, create_table_if_not_exists

from time import sleep
import logging
from dask import delayed
from dask.distributed import wait
import os
from json import load
from random import shuffle
import random
import pandas as pd 
import numpy as np
from .utils import garbage_collection
from gensim.corpora import Dictionary
from .batch_estimation import estimate_batches_large_docs_v2, estimate_batches_large_optimized
from .mathstats import calculate_torch_coherence

def futures_create_lda_datasets(filename, train_ratio, validation_ratio, batch_size):
    with open(filename, 'r', encoding='utf-8') as jsonfile:
        data = load(jsonfile)
        print(f"The number of records read from the JSON file: {len(data)}")
        num_samples = len(data)
      
    indices = list(range(num_samples))
    shuffle(indices)
        
    num_train_samples = int(num_samples * train_ratio)
    num_validation_samples = int(num_samples * validation_ratio)
    num_test_samples = num_samples - num_train_samples - num_validation_samples

    # Print the total number of documents assigned to each split
    print(f"Total documents assigned to training set: {num_train_samples}")
    print(f"Total documents assigned to validation set: {num_validation_samples}")
    print(f"Total documents assigned to test set: {num_test_samples}")

    cumulative_count = 0
    train_count, validation_count, test_count = 0, num_train_samples, num_train_samples + num_validation_samples

    while train_count < num_train_samples or validation_count < num_train_samples + num_validation_samples or test_count < num_samples:
        if train_count < num_train_samples:
            # Ensure we only yield up to the remaining samples in the training set
            train_indices_batch = indices[train_count:min(train_count + batch_size, num_train_samples)]
            train_data_batch = [data[idx] for idx in train_indices_batch]       

            if len(train_data_batch) == 0:
                # Abort the system if the training batch is empty, as it will result in invalid modeling
                print("Error: Training batch is empty. Aborting the process to ensure data integrity.")
                sys.exit(1)

            if len(train_data_batch) > 0:
                cumulative_count += len(train_data_batch)
                yield {
                    'type': "train",
                    'data': train_data_batch,
                    'indices_batch': train_indices_batch,
                    'cumulative_count': cumulative_count,
                    'num_samples': num_train_samples
                }
                train_count += len(train_data_batch)

        elif validation_count < num_train_samples + num_validation_samples:
            # Ensure we only yield up to the remaining samples in the validation set
            validation_indices_batch = indices[validation_count:min(validation_count + batch_size, num_train_samples + num_validation_samples)]
            validation_data_batch = [data[idx] for idx in validation_indices_batch]
            if len(validation_data_batch) > 0:
                cumulative_count += len(validation_data_batch)
                #print(f"Yielding validation batch: {len(validation_data_batch)}, cumulative_count: {cumulative_count}")
                yield {
                    'type': 'validation',
                    'data': validation_data_batch,
                    'indices_batch': validation_indices_batch,
                    'cumulative_count': cumulative_count,
                    'num_samples': num_validation_samples
                }
                validation_count += len(validation_data_batch)

        elif test_count < num_samples:
            # Ensure we only yield up to the remaining samples in the test set
            test_indices_batch = indices[test_count:min(test_count + batch_size, num_samples)]
            test_data_batch = [data[idx] for idx in test_indices_batch]
            if len(test_data_batch) > 0:
                cumulative_count += len(test_data_batch)
                #print(f"Yielding test batch: {len(test_data_batch)}, cumulative_count: {cumulative_count}")
                yield {
                    'type': 'test',
                    'data': test_data_batch,
                    'indices_batch': test_indices_batch,
                    'cumulative_count': cumulative_count,
                    'num_samples': num_test_samples
                }
                test_count += len(test_data_batch)
    
    print(f"Final cumulative count after all batches: {cumulative_count}")



def futures_create_lda_datasets_v2(documents_path, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
    # Load the document data from JSON file
    with open(documents_path, 'r', encoding='utf-8') as jsonfile:
        documents = load(jsonfile)

    # Create a unified dictionary using the entire corpus (list of lists of tokens)
    dictionary = Dictionary(documents)

    # Estimate batch size based on the documents using the estimator
    batch_size = estimate_batches_large_optimized(documents_path)

    # Calculate diversity-based sampling probabilities
    weights = [len(set(doc)) for doc in documents]  # Number of unique words in each document
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]

    # Ensure probabilities sum to 1 due to floating point imprecision
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # Determine the number of documents for each split
    total_documents = len(documents)
    train_size = int(train_ratio * total_documents)
    validation_size = int(validation_ratio * total_documents)
    test_size = total_documents - train_size - validation_size

    # Print the total number of documents assigned to each split
    print(f"Total documents assigned to training set: {train_size}")
    print(f"Total documents assigned to validation set: {validation_size}")
    print(f"Total documents assigned to test set: {test_size}")

    # Weighted sampling for train, validation, and test indices
    all_indices = list(range(total_documents))
    train_indices = np.random.choice(all_indices, size=train_size, replace=False, p=probabilities)
    remaining_indices = list(set(all_indices) - set(train_indices))

    # Normalize probabilities for the remaining indices
    remaining_probabilities = [probabilities[i] for i in remaining_indices]
    remaining_total_weight = sum(remaining_probabilities)
    normalized_remaining_probabilities = [p / remaining_total_weight for p in remaining_probabilities]

    # Ensure normalized probabilities sum to 1
    normalized_remaining_probabilities = np.array(normalized_remaining_probabilities)
    normalized_remaining_probabilities /= normalized_remaining_probabilities.sum()

    # Validation sampling
    validation_indices = np.random.choice(remaining_indices, size=validation_size, replace=False, p=normalized_remaining_probabilities)
    test_indices = list(set(remaining_indices) - set(validation_indices))

    # Use sampled indices to create datasets
    train_documents = [documents[i] for i in train_indices]
    validation_documents = [documents[i] for i in validation_indices]
    test_documents = [documents[i] for i in test_indices]

    # Yield the dictionary as the first item
    yield {"type": "dictionary", "data": dictionary}

    # Yield batches for each phase as a generator, including type information
    def create_batches(dataset, phase_type):
        num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
        for i in range(num_batches):
            yield {
                "type": phase_type,
                "data": dataset[i * batch_size: (i + 1) * batch_size]
            }

    yield from create_batches(train_documents, "train")
    yield from create_batches(validation_documents, "validation")
    yield from create_batches(test_documents, "test")



def futures_create_lda_datasets_v3(documents_path, train_ratio=0.7, validation_ratio=0.15, seed=42):
    """
    Create training, validation, and test datasets for LDA with diversity-based sampling and batch generation.

    Args:
        documents_path (str): Path to the document JSON file.
        train_ratio (float): Proportion of documents for training.
        validation_ratio (float): Proportion of documents for validation.
        seed (int): Random seed for reproducibility. Default is 42.

    Yields:
        dict: Batches of data with type ('train', 'validation', or 'test') and document data.
    """
    
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Load the document data from JSON file
    with open(documents_path, 'r', encoding='utf-8') as jsonfile:
        documents = load(jsonfile)

    total_raw_documents = len(documents)

    # validate documents
    cleaned_documents = []
    for doc_tokens in documents:
        # Validate `doc_tokens`
        if not isinstance(doc_tokens, list):
            logging.warning(f"Unexpected structure for doc_tokens: {type(doc_tokens)}, content: {doc_tokens}")
            logging.warning("Skipping token in [futures_create_lda_datasets_v3].")
            continue  # Skip invalid document

        # Validate non-empty tokens
        if len(doc_tokens) == 0:
            logging.warning(f"Skipping empty document: {doc_tokens}")
            logging.warning("Skipping document in [futures_create_lda_datasets_v3].")
            continue  # Skip empty documents
        
        cleaned_documents.append(doc_tokens)

    # Create a unified dictionary using the entire corpus (list of lists of tokens)
    print("Creating dictinoary...\n")
    dictionary = Dictionary(cleaned_documents)

    # Estimate batch size based on the documents using the estimator
    batch_size = estimate_batches_large_optimized(
        documents_path, min_batch_size=5, max_batch_size=15, memory_limit_ratio=0.4, cpu_factor=4
    )

    # Determine the number of documents for each split
    total_documents = len(cleaned_documents)
    train_size = int(train_ratio * total_documents)
    validation_size = int(validation_ratio * total_documents)
    test_size = total_documents - train_size - validation_size

    # calculate percent documents loss during cleaning
    percent_lost = ((total_raw_documents - total_documents) / total_raw_documents) * 100
    
    # Print dataset split sizes
    print(f"Total number of documents read from JSON: {total_raw_documents}.")
    print(f"Total number of cleaned documents: {total_documents}.")
    print(f"Percentage of documents lost during cleaning: {percent_lost}\n")

    print(f"Total documents assigned to training set: {train_size}")
    print(f"Total documents assigned to validation set: {validation_size}")
    print(f"Total documents assigned to test set: {test_size} \n")

    # Calculate diversity-based sampling probabilities
    weights = [len(set(doc)) + len(doc) * 0.1 for doc in documents]
    total_weight = sum(weights)
    probabilities = np.array([weight / total_weight for weight in weights], dtype=float)

    # Weighted sampling for train, validation, and test indices
    all_indices = np.arange(total_documents)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False, p=probabilities)
    remaining_indices = np.setdiff1d(all_indices, train_indices)

    # Normalize probabilities for the remaining indices
    remaining_probabilities = probabilities[remaining_indices]
    remaining_probabilities /= remaining_probabilities.sum()  # Normalize to sum to 1

    # Validation sampling
    validation_indices = np.random.choice(
        remaining_indices, size=validation_size, replace=False, p=remaining_probabilities
    )
    test_indices = np.setdiff1d(remaining_indices, validation_indices)

    # Ensure no overlap between splits
    assert not (set(train_indices) & set(validation_indices)), "Train and validation indices overlap!"
    assert not (set(validation_indices) & set(test_indices)), "Validation and test indices overlap!"
    assert not (set(train_indices) & set(test_indices)), "Train and test indices overlap!"

    # Use sampled indices to create datasets
    train_documents = [cleaned_documents[i] for i in train_indices]
    validation_documents = [cleaned_documents[i] for i in validation_indices]
    test_documents = [cleaned_documents[i] for i in test_indices]

    # Log batch counts
    print(f"Training batches: {len(train_documents) // batch_size + (1 if len(train_documents) % batch_size != 0 else 0)}")
    print(f"Validation batches: {len(validation_documents) // batch_size + (1 if len(validation_documents) % batch_size != 0 else 0)}")
    print(f"Test batches: {len(test_documents) // batch_size + (1 if len(test_documents) % batch_size != 0 else 0)}\n")

    # Yield the dictionary as the first item
    yield {"type": "dictionary", "data": dictionary}

    # Yield batches for each phase
    def create_batches(dataset, phase_type):
        num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
        for i in range(num_batches):
            yield {
                "type": phase_type,
                "data": dataset[i * batch_size: (i + 1) * batch_size]
            }

    yield from create_batches(train_documents, "train")
    yield from create_batches(validation_documents, "validation")
    yield from create_batches(test_documents, "test")




def process_completed_futures(phase, connection_string, corpus_label, \
                            completed_train_futures, completed_validation_futures, completed_test_futures, \
                            num_documents, workers, \
                            batchsize, texts_zip_dir, vis_pylda=None, vis_pcoa=None, vis_pca=None):

    # Create a mapping from model_data_id to visualization results
    #his is the vis_pyldaprint(f"This is the vis_pylda: {vis_pylda}")
    pylda_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pylda if vis_result}
    pcoa_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pcoa if vis_result}
    pca_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pca if vis_result}
    # do union of items
    # Combine both maps into a unified vis_results_map
    vis_results_map = {}
    for key in set(pylda_results_map) | set(pcoa_results_map):
        create_pylda = pylda_results_map.get(key)
        create_pcoa = pcoa_results_map.get(key)
        create_pca_gpu = pca_results_map.get(key)
        vis_result = (create_pylda if create_pylda is not None else (None, None),
                      create_pcoa if create_pcoa is not None else (None, None),
                      create_pca_gpu if create_pca_gpu is not None else (None, None))
        vis_results_map[key] = vis_result


    # DEBUGGING
    #if visualization_results and len(visualization_results) >= 2:
    #    print(visualization_results[0], visualization_results[1])
    #print(f"this is the vis_results_map(): {vis_results_map}")

    # Process training futures
    if len(completed_train_futures) > 0:
        for models_data in completed_train_futures:
            logging.info(f"Train Model data for database write: {models_data}")

            try:
                # Ensure models_data is a list; if not, convert it to a list containing the item
                if isinstance(models_data, dict):
                    models_data = [models_data]
                elif not isinstance(models_data, list):
                    logging.error(f"Unexpected type for models_data: {type(models_data)}. Attempting to convert.")
                    models_data = [models_data]  # Attempt conversion for processing

                for model_data in models_data:
                            # Ensure each item in models_data is a dictionary; otherwise, log an error
                            if not isinstance(model_data, dict):
                                logging.error(f"Unexpected type for model_data: {type(model_data)}. Converting to empty dictionary.")
                                model_data = {}  # Convert to an empty dictionary to avoid errors and retain the item

                            unique_id = model_data.get('time_key')
                            # Check for exact match in vis_results_map keys
                            #logging.info(f"Looking up unique_id: '{unique_id}' (length: {len(unique_id)})")

                            # Check for exact match in vis_results_map keys
                            #match_found = False
                            #for key in vis_results_map.keys():
                            #    logging.info(f"Comparing unique_id '{unique_id}' with vis_results_map key '{key}'")
                            #    if unique_id == key:
                             #       logging.info(f"Exact match found: unique_id ({unique_id}) matches key ({key})")
                            #        match_found = True
                            #        break  # Exit loop as soon as a match is found
                            #if not match_found:
                            #    logging.info(f"No match found for unique_id: {unique_id}")

                            if unique_id and unique_id in vis_results_map:
                                create_pylda, create_pcoa, create_pca_gpu = vis_results_map[unique_id]
                                logging.info(f"Train: Found results for unique_id {unique_id}: create_pylda={create_pylda}, create_pcoa={create_pcoa}, create_pca_gpu={create_pca_gpu}")
                                model_data['create_pylda'] = create_pylda[0]
                                model_data['create_pcoa'] = create_pcoa[0]
                                model_data['create_pca_gpu'] = create_pca_gpu[0]
                                model_data['num_documents'] = num_documents
                                model_data['batch_size'] = batchsize
                                model_data['num_workers'] = workers
            except Exception as e:
                    logging.error(f"Error occurred during process_completed_futures() TRAIN: {e}")
            try:
                #print("We are prior to DynamicModelMetadata")
                DynamicModelMetadata = create_dynamic_table_class(corpus_label)
                #print("\nWe are prior to create_table_if_not_exist()")
                create_table_if_not_exists(DynamicModelMetadata, connection_string)
                #print("\nwe are prior to add_model_data_to_database()")
                add_model_data_to_database(model_data, phase, corpus_label, connection_string,
                                                num_documents, workers, batchsize, texts_zip_dir)
            except Exception as e:
                logging.error(f"Error occurred during process_completed_futures() add_model_data_to_database() TRAIN: {e}")

    # Process evaluation futures
    #vis_futures = []
    if len(completed_validation_futures) > 0:
        for models_data in completed_validation_futures:
            logging.info(f"Validation Model data for database write: {models_data}")
            try:
                # Ensure models_data is a list; if not, convert it to a list containing the item
                if isinstance(models_data, dict):
                    models_data = [models_data]
                elif not isinstance(models_data, list):
                    logging.error(f"Unexpected type for models_data: {type(models_data)}. Attempting to convert.")
                    models_data = [models_data]  # Attempt conversion for processing

                for model_data in models_data:
                            # Ensure each item in models_data is a dictionary; otherwise, log an error
                            if not isinstance(model_data, dict):
                                logging.error(f"Unexpected type for model_data: {type(model_data)}. Converting to empty dictionary.")
                                model_data = {}  # Convert to an empty dictionary to avoid errors and retain the item

                            unique_id = model_data.get('time_key')
                            if unique_id and unique_id in vis_results_map:
                                create_pylda, create_pcoa, create_pca_gpu = vis_results_map[unique_id]
                                logging.info(f"Validation: Found results for unique_id {unique_id}: create_pylda={create_pylda}, create_pcoa={create_pcoa}, create_pca_gpu={create_pca_gpu}")
                                model_data['create_pylda'] = create_pylda[0]
                                model_data['create_pcoa'] = create_pcoa[0]
                                model_data['create_pca_gpu'] = create_pca_gpu[0]
                                model_data['num_documents'] = num_documents
                                model_data['batch_size'] = batchsize
                                model_data['num_workers'] = workers
            except Exception as e:
                logging.error(f"Error occurred during process_completed_futures() EVAL: {e}")
            try:
                DynamicModelMetadata = create_dynamic_table_class(corpus_label)
                create_table_if_not_exists(DynamicModelMetadata, connection_string)
                add_model_data_to_database(model_data,phase, corpus_label, connection_string,
                                        num_documents, workers, batchsize, texts_zip_dir)
            except Exception as e:
                logging.error(f"Error occurred during process_completed_futures() add_model_data_to_database() VALIDATION: {e}")

    if len(completed_test_futures) > 0:    
        for models_data in completed_test_futures:
            logging.info(f"Test Model data for database write: {models_data}")
            try:
                # Ensure models_data is a list; if not, convert it to a list containing the item
                if isinstance(models_data, dict):
                    models_data = [models_data]
                elif not isinstance(models_data, list):
                    logging.error(f"Unexpected type for models_data: {type(models_data)}. Attempting to convert.")
                    models_data = [models_data]  # Attempt conversion for processing

                for model_data in models_data:
                    # Ensure each item in models_data is a dictionary; otherwise, log an error
                    if not isinstance(model_data, dict):
                        logging.error(f"Unexpected type for model_data: {type(model_data)}. Converting to empty dictionary.")
                        model_data = {}  # Convert to an empty dictionary to avoid errors and retain the item

                    unique_id = model_data.get('time_key')
                    if unique_id and unique_id in vis_results_map:
                        create_pylda, create_pcoa, create_pca_gpu = vis_results_map[unique_id]
                        logging.info(f"Test: Found results for unique_id {unique_id}: create_pylda={create_pylda}, create_pcoa={create_pcoa}, create_pca_gpu={create_pca_gpu}")
                        model_data['create_pylda'] = create_pylda[0]
                        model_data['create_pcoa'] = create_pcoa[0]
                        model_data['create_pca_gpu'] = create_pca_gpu[0]
                        model_data['num_documents'] = num_documents
                        model_data['batch_size'] = batchsize
                        model_data['num_workers'] = workers
            except Exception as e:
                logging.error(f"Error occurred during process_completed_futures() EVAL: {e}")
            try:
                DynamicModelMetadata = create_dynamic_table_class(corpus_label)
                create_table_if_not_exists(DynamicModelMetadata, connection_string)
                add_model_data_to_database(model_data, phase, corpus_label, connection_string,
                                        num_documents, workers, batchsize, texts_zip_dir)
            except Exception as e:
                logging.error(f"Error occurred during process_completed_futures() add_model_data_to_database() TEST: {e}")

    return completed_train_futures, completed_validation_futures, completed_test_futures


# Batch process to get topics and calculate coherence efficiently
@delayed
def get_show_topics(ldamodel, num_words):
    try:
        show_topics = ldamodel.show_topics(num_topics=-1, num_words=num_words, formatted=False)
        processed_topics = [
            {
                "method": "show_topics",
                "topic_id": topic[0],
                "words": [{"word": word, "prob": prob} for word, prob in topic[1]]
            }
            for topic in show_topics
        ]
        return processed_topics
    except Exception as e:
        logging.warning(f"An error occurred while processing show_topics: {e}")
        return [{"method": "show_topics", "topic_id": None, "words": [], "error": str(e)}]



# Module for delayed topic extraction
@delayed
def get_and_process_show_topics(ldamodel, num_words, kwargs=None):
    """
    Delayed function to get and process show_topics from an LDA model.

    Parameters:
    - ldamodel: Trained LDA model.
    - num_words: Number of words to extract for each topic.
    - kwargs (optional): Additional parameters for diagnostic information.

    Returns:
    - A list of processed topics in the desired format.
    """
    try:
        # Extract topics from the LDA model
        show_topics = ldamodel.show_topics(num_topics=-1, num_words=num_words, formatted=False)
        processed_topics = [
            {
                "method": "show_topics",
                "topic_id": topic[0],
                "words": [{"word": word, "prob": prob} for word, prob in topic[1]]
            }
            for topic in show_topics
        ]
        return processed_topics
    except Exception as e:
        # Handle errors and provide diagnostic information
        logging.warning(f"An error occurred while processing show_topics: {e}")
        return [
            {
                "method": "show_topics",
                "topic_id": None,
                "words": [],
                "error": str(e),
                "record_id": kwargs.get("record_id", "unknown") if kwargs else "unknown",
                "parameters": {
                    "num_topics": kwargs.get("num_topics", "unknown") if kwargs else "unknown",
                    "num_words": kwargs.get("num_words", "unknown") if kwargs else "unknown",
                    "alpha": kwargs.get("alpha", "unknown") if kwargs else "unknown",
                    "beta": kwargs.get("beta", "unknown") if kwargs else "unknown",
                }
            }
        ]

@delayed
def extract_topics_with_get_topic_terms(ldamodel, num_words, kwargs=None):
    """
    Delayed function to extract topics using get_topic_terms from an LDA model.

    Parameters:
    - ldamodel: Trained LDA model.
    - num_words: Number of words to extract for each topic.
    - kwargs (optional): Additional parameters for diagnostic information.

    Returns:
    - A list of processed topics in the desired format.
    """
    try:
        # Extract topic-word distributions for all topics
        topics = [
            {
                "method": "get_topic_terms",
                "topic_id": topic_id,
                "words": [{"word": word, "prob": prob} for word, prob in ldamodel.get_topic_terms(topic_id, num_words)]
            }
            for topic_id in range(ldamodel.num_topics)
        ]
        return topics
    except Exception as e:
        # Handle errors and provide diagnostic information
        logging.warning(f"An error occurred while extracting topics: {e}")
        return [
            {
                "method": "get_topic_terms",
                "topic_id": None,
                "words": [],
                "error": str(e),
                "record_id": kwargs.get("record_id", "unknown") if kwargs else "unknown",
                "parameters": {
                    "num_topics": kwargs.get("num_topics", "unknown") if kwargs else "unknown",
                    "num_words": kwargs.get("num_words", "unknown") if kwargs else "unknown",
                    "alpha": kwargs.get("alpha", "unknown") if kwargs else "unknown",
                    "beta": kwargs.get("beta", "unknown") if kwargs else "unknown",
                }
            }
        ]
    
# Batch process to get topics for a batch of documents
@delayed
def get_document_topics_batch(ldamodel, bow_docs):
    """
    Retrieve significant topics for a batch of documents in a delayed Dask task.

    Parameters:
    - ldamodel (LdaModel): Trained LDA model to extract document topics.
    - bow_docs (list of list of tuples): List of Bag-of-Words representations of documents.

    Returns:
    - list: A list of topic lists for each document with their respective probabilities.
    """
    batch_results = []
    for bow_doc in bow_docs:
        try:
            topics = ldamodel.get_document_topics(bow_doc, minimum_probability=0)
            if not topics:
                logging.warning(f"No significant topics found for document: {bow_doc}")
                # Append a placeholder for empty results
                batch_results.append([{"topic_id": None, "probability": 0}])
            else:
                batch_results.append(topics)
        except Exception as e:
            # Log the error and append a placeholder result
            logging.error(f"Error getting document topics for document {bow_doc}: {e}")
            batch_results.append([{"topic_id": None, "probability": 0}])
            # Do NOT raise the exception; continue processing the next document

    return batch_results

