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

from .utils import exponential_backoff, garbage_collection
from .write_to_postgres import add_model_data_to_database, create_dynamic_table_class, create_table_if_not_exists

from time import sleep
import logging
from dask.distributed import wait
import os
from json import load
from random import shuffle
import pandas as pd 
from .utils import garbage_collection


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
            if len(train_data_batch) > 0:
                cumulative_count += len(train_data_batch)
                #print(f"Yielding train batch: {len(train_data_batch)}, cumulative_count: {cumulative_count}")
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


def process_completed_futures(phase, connection_string, corpus_label, \
                            completed_train_futures, completed_validation_futures, completed_test_futures, \
                            num_documents, workers, \
                            batchsize, texts_zip_dir, vis_pylda=None, vis_pcoa=None):

    # Create a mapping from model_data_id to visualization results
    #his is the vis_pyldaprint(f"This is the vis_pylda: {vis_pylda}")
    pylda_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pylda if vis_result}
    pcoa_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pcoa if vis_result}
    # do union of items
    # Combine both maps into a unified vis_results_map
    vis_results_map = {}
    for key in set(pylda_results_map) | set(pcoa_results_map):
        create_pylda = pylda_results_map.get(key)
        create_pcoa = pcoa_results_map.get(key)
        vis_result = (create_pylda if create_pylda is not None else (None, None),
                      create_pcoa if create_pcoa is not None else (None, None))
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
                                create_pylda, create_pcoa = vis_results_map[unique_id]
                                logging.info(f"Found results for unique_id {unique_id}: create_pylda={create_pylda}, create_pcoa={create_pcoa}")
                                model_data['create_pylda'] = create_pylda[0]
                                model_data['create_pcoa'] = create_pcoa[0]
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
                                create_pylda, create_pcoa = vis_results_map[unique_id]
                                model_data['create_pylda'] = create_pylda[0]
                                model_data['create_pcoa'] = create_pcoa[0]
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
                        create_pylda, create_pcoa = vis_results_map[unique_id]
                        model_data['create_pylda'] = create_pylda[0]
                        model_data['create_pcoa'] = create_pcoa[0]
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