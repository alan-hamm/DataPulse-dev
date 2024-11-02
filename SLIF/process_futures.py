# Developed traditionally with the addition of AI assistance
# Author: Alan Hamm (pqn7)
# Date: April 2024

from .utils import exponential_backoff, garbage_collection
from .write_to_postgres import add_model_data_to_database, create_dynamic_table_class, create_table_if_not_exists

from time import sleep
import logging
from dask.distributed import wait


def process_completed_futures(connection_string, corpus_label, completed_train_futures, completed_eval_futures, num_documents, workers, \
                                batchsize, texts_zip_dir, metadata_dir=None, vis_pylda=None, vis_pcoa=None):

    # Create a mapping from model_data_id to visualization results
    pylda_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pylda if vis_result}
    pcoa_results_map = {vis_result[0]: vis_result[1:] for vis_result in vis_pcoa if vis_result}
    # do union of items
    #vis_results_map = dict(pylda_results_map.items() | pcoa_results_map.items())
    vis_results_map = {}
    for key in set(pylda_results_map) | set(pcoa_results_map):
        pylda_result = pylda_results_map.get(key)
        pcoa_result = pcoa_results_map.get(key)
        
        # Debug output
        #logging.info(f"Key: {key}, PyLDA Result: {pylda_result}, PCoA Result: {pcoa_result}")

        # You may choose what to do if one result is missing - perhaps use None or a default value
        vis_result = (pylda_result if pylda_result is not None else (None, None),
                    pcoa_result if pcoa_result is not None else (None, None))
        
        vis_results_map[key] = vis_result
    # DEBUGGING
    #if visualization_results and len(visualization_results) >= 2:
    #    print(visualization_results[0], visualization_results[1])
    #print(f"this is the vis_results_map(): {vis_results_map}")

    # Process training futures
    for future in completed_train_futures:
        try:
            models_data = future.result()  # This should be a list of dictionaries
            if not isinstance(models_data, list):
                models_data = [models_data]  # Ensure it is a list

            for model_data in models_data:
                unique_id = model_data['time_key']
                
                # Retrieve visualization results using filename hash as key
                if unique_id in vis_results_map:
                    #print(f"We are in the process_completed mapping time hash key.")
                    create_pylda, create_pcoa = vis_results_map[unique_id]
                    model_data['create_pylda'] = create_pylda[0]
                    model_data['create_pcoa'] = create_pcoa[0]
                    model_data['num_documents'] = num_documents
                    model_data['batch_size'] = batchsize
                    model_data['num_workers'] = workers
                    #logging.info(f"TRAIN Assigned 'create_pylda': {model_data['create_pylda']}, 'create_pcoa': {model_data['create_pcoa']}")
        except Exception as e:
            logging.error(f"Error occurred during process_completed_futures() TRAIN: {e}")
        try:
                DynamicModelMetadata = create_dynamic_table_class(corpus_label)
                create_table_if_not_exists(DynamicModelMetadata, connection_string)
                add_model_data_to_database(model_data, corpus_label, connection_string,
                                        num_documents, workers, batchsize, texts_zip_dir)
        except Exception as e:
            logging.error(f"Error occurred during process_completed_futures() add_model_data_to_database() TRAIN: {e}")

    # Process evaluation futures
    #vis_futures = []
    for future in completed_eval_futures:
        try:
            models_data = future.result()  # This should be a list of dictionaries
            if not isinstance(models_data, list):
                models_data = [models_data]  # Ensure it is a list

            for model_data in models_data:
                unique_id = model_data['time_key']
                
                # Retrieve visualization results using filename hash as key
                if unique_id in vis_results_map:
                    create_pylda, create_pcoa = vis_results_map[unique_id]
                    model_data['create_pylda'] = create_pylda[0]
                    model_data['create_pcoa'] = create_pcoa[0]
                    model_data['num_documents'] = num_documents
                    model_data['batch_size'] = batchsize
                    model_data['num_workers'] = workers
                    #logging.info(f"EVAL Assigned 'create_pylda': {model_data['create_pylda']}, 'create_pcoa': {model_data['create_pcoa']}")
        except Exception as e:
            logging.error(f"Error occurred during process_completed_futures() EVAL: {e}")
        try:
            DynamicModelMetadata = create_dynamic_table_class(corpus_label)
            create_table_if_not_exists(DynamicModelMetadata, connection_string)
            add_model_data_to_database(model_data, corpus_label, connection_string,
                                        num_documents, workers, batchsize, texts_zip_dir)
        except Exception as e:
            logging.error(f"Error occurred during process_completed_futures() add_model_data_to_database() EVAL: {e}")
        
                    
    #del models_data            
    #garbage_collection(False, 'process_completed_futures(...)')
    return completed_eval_futures, completed_train_futures


# Function to retry processing with incomplete futures
def retry_processing(incomplete_train_futures, incomplete_eval_futures, failed_model_params, future_to_params, timeout=None):
    # Retry processing with incomplete futures using an extended timeout
    # Wait for completion of eval_futures
    done_eval, not_done_eval = wait(incomplete_eval_futures, timeout=timeout)  # return_when='FIRST_COMPLETED'
    #print(f"This is the size of the done_eval list: {len(done_eval)} and this is the size of the not_done_eval list: {len(not_done_eval)}")

    # Wait for completion of train_futures
    done_train, not_done_train = wait(incomplete_train_futures, timeout=timeout)  # return_when='FIRST_COMPLETED'

    done = done_train.union(done_eval)
    not_done = not_done_eval.union(not_done_train)
  
    completed_train_futures = [f for f in done_train]
    completed_eval_futures = [f for f in done_eval]

    #logging.info(f"This is the size of completed_train_futures {len(completed_train_futures)} and this is the size of completed_eval_futures {len(completed_eval_futures)}")
    if len(completed_eval_futures) > 0 or len(completed_train_futures) > 0:
        process_completed_futures(completed_train_futures, completed_eval_futures) 
    
    # Record parameters of still incomplete futures for later review
    failed_model_params.extend(future_to_params[future] for future in not_done)
    print("We have exited the retry_preprocessing() method.")
    logging.info(f"There were {len(not_done_eval)} EVAL documents that couldn't be processed in retry_processing().")
    logging.info(f"There were {len(not_done_train)} TRAIN documents that couldn't be processed in retry_processing().")

    return failed_model_params
    #garbage_collection(False, 'retry_processing(...)')


# Function to handle failed futures and potentially retry them
def handle_failed_future(train_model, client, future, future_to_params, train_futures, eval_futures, MAX_RETRIES=1):
    # Dictionary to keep track of retries for each task
    task_retries = {}
    
    logging.info("We are in the handle_failed_future() method.\n")
    params = future_to_params[future]
    attempt = task_retries.get(params, 0)
    
    if attempt < MAX_RETRIES:
        logging.info(f"Retrying task {params} (attempt {attempt + 1}/{MAX_RETRIES})")
        wait_time = exponential_backoff(attempt)
        sleep(wait_time)  
        
        task_retries[params] = attempt + 1
        
        new_future_train = client.submit(train_model, *params)
        new_future_eval = client.submit(train_model, *params)
        
        future_to_params[new_future_train] = params
        future_to_params[new_future_eval] = params
        
        train_futures.append(new_future_train)
        eval_futures.append(new_future_eval)
    else:
        logging.info(f"Task {params} failed after {MAX_RETRIES} attempts. No more retries.")