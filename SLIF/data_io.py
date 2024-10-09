# developed traditionally in addition to pair programming
import os
from json import load
from random import shuffle
import pandas as pd 
import hashlib
import zipfile
import logging
from .utils import garbage_collection

def get_num_records(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as jsonfile:
        data = load(jsonfile)
        data = data
        num_samples = len(data)  # Count the total number of samples
    return num_samples

def futures_create_lda_datasets(filename, train_ratio, batch_size):
    #with open(filename, 'r', encoding='utf-8', errors='ignore') as jsonfile:
    with open(filename, 'r', encoding='utf-8') as jsonfile:
        data = load(jsonfile)
        print(f"the number of records read from the JSON file: {len(data)}")
        num_samples = len(data)  # Count the total number of samples
        #print(f"the number of documents sampled from the JSON file: {len(data)}\n")
        
    # Shuffle data indices since we can't shuffle actual lines in a file efficiently
    indices = list(range(num_samples))
    shuffle(indices)
        
    num_train_samples = int(num_samples * train_ratio)  # Calculate number of samples for training
        
    cumulative_count = 0  # Initialize cumulative count
    # Initialize counters for train and eval datasets
    train_count = 0
    eval_count = num_train_samples
        
    # Yield batches as dictionaries for both train and eval datasets along with their sample count
    while (train_count < num_train_samples or eval_count < num_samples):
        if train_count < num_train_samples:
            # Yield a training batch
            train_indices_batch = indices[train_count:train_count + batch_size]
            train_data_batch = [data[idx] for idx in train_indices_batch]
            if len(train_data_batch) > 0:
                print(f"Yielding training batch: {train_count} to {train_count + len(train_data_batch)}") # ... existing code for yielding training batch
                yield {
                    'type': 'train',
                    'data': train_data_batch,
                    'indices_batch': train_indices_batch,
                    'cumulative_count': train_count,
                    'num_samples': num_train_samples,
                    'whole_dataset': data[:num_train_samples]
                    }
                train_count += len(train_data_batch)
                cumulative_count += train_count
            
        if (eval_count < num_samples or train_count >= num_train_samples):
            # Yield an evaluation batch
            #print("we are in the method to create the futures trying to create the eval data.")
            #print(f"the eval count is {eval_count} and the train count is {train_count} and the num train samples is {num_train_samples}\n")
            eval_indices_batch = indices[eval_count:eval_count + batch_size]
            eval_data_batch = [data[idx] for idx in eval_indices_batch]
            #print(f"This is the size of the eval_data_batch from the create futures method {len(eval_data_batch)}\n")
            if len(eval_data_batch) > 0:
                #print(f"Yielding evaluation batch: {eval_count} to {eval_count + len(eval_data_batch)}") # ... existing code for yielding evaluation batch ...
                yield {
                    'type': 'eval',
                    'data': eval_data_batch,
                    'indices_batch': eval_indices_batch,
                    'cumulative_count': num_train_samples - eval_count,
                    'num_samples': num_train_samples - num_samples,
                    'whole_dataset': data[num_train_samples:]
                    }
                eval_count += len(eval_data_batch)
                cumulative_count += eval_count
    garbage_collection(False, "futures_create_lda_datasets(..)")


# Function to save text data and model to single ZIP file
def save_to_zip(time, top_folder, text_data, text_json, ldamodel, corpus, dictionary, texts_zip_dir):
    # Generate a unique filename based on current timestamp
    timestamp_str = hashlib.md5(time.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
    text_zip_filename = f"{timestamp_str}.zip"
    
    # Write the text content and model to a zip file within TEXTS_ZIP_DIR
    zip_path = os.path.join(texts_zip_dir,top_folder)
    zpath = os.path.join(zip_path, text_zip_filename)
    with zipfile.ZipFile(zpath, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"doc_{timestamp_str}.txt", text_data)
        zf.writestr(f"model_{timestamp_str}.pkl", ldamodel)
        zf.writestr(f"dict_{timestamp_str}.pkl", dictionary)
        zf.writestr(f"corpus_{timestamp_str}.pkl", corpus)
        zf.writestr(f"json_{timestamp_str}.pkl", text_json)
    return zpath


# Function to add new model data to metadata Parquet file
def add_model_data_to_metadata(model_data, num_documents, workers, batchsize, texts_zip_dir, metadata_dir):
    #print("we are in the add_model_data_to_metadata method()")
    # Save large body of text to zip and update model_data reference
    texts_zipped = []
    
    # Path to the distinct document folder to contain all related documents
    document_dir = os.path.join(texts_zip_dir, model_data['text_md5'])
    os.makedirs(document_dir, exist_ok=True)

    #for text_list in model_data['text']:
    for text_list in model_data['text']:
        combined_text = ''.join([''.join(sent) for sent in text_list])  # Combine all sentences into one string
        zip_path = save_to_zip(model_data['time'], document_dir, combined_text, \
                               model_data['text_json'], model_data['lda_model'], model_data['corpus'], model_data['dictionary'], texts_zip_dir)
        texts_zipped.append(zip_path)
    # Update model data with zipped paths
    model_data['text'] = texts_zipped
     # Ensure other fields are not lists, or if they are, they should have only one element per model
    for key, value in model_data.items():
        #if isinstance(value, list) and key != 'text':
        if isinstance(value, list) and key not in ['text', 'top_words']:
            assert len(value) == 1, f"Field {key} has multiple elements"
            model_data[key] = value[0]  # Unwrap single-element list
               
    # Define the expected data types for each column
    expected_dtypes = {
        'time_key': str, 
        'type': str,
        'num_workers': int,
        'batch_size': int,
        'num_documents': int,
        'text': object,  # Use object dtype for lists of strings (file paths)
        'text_json': object,
        'text_sha256': str,
        'text_md5': str,
        'convergence': 'float32',
        'perplexity': 'float32',
        'coherence': 'float32',
        'topics': int,
        # Use pd.Categorical.dtype for categorical columns
        # Ensure alpha and beta are already categorical when passed into this function
        # They should not be wrapped again with CategoricalDtype here.
        'alpha_str': str,
        'n_alpha': 'float32',
        'beta_str': str,
        'n_beta': 'float32',
        'passes': int,
        'iterations': int,
        'update_every': int,
        'eval_every': int,
        'chunksize': int,
        'random_state': int,
        'per_word_topics': bool,
        'top_words': object,
        'lda_model': object,
        'corpus': object,
        'dictionary': object,
        'create_pylda': bool, 
        'create_pcoa': bool, 
        # Enforce datetime type for time
        'time': 'datetime64[ns]',
        'end_time': 'datetime64[ns]',
    }   

    
    try:
        #df_new_metadata = pd.DataFrame({key: [value] if not isinstance(value, list) else value 
        #                                for key, value in model_data.items()}).astype(expected_dtypes)
        # Create a new DataFrame without enforcing dtypes initially
        df_new_metadata = pd.DataFrame({key: [value] if not isinstance(value, list) else value 
                                        for key, value in model_data.items()})
        
        # Apply type conversion selectively
        #for col_name in ['convergence', 'perplexity', 'coherence', 'n_beta', 'n_alpha']:
        for col_name in ['convergence', 'perplexity', 'coherence', 'n_beta', 'n_alpha']:
            df_new_metadata[col_name] = df_new_metadata[col_name].astype('float32')
            
        df_new_metadata['topics'] = df_new_metadata['topics'].astype(int)
        #df_new_metadata['time'] = pd.to_datetime(df_new_metadata['time'])
        df_new_metadata['batch_size'] = batchsize
        df_new_metadata['num_workers'] = workers
        df_new_metadata['num_documents'] = num_documents
        #df_new_metadata['create_pylda'] = pylda_success
        #df_new_metadata['create_pcoa'] = pcoa_success
        # drop lda model from dataframe
        df_new_metadata = df_new_metadata.drop('dictionary', axis=1)
        df_new_metadata = df_new_metadata.drop('corpus', axis=1)
        df_new_metadata = df_new_metadata.drop('lda_model', axis=1)
        df_new_metadata = df_new_metadata.drop('text_json', axis=1)
        df_new_metadata = df_new_metadata.drop('text', axis=1)
    except ValueError as e:
        # Initialize an error message list
        error_messages = [f"Error converting model_data to DataFrame with enforced dtypes: {e}"]
        
        
        # Iterate over each item in model_data to collect its key, expected dtype, and actual value
        for key, value in model_data.items():
            expected_dtype = expected_dtypes.get(key, 'No expected dtype specified')
            actual_dtype = type(value).__name__
            error_messages.append(f"Column: {key}, Expected dtype: {expected_dtype}, Actual dtype: {actual_dtype}, Value: {value}")
        
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)

        logging.error(full_error_message)

        raise ValueError("Data type mismatch encountered during DataFrame conversion. Detailed log available.")

    # Path to the metadata Parquet file
    parquet_file_path = os.path.join(metadata_dir, "metadata.parquet")

    # Check if the Parquet file already exists
    if os.path.exists(parquet_file_path): 
        # If it exists, read the existing metadata and append the new data 
        df_metadata = pd.read_parquet(parquet_file_path) 
        df_metadata = pd.concat([df_metadata, df_new_metadata], ignore_index=True) 
    else: 
        # If it doesn't exist, use the new data as the starting point 
        df_metadata = df_new_metadata


    # Save updated metadata DataFrame back to Parquet file
    garbage_collection(False, "add_model_data_to_metadata(...)")
    df_metadata.to_parquet(parquet_file_path)
    #del df_metadata, df_new_metadata, model_data
    #garbage_collection(False, 'add_model_data_to_metadata(...)')
    #print("\nthis is the value of the parquet file")
    #print(df_metadata)
