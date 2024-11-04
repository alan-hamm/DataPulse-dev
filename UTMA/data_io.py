# Developed traditionally with the addition of AI assistance
# Author: Alan Hamm
# Date: April 2024

import os
from json import load
from random import shuffle
import pandas as pd 
from .utils import garbage_collection

#def get_num_records(filename):
#    with open(filename, 'r', encoding='utf-8', errors='ignore') as jsonfile:
##        data = load(jsonfile)
#        data = data
#        num_samples = len(data)  # Count the total number of samples
#    return num_samples

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
            eval_indices_batch = indices[eval_count:eval_count + batch_size]
            eval_data_batch = [data[idx] for idx in eval_indices_batch]
            if len(eval_data_batch) > 0:
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
    #garbage_collection(False, "futures_create_lda_datasets(..)")
