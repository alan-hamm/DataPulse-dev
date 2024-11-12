# utils.py - SpectraSync: Core Utility Functions for System Optimization
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This module is the backbone of SpectraSync's operational stability, offering a suite of utility functions for 
# resource management, memory optimization, and streamlined data handling. Designed for efficiency, these utilities 
# ensure SpectraSync runs with precision and adaptability, managing everything from garbage collection to multiprocessing 
# with tools that maximize performance in high-demand environments.
#
# Functions:
# - garbage_collection: Executes garbage collection with debugging options, critical for memory stability in large datasets.
# - Additional utilities: Helper functions for data processing, logging, progress tracking, and system resource management.
#
# Dependencies:
# - Python libraries: os, logging, datetime, multiprocessing, gc, numpy, requests, tqdm
#
# Developed with AI assistance to support SpectraSync’s high-efficiency analytical framework.

import os
import logging
import json
from datetime import datetime
import multiprocessing
import gc
import numpy as np
import requests
from tqdm import tqdm
import os
import time
import shutil
from decimal import Decimal


# Define a custom encoder class for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        # Add other conversions as needed
        return super().default(obj)
            
# Helper function to ensure JSON compatibility by converting float32 and float values to native Python floats
def convert_float32_to_float(data):
    if isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_float32_to_float(value) for key, value in data.items()}
    elif isinstance(data, (np.float32, np.float64, float, Decimal)):
        return float(data)  # Convert numpy floats, Decimal, and regular floats to JSON-compatible floats
    elif isinstance(data, (np.ndarray,)):
        return [convert_float32_to_float(item) for item in data.tolist()]  # Convert numpy arrays to lists and handle recursively
    else:
        return data

def json_fallback_handler(obj):
    if isinstance(obj, (np.float32, np.float64, float)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    # Handle other types if necessary
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
 
def garbage_collection(development: bool, location: str):
    if development:
        # Enable debugging flags for leak statistics
        gc.set_debug(gc.DEBUG_LEAK)

    # Before calling collect, get a count of existing objects
    before = len(gc.get_objects())

    # Perform garbage collection
    collected = gc.collect()

    # After calling collect, get a new count of existing objects
    after = len(gc.get_objects())

    # Print or log before and after counts along with number collected
    logging.info(f"Garbage Collection at {location}:")
    logging.info(f"  Before GC: {before} objects")
    logging.info(f"  After GC: {after} objects")
    logging.info(f"  Collected: {collected} objects\n")

# Function to perform exponential backoff
def exponential_backoff(attempt, BASE_WAIT_TIME=None):
    return BASE_WAIT_TIME * (2 ** attempt)

def archive_log(lock, LOGFILE, LOG_DIRECTORY):
    """Archive log if it already exists."""
    if os.path.exists(LOGFILE):
        ARCHIVE_DIR = f"{LOG_DIRECTORY}/archived_logs"
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        creation_time = os.path.getctime(LOGFILE)
        creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d_%H-%M-%S')
        archived_logfile = os.path.join(ARCHIVE_DIR, f"log_{creation_date}.log")
        
        # Rename with a lock to prevent multiple access
        with lock:
            os.rename(LOGFILE, archived_logfile)

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def get_file_size(file_path):
    """Get the size of a local file."""
    return os.path.getsize(file_path)

def download_from_url(url, output_path):
    """Download a file from a URL with a progress bar."""
    response = requests.head(url)
    file_size = int(response.headers.get('Content-Length', 0))
    
    with requests.get(url, stream=True) as response, open(output_path, "wb") as file, tqdm(
        total=file_size, unit='B', unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
    print("Download complete.")

def process_local_file(file_path):
    """Process a local file with a progress bar."""
    file_size = get_file_size(file_path)
    
    with open(file_path, "rb") as file, tqdm(
        total=file_size, unit='B', unit_scale=True, desc="Processing"
    ) as progress_bar:
        for chunk in iter(lambda: file.read(1024), b''):
            # Simulate processing
            progress_bar.update(len(chunk))
    print("File processing complete.")


def clear_temp_files(temp_dir, age_threshold=60):
    """Clear files older than the age threshold in minutes."""
    current_time = time.time()
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            file_path = os.path.join(root, f)
            if current_time - os.path.getmtime(file_path) > age_threshold * 60:
                os.remove(file_path)


def periodic_cleanup(temp_dir, interval=1800):  # Run every 30 minutes
    while True:
        clear_temp_files(temp_dir)
        time.sleep(interval)
