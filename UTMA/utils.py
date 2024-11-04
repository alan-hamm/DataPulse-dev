# utils.py - Utility Functions for SLIF
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This script provides a collection of utility functions for the Unified Topic Modeling and Analysis (UTMA),
# including garbage collection management, multiprocessing tools, and general-purpose helper functions.
#
# Functions:
# - garbage_collection: Performs garbage collection with debugging options, useful for memory management.
# - Additional utilities: Provides miscellaneous helper functions for data processing and system resource management.
#
# Dependencies:
# - Python libraries: os, logging, datetime, multiprocessing, gc, numpy
#
# Developed with AI assistance.


import os
import logging
from datetime import datetime
import multiprocessing
import gc
import numpy as np

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

# Helper function to ensure JSON compatibility by converting float32 to float
def convert_float32_to_float(data):
    if isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_float32_to_float(value) for key, value in data.items()}
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data