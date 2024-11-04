# logging_helper.py - Logging Setup and Configuration Functions
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This script defines helper functions to set up structured logging for the Unified Topic Modeling and Analysis (UTMA),
# including directory creation, log rotation, and file locking to ensure concurrency-safe logging.
#
# Functions:
# - setup_logging: Configures a rotating file handler and clears existing handlers to prevent duplicates.
#
# Dependencies:
# - Python libraries: logging, os, datetime, shutil
# - External libraries: filelock (for handling concurrent logging)
#
# Developed with AI assistance.

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from filelock import FileLock
import shutil

def setup_logging(log_dir="logs", log_filename="log.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # Get the logger and clear existing handlers
    logger = logging.getLogger("topic_analysis_logger")
    if logger.hasHandlers():
        logger.handlers.clear()  # Clear to prevent duplicate handlers

    # Configure RotatingFileHandler
    handler = RotatingFileHandler(log_path, maxBytes=10**6, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Attach the handler to the logger
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

def archive_log(logger, log_file, log_directory):
    archive_directory = os.path.join(log_directory, "archive")
    os.makedirs(archive_directory, exist_ok=True)

    logger.info("Archiving log file.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_log = os.path.join(archive_directory, f"log_{timestamp}.log")
    
    # Close and remove all handlers
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    # Copy instead of renaming to prevent permission issues
    if os.path.exists(log_file):
        shutil.copy2(log_file, archived_log)
        open(log_file, 'w').close()  # Clear the contents of the original log file
        logger.info(f"Archived log file to: {archived_log}")
    
    # Re-add the file handler to start a new log file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Close logger function
def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
