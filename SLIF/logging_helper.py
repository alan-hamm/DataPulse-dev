import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from filelock import FileLock

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

def archive_log(lock, logger, log_file, log_directory):
    archive_directory = os.path.join(log_directory, "archive")
    os.makedirs(archive_directory, exist_ok=True)

    with lock:
        logger.info("Archiving log file.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_log = os.path.join(archive_directory, f"log_{timestamp}.log")
        
        # Close all handlers to fully release the log file
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        
        # Rename and move the log file now that it's released
        if os.path.exists(log_file):
            os.rename(log_file, archived_log)
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
