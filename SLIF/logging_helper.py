import os
import sys
import logging
from datetime import datetime
import io
import logging
import os
from logging.handlers import RotatingFileHandler
from filelock import FileLock

class StderrToFileHandler(logging.Handler):
    """Custom logging handler that captures stderr output to a file."""
    def __init__(self, file_path):
        super().__init__()
        self.file = open(file_path, "a")

    def emit(self, record):
        self.file.write(self.format(record) + "\n")

    def close(self):
        if not self.file.closed:
            self.file.close()
        super().close()

class SafeStream(io.TextIOWrapper):
    """A custom stream that discards writes after redirection."""
    def write(self, *args, **kwargs):
        pass  # Discards any output to prevent 'I/O operation on closed file' errors

# Configure logging with a rotating file handler to manage log files efficiently
def setup_logging(log_dir="logs", log_filename="stderr_out.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    lock_path = log_path + ".lock"  # Lock file path for file locking

    # Set up file locking to prevent race conditions
    lock = FileLock(lock_path)

    with lock:
        # Using RotatingFileHandler to rotate logs and manage size
        handler = RotatingFileHandler(log_path, maxBytes=10**6, backupCount=5)
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
        logger.addHandler(handler)

        # Optionally, remove existing handlers to avoid duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)

    return logger


def archive_log(lock, logger, stderr_file, log_file, log_directory):
    """
    Archives the stderr and log files by renaming and moving them to an archive directory,
    with lock handling and logging of lock acquisition/release.
    """
    archive_directory = os.path.join(log_directory, "archive")
    os.makedirs(archive_directory, exist_ok=True)

    # Only the critical section for renaming/moving files holds the lock
    try:
        with lock:
            logger.info("Lock acquired for archiving logs.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_stderr = os.path.join(archive_directory, f"stderr_out_{timestamp}.txt")
            archived_log = os.path.join(archive_directory, f"log_{timestamp}.log")

            # Archive stderr file if it exists
            if os.path.exists(stderr_file):
                os.rename(stderr_file, archived_stderr)
                logger.info(f"Archived stderr file to: {archived_stderr}")

            # Archive log file if it exists
            if os.path.exists(log_file):
                os.rename(log_file, archived_log)
                logger.info(f"Archived log file to: {archived_log}")

            logger.info("Lock released after archiving logs.")

    except Exception as e:
        # Log any exceptions during the archiving process for debugging purposes
        logger.error(f"Error archiving logs: {e}")
    finally:
        logger.info("archive_log function completed.")

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def close_stderr():
    # Safely redirect sys.stderr to avoid the closed file operation error
    try:
        sys.stderr.flush()  # Ensure any remaining output is written
        sys.stderr = open(os.devnull, "w")  # Redirect stderr to a harmless target
    except Exception as e:
        print(f"Error closing sys.stderr: {e}")
    finally:
        # Now attempt to close if necessary
        try:
            sys.stderr.close()
        except Exception as e:
            print(f"Error on final stderr close: {e}")