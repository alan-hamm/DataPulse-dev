# written with pair programming
import logging
import gc

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
