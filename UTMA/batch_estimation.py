# batch_estimation.py - Adaptive Batch Estimation for UTMA
# Author: Alan Hamm
# Date: November 2024
#
# Description:
# This script provides functions to estimate optimal batch sizes for the Unified Topic Modeling and Analysis (UTMA)
# framework. It dynamically calculates `futures_batches` based on document structure and available system resources.
# The functions consider document complexity, memory, and CPU constraints to provide efficient and balanced batch counts.
#
# Functions:
# - estimate_futures_batches: Estimates a reasonable `futures_batches` count for standard documents.
# - estimate_futures_batches_large_docs: Adjusted batch estimation for processing very large documents.
#
# Dependencies:
# - Python libraries: psutil, math
#
# Developed with AI assistance.

import psutil
import math
import json
import codecs

def estimate_futures_batches(document, min_batch_size=10, max_batch_size=100, memory_limit_ratio=0.5, cpu_factor=2):
    """
    Estimates a reasonable `futures_batch` size based on document structure and system resources.

    Args:
        document (list of lists): The document containing lists of tokenized sentences/paragraphs.
        min_batch_size (int): Minimum allowable batch count.
        max_batch_size (int): Maximum allowable batch count.
        memory_limit_ratio (float): Fraction of available memory to use.
        cpu_factor (int): Factor for scaling batch count based on CPU.

    Returns:
        int: Estimated futures_batch size.
    """

    # Load document data
    with open(document, "r") as file:
        document = json.load(file)
        
    # Step 1: Document Analysis
    num_elements = len(document)
    avg_tokens_per_element = sum(len(element) for element in document) / max(1, num_elements)
    print(f"Document has {num_elements} elements, with an average of {avg_tokens_per_element:.2f} tokens per element.")

    # Step 2: Memory Constraints
    available_memory = psutil.virtual_memory().available * memory_limit_ratio
    estimated_doc_size = num_elements * avg_tokens_per_element * 4  # ~4 bytes per token assumption

    # Step 3: Estimate Batch Count
    if estimated_doc_size < available_memory:
        # Calculate batch count based on available system resources
        batch_count = max(min(int(num_elements / avg_tokens_per_element) // cpu_factor, max_batch_size), min_batch_size)
    else:
        # Constrain if memory-limited
        batch_count = min(max(int(available_memory / (avg_tokens_per_element * 4)) // cpu_factor, min_batch_size), max_batch_size)

    print(f"Optimized futures_batches size: {batch_count}")
    return batch_count


def estimate_futures_batches_large_docs(document, min_batch_size=5, max_batch_size=50, memory_limit_ratio=0.4, cpu_factor=3):
    """
    Estimates `futures_batch` size with additional adjustments for very large documents.

    Args:
        document (list of lists): Document with tokenized text.
        min_batch_size (int): Minimum batch count.
        max_batch_size (int): Maximum batch count.
        memory_limit_ratio (float): Fraction of memory to use.
        cpu_factor (int): Factor for CPU-based scaling.

    Returns:
        int: Estimated futures_batch size.
    """

    # Load document data
    with codecs.open(document, "r", encoding='utf-8') as file:
        document = json.load(file)
        
    # Step 1: Document Analysis
    num_elements = len(document)
    total_tokens = sum(len(element) for element in document)
    avg_tokens_per_element = total_tokens / max(1, num_elements)
    print(f"Total tokens: {total_tokens}, Average tokens per element: {avg_tokens_per_element:.2f}")

    # Step 2: System Constraints
    available_memory = psutil.virtual_memory().available * memory_limit_ratio
    estimated_doc_size = total_tokens * 4  # ~4 bytes per token

    # Step 3: Batch Size Estimation
    if estimated_doc_size < available_memory:
        batch_count = max(min(int(num_elements / avg_tokens_per_element) // cpu_factor, max_batch_size), min_batch_size)
    else:
        batch_count = min(max(int(available_memory / (avg_tokens_per_element * 4)) // cpu_factor, min_batch_size), max_batch_size)

    print(f"Optimized futures_batches size for large document: {batch_count}")
    return batch_count

