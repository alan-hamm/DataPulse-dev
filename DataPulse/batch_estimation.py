# batch_estimation.py - SpectraSync: Adaptive Batch Estimation Engine
# Author: Alan Hamm
# Date: November 2024
#
# Description:
# This module equips SpectraSync with dynamic, precision-tuned batch estimation functions, designed to optimize
# workload distribution for high-performance topic modeling. By analyzing document complexity and system capacity,
# it determines the ideal `futures_batches` for efficient and balanced processing, aligning with SpectraSync's resource-adaptive ethos.
#
# Functions:
# - estimate_futures_batches: Calculates optimal `futures_batches` for typical document sets, balancing throughput and resource use.
# - estimate_futures_batches_large_docs: Tailors batch estimation for particularly large or complex documents, maximizing system efficiency.
#
# Dependencies:
# - Python libraries: psutil, math, cupy (for GPU-accelerated computation)
#
# Developed with AI assistance to power SpectraSync’s adaptive, high-capacity processing.

import sys
import psutil
import math
import json

def estimate_batches(document, min_batch_size=5, max_batch_size=50, memory_limit_ratio=0.5, cpu_factor=2):
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


def estimate_batches_large_docs(document, min_batch_size=5, max_batch_size=50, memory_limit_ratio=0.4, cpu_factor=3):
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
    with open(document, "r") as file:
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


def estimate_batches_large_docs_v2(document_path, min_batch_size=5, max_batch_size=50, memory_limit_ratio=0.4, cpu_factor=3):
    """
    Estimates `futures_batch` size with additional adjustments for very large documents.

    Args:
        document_path (str): Path to the document with tokenized text.
        min_batch_size (int): Minimum batch count.
        max_batch_size (int): Maximum batch count.
        memory_limit_ratio (float): Fraction of memory to use.
        cpu_factor (int): Factor for CPU-based scaling.

    Returns:
        int: Estimated futures_batch size.
    """

    # Step 1: Lazy Load Document Data
    with open(document_path, 'r', encoding='utf-8') as file:
        document = json.load(file)

    # Step 2: Document Analysis
    num_elements = len(document)
    total_tokens = sum(len(element) for element in document)
    avg_tokens_per_element = total_tokens / max(1, num_elements)
    estimated_doc_size = sum(sys.getsizeof(element) for element in document)
    
    #print(f"Total tokens: {total_tokens}, Average tokens per element: {avg_tokens_per_element:.2f}")
    #print(f"Estimated document size in memory: {estimated_doc_size / (1024 ** 2):.2f} MB")

    # Step 3: System Constraints
    available_memory = psutil.virtual_memory().available * memory_limit_ratio

    # Step 4: Batch Size Estimation
    if estimated_doc_size < available_memory:
        batch_count = max(min(int(num_elements / cpu_factor), max_batch_size), min_batch_size)
    else:
        batch_count = max(int(available_memory / estimated_doc_size), min_batch_size)

    #print(f"Optimized futures_batches size for large document: {batch_count}")
    return batch_count


def estimate_batches_large_optimized(document_path, min_batch_size=5, max_batch_size=50, memory_limit_ratio=0.4, cpu_factor=3):
    """
    Estimates `futures_batch` size with additional adjustments for very large documents.

    Args:
        document_path (str): Path to the document with tokenized text.
        min_batch_size (int): Minimum batch count.
        max_batch_size (int): Maximum batch count.
        memory_limit_ratio (float): Fraction of memory to use.
        cpu_factor (int): Factor for CPU-based scaling.

    Returns:
        int: Estimated futures_batch size.
    """

    # Step 1: Lazy Load Document Data
    with open(document_path, 'r', encoding='utf-8') as file:
        document = json.load(file)

    # Step 2: Document Analysis
    num_elements = len(document)
    total_tokens = sum(len(element) for element in document)
    avg_tokens_per_element = total_tokens / max(1, num_elements)
    estimated_doc_size = sum(sys.getsizeof(element) for element in document)
    
    print(f"Total tokens: {total_tokens}, Average tokens per element: {avg_tokens_per_element:.2f}")
    print(f"Estimated document size in memory: {estimated_doc_size / (1024 ** 2):.2f} MB")

    # Step 3: System Constraints
    available_memory = psutil.virtual_memory().available * memory_limit_ratio
    num_cpus = psutil.cpu_count(logical=False)  # Use physical cores for better scaling

    # Step 4: Batch Size Estimation
    if estimated_doc_size < available_memory:
        # Scale batches based on CPU availability and cpu_factor
        batch_count = max(min(int(num_elements / (num_cpus * cpu_factor)), max_batch_size), min_batch_size)
    else:
        # Scale based on memory constraints
        batch_count = max(int(available_memory / estimated_doc_size), min_batch_size)

    print(f"Optimized futures_batches size for large document: {batch_count}")
    return batch_count


def estimate_batches_large_optimized_v2(
    document_path,
    client,
    min_batch_size=5,
    max_batch_size=50,
    memory_limit_ratio=0.4,
    cpu_factor=3
    ):
    """
    Estimates `futures_batch` size with adjustments for very large documents, 
    considering both system resources and the current state of the Dask cluster.

    Args:
        document_path (str): Path to the document containing tokenized text.
        client (dask.distributed.Client): Dask client instance to assess the cluster state.
        min_batch_size (int): Minimum batch size for processing (default: 5).
        max_batch_size (int): Maximum batch size for processing (default: 15).
        memory_limit_ratio (float): Fraction of available memory to allocate for processing (default: 0.4).
        cpu_factor (int): Scaling factor for CPU-based batch sizing (default: 3).

    Returns:
        int: Optimized batch size (`futures_batch`) considering document size, system resources,
             and the Dask cluster's current workload.

    Functionality:
        1. **Document Analysis:**
           - Reads the document and evaluates its size, token count, and average tokens per element.
        2. **System Resource Evaluation:**
           - Calculates available memory and CPU cores on the host system.
        3. **Dask Cluster State Integration:**
           - Retrieves memory usage and task load across all Dask workers using the provided client.
           - Adjusts batch size dynamically based on the cluster's free memory and task queue.
        4. **Memory-Based Scaling:**
           - Computes batch size based on free memory available in the Dask cluster.
        5. **CPU-Based Scaling:**
           - Derives batch size based on the number of elements in the document and available CPU cores.
        6. **Task Queue Adjustment:**
           - Reduces CPU-based batch size when the cluster is heavily loaded with active tasks.
        7. **Constraints:**
           - Ensures the final batch size is within the provided minimum and maximum limits.
        
    Debugging Information:
        The function prints detailed metrics during execution, including:
        - Document size and token statistics.
        - System memory and CPU availability.
        - Dask cluster memory and task load state.
        - Intermediate and final batch size calculations.

    Example Usage:
        >>> from dask.distributed import Client
        >>> client = Client()
        >>> batch_size = estimate_futures_batches_large_optimized_v2(
        ...     document_path="path/to/document.json",
        ...     client=client,
        ...     min_batch_size=10,
        ...     max_batch_size=50,
        ...     memory_limit_ratio=0.5,
        ...     cpu_factor=4
        ... )
        >>> print(f"Estimated batch size: {batch_size}")
    """
    import json
    import sys
    import psutil

    # Step 1: Lazy Load Document Data
    with open(document_path, 'r', encoding='utf-8') as file:
        document = json.load(file)

    # Step 2: Document Analysis
    num_elements = len(document)
    total_tokens = sum(len(element) for element in document)
    avg_tokens_per_element = total_tokens / max(1, num_elements)
    estimated_doc_size = sum(sys.getsizeof(element) for element in document)

    # Debugging Information
    print(f"Total tokens: {total_tokens}, Average tokens per element: {avg_tokens_per_element:.2f}")
    print(f"Estimated document size in memory: {estimated_doc_size / (1024 ** 2):.2f} MB")

    # Step 3: System Constraints (Physical System)
    available_memory = psutil.virtual_memory().available * memory_limit_ratio
    num_cpus = psutil.cpu_count(logical=False)  # Use physical cores for scaling

    # Debugging Information
    print(f"Available memory for processing: {available_memory / (1024 ** 2):.2f} MB")
    print(f"Number of physical CPUs: {num_cpus}")

    # Step 4: Dask System State
    cluster_info = client.scheduler_info()
    total_dask_memory = sum(worker["memory_limit"] for worker in cluster_info["workers"].values())
    used_dask_memory = sum(worker["memory_used"] for worker in cluster_info["workers"].values())
    free_dask_memory = total_dask_memory - used_dask_memory

    active_tasks = sum(worker["nthreads"] for worker in cluster_info["workers"].values())

    # Debugging Information
    print(f"Total Dask memory: {total_dask_memory / (1024 ** 2):.2f} MB")
    print(f"Used Dask memory: {used_dask_memory / (1024 ** 2):.2f} MB")
    print(f"Free Dask memory: {free_dask_memory / (1024 ** 2):.2f} MB")
    print(f"Active tasks across all workers: {active_tasks}")

    # Step 5: Memory-Based Scaling
    memory_batches = max(int(free_dask_memory / estimated_doc_size), 1)

    # Step 6: CPU-Based Scaling
    cpu_batches = max(int(num_elements / (num_cpus * cpu_factor)), 1)

    # Step 7: Task Queue-Based Adjustment
    task_load_factor = max(1, int(active_tasks / num_cpus))
    adjusted_cpu_batches = max(cpu_batches // task_load_factor, 1)

    # Step 8: Combined Scaling and Constraints
    batch_count = max(
        min(min(memory_batches, adjusted_cpu_batches), max_batch_size),
        min_batch_size
    )

    # Debugging Information
    print(f"Memory-based batch count: {memory_batches}")
    print(f"CPU-based batch count: {cpu_batches}")
    print(f"Adjusted CPU-based batch count (with task load factor): {adjusted_cpu_batches}")
    print(f"Optimized futures_batches size for large document: {batch_count}")

    return batch_count

