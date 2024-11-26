# mathstats.py - SpectraSync: High-Precision Statistics and Coherence Calculations
# Author: Alan Hamm
# Date: November 2024
#
# Description:
# This module provides GPU-accelerated statistical computations and coherence sampling functions, crafted to optimize
# SpectraSync's analytical edge in topic modeling. Designed for high-performance data analysis, it leverages CuPy and Dask
# to deliver rapid, scalable insights into model coherence, enabling SpectraSync to stay adaptive and efficient.
#
# Functions:
# - sample_coherence: Samples coherence scores across documents, providing a snapshot of topic coherence.
# - calculate_statistics: Computes mean, median, and mode using GPU acceleration for maximum performance.
# - sample_coherence_for_phase: Wraps coherence sampling for different training phases to ensure accurate phase-specific insights.
#
# Dependencies:
# - Python libraries: CuPy (for GPU acceleration), Dask, Gensim, Numpy, and Decimal for high-precision calculations.
#
# Developed with AI assistance to enhance SpectraSync’s statistical and coherence analysis capabilities.

import os

import cupy as cp

# Set up memory pooling for CuPy
#cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
#cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

# Disable CuPy's file caching to prevent PermissionErrors on Windows
#cp.cuda.set_cub_cache_enabled(False)

# Set a unique cache directory for each worker or process (optional if caching is re-enabled)
#cache_dir = os.path.join(os.getenv('LOCALAPPDATA'), 'cupy_cache', 'worker_{}'.format(os.getpid()))
#cp.cuda.set_cub_cache_dir(cache_dir)

import torch

import re
import dask
from dask import delayed
from dask.distributed import get_client
import logging
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np
import cupy as cp
import math
from decimal import Decimal, InvalidOperation
from scipy.stats import gaussian_kde
import random
from scipy import stats
import random
from .batch_estimation import estimate_batches_large_docs_v2


import numpy as np
import cupy as cp
import logging

import torch
import numpy as np
import logging

from scipy import stats  # Import for KDE

import statistics

def get_word_count(phase_corpus, dictionary, tallied_word_count):
    """
    Determine the most accurate word count for perplexity calculation.
    
    Args:
        phase_corpus: The corpus (list of lists) processed into a bag-of-words format.
        dictionary: Gensim Dictionary associated with the corpus.
        tallied_word_count: Word count recorded during ingestion or creation of the corpus.
        
    Returns:
        An appropriate word count for perplexity calculation.
    """
    # Calculate word counts
    corpus_wrd_count = sum(count for doc in phase_corpus for _, count in doc)
    dictionary_wrd_count = dictionary.num_pos

    # Case: All counts are valid and available
    if tallied_word_count > 0 and corpus_wrd_count > 0 and dictionary_wrd_count > 0:
        if tallied_word_count == corpus_wrd_count == dictionary_wrd_count:
            return tallied_word_count  # All counts agree
        elif corpus_wrd_count == dictionary_wrd_count:
            return corpus_wrd_count  # Corpus and dictionary agree
        elif tallied_word_count == dictionary_wrd_count:
            return dictionary_wrd_count  # Tallied and dictionary agree
        elif tallied_word_count == corpus_wrd_count:
            return corpus_wrd_count  # Tallied and corpus agree
        else:
            return dictionary_wrd_count  # Default to dictionary count
    elif corpus_wrd_count > 0 and dictionary_wrd_count > 0:
        # Case: Use corpus or dictionary counts if available
        if corpus_wrd_count == dictionary_wrd_count:
            return corpus_wrd_count
        else:
            return dictionary_wrd_count  # Default to dictionary
    elif dictionary_wrd_count > 0:
        # Case: Only dictionary count is available
        return dictionary_wrd_count
    elif corpus_wrd_count > 0:
        # Case: Only corpus count is available
        return corpus_wrd_count
    elif tallied_word_count > 0:
        # Case: Only tallied count is available
        return tallied_word_count
    else:
        # Case: No valid counts
        raise ValueError("Unable to determine a valid word count for perplexity calculation.")



@delayed
def gpu_simulate_coherence_scores_with_lln(alpha, initial_size=100, max_attempts=1000, growth_factor=2, convergence_threshold=0.01, device="cuda"):
    """
    Simulate coherence scores using the Dirichlet distribution and the Law of Large Numbers (LLN) until key statistics converge.

    Parameters:
        alpha (float or list of float): The concentration parameter(s) for the Dirichlet distribution.
        initial_size (int): The initial number of coherence scores to simulate. Default is 100.
        max_attempts (int): The maximum number of iterations to stabilize coherence scores. Default is 1000.
        growth_factor (int): Factor by which to increase the sample size each iteration. Default is 2.
        convergence_threshold (float): The threshold for the relative change in statistics (mean, median, std) to determine convergence. Default is 0.01.
        device (str): The device to perform calculations on ('cuda' for GPU, 'cpu' for CPU). Default is 'cuda'.

    Returns:
        dict: A dictionary containing the following statistics:
            - "mean" (float): The mean of the coherence scores.
            - "median" (float): The median of the coherence scores.
            - "std" (float): The standard deviation of the coherence scores.
            - "mode" (float): The mode of the coherence scores, calculated using Kernel Density Estimation (KDE). Falls back to the mean if KDE fails.
    
    Raises:
        ValueError: If the coherence scores list is empty after all attempts.
        RuntimeError: If the mode calculation using KDE fails.

    Notes:
        - This function is GPU-accelerated if CUDA is available and the `device` parameter is set to "cuda".
        - The simulation stops early if the relative change in mean, median, and std is below the convergence threshold.
        - Default fallback values are returned if no valid coherence scores are generated.

    Example:
        >>> gpu_simulate_coherence_scores_with_lln(
                alpha=[0.1, 0.2, 0.3],
                initial_size=50,
                max_attempts=500,
                growth_factor=2,
                convergence_threshold=0.01,
                device="cuda"
            )
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    attempts = 0
    previous_stats = None
    sample_size = initial_size
    coherence_scores = torch.tensor([], dtype=torch.float32, device=device)

    while attempts < max_attempts:
        attempts += 1
        dirichlet_sample = torch.tensor(
            np.random.dirichlet(alpha, sample_size).flatten(), dtype=torch.float32, device=device
        )
        coherence_scores = torch.cat((coherence_scores, dirichlet_sample))

        current_stats = {
            "mean": torch.mean(coherence_scores).item(),
            "median": torch.median(coherence_scores).item(),
            "std": torch.std(coherence_scores).item(),
        }

        if previous_stats is not None:
            relative_change = {
                key: abs(current_stats[key] - previous_stats[key]) / (previous_stats[key] + 1e-10)
                for key in current_stats
            }
            if all(change < convergence_threshold for change in relative_change.values()):
                break

        previous_stats = current_stats
        sample_size *= growth_factor

    if coherence_scores.size(0) == 0:
        return {"mean": 0.5, "median": 0.5, "std": 0.0, "mode": 0.5}  # Default values

    # Calculate mode using KDE
    try:
        coherence_scores_np = coherence_scores.cpu().numpy()  # Convert to NumPy for KDE
        kde = stats.gaussian_kde(coherence_scores_np)
        mode_index = np.argmax(kde.evaluate(coherence_scores_np))
        mode = coherence_scores_np[mode_index]
    except Exception as e:
        logging.warning(f"Mode calculation using KDE failed. Falling back to mean: {e}")
        mode = current_stats["mean"]

    # Return all statistics
    return {
        "mean": current_stats["mean"],
        "median": current_stats["median"],
        "std": current_stats["std"],
        "mode": mode,
    }



@delayed
def cpu_simulate_coherence_scores_with_lln(alpha, initial_size=100, max_attempts=1000, growth_factor=2, convergence_threshold=0.01):
    """
    Simulate coherence scores using the Dirichlet distribution on the CPU until key statistics converge,
    based on the Law of Large Numbers (LLN).

    Parameters:
        alpha (list of float): The concentration parameters for the Dirichlet distribution.
        initial_size (int): The initial number of samples to generate. Default is 100.
        max_attempts (int): The maximum number of iterations for the simulation. Default is 1000.
        growth_factor (int): The factor by which the sample size increases each iteration. Default is 2.
        convergence_threshold (float): The threshold for relative change in statistics (mean, median, std) 
                                       to determine convergence. Default is 0.01.

    Returns:
        dict: A dictionary containing the following statistics:
            - "mean" (float): The mean of the simulated coherence scores.
            - "median" (float): The median of the simulated coherence scores.
            - "std" (float): The standard deviation of the simulated coherence scores.
            - "mode" (float): The mode of the simulated coherence scores, calculated using 
                              Kernel Density Estimation (KDE). Falls back to the mean if KDE fails.

    Notes:
        - This implementation performs all computations on the CPU, using `torch` for tensor operations.
        - The simulation halts early if the relative change in key statistics falls below the convergence threshold.
        - The mode is computed using KDE; if this computation fails, the mean is used as a fallback.

    Example:
        >>> cpu_simulate_coherence_scores_with_lln(
                alpha=[0.1, 0.2, 0.3],
                initial_size=50,
                max_attempts=500,
                growth_factor=2,
                convergence_threshold=0.01
            )
    """
    attempts = 0
    previous_stats = None
    sample_size = initial_size
    coherence_scores = torch.tensor([], dtype=torch.float32)

    while attempts < max_attempts:
        attempts += 1
        dirichlet_sample = torch.tensor(
            np.random.dirichlet(alpha, sample_size).flatten(), dtype=torch.float32
        )
        coherence_scores = torch.cat((coherence_scores, dirichlet_sample))

        current_stats = {
            "mean": torch.mean(coherence_scores).item(),
            "median": torch.median(coherence_scores).item(),
            "std": torch.std(coherence_scores).item(),
        }

        if previous_stats is not None:
            relative_change = {
                key: abs(current_stats[key] - previous_stats[key]) / (previous_stats[key] + 1e-10)
                for key in current_stats
            }
            if all(change < convergence_threshold for change in relative_change.values()):
                break

        previous_stats = current_stats
        sample_size *= growth_factor

    if coherence_scores.size(0) == 0:
        return {"mean": 0.5, "median": 0.5, "std": 0.0, "mode": 0.5}  # Default values

    # Calculate mode using KDE
    try:
        coherence_scores_np = coherence_scores.numpy()  # Convert to NumPy for KDE
        kde = stats.gaussian_kde(coherence_scores_np)
        mode_index = np.argmax(kde.evaluate(coherence_scores_np))
        mode = coherence_scores_np[mode_index]
    except Exception as e:
        logging.warning(f"Mode calculation using KDE failed. Falling back to mean: {e}")
        mode = current_stats["mean"]

    # Return all statistics
    return {
        "mean": current_stats["mean"],
        "median": current_stats["median"],
        "std": current_stats["std"],
        "mode": mode,
    }


@delayed
def simulate_coherence_scores_with_lln_optimized(alpha, initial_size=100, max_attempts=1000, growth_factor=2, convergence_threshold=0.01):
    """
    Simulate coherence scores using the Dirichlet distribution with an optimized implementation 
    for improved memory efficiency and batch statistics computation.

    Parameters:
        alpha (list of float): The concentration parameters for the Dirichlet distribution.
        initial_size (int): The initial number of samples to generate. Default is 100.
        max_attempts (int): The maximum number of iterations for the simulation. Default is 1000.
        growth_factor (int): The factor by which the sample size increases each iteration. Default is 2.
        convergence_threshold (float): The threshold for relative change in statistics (mean, median, std) 
                                       to determine convergence. Default is 0.01.

    Returns:
        dict: A dictionary containing the following statistics:
            - "mean" (float): The mean of the simulated coherence scores.
            - "median" (float): The median of the simulated coherence scores.
            - "std" (float): The standard deviation of the simulated coherence scores.
            - "mode" (float): The mode of the simulated coherence scores, calculated using 
                              Kernel Density Estimation (KDE). Falls back to the mean if KDE fails.

    Notes:
        - This implementation uses preallocated arrays (`np.empty`) for memory efficiency and faster computation.
        - The simulation halts early if the relative change in key statistics falls below the convergence threshold.
        - The mode is computed using KDE, and if the computation fails, the mean is used as a fallback.

    Example:
        >>> simulate_coherence_scores_with_lln_optimized(
                alpha=[0.1, 0.2, 0.3],
                initial_size=50,
                max_attempts=500,
                growth_factor=2,
                convergence_threshold=0.01
            )
    """
    attempts = 0
    previous_stats = None
    sample_size = initial_size
    total_samples = 0
    max_sample_size = sum([initial_size * (growth_factor**i) for i in range(max_attempts)])
    coherence_scores = np.empty((max_sample_size,), dtype=np.float32)
    
    while attempts < max_attempts:
        attempts += 1
        dirichlet_sample = np.random.dirichlet(alpha, size=sample_size).flatten()
        coherence_scores[total_samples:total_samples + dirichlet_sample.size] = dirichlet_sample
        total_samples += dirichlet_sample.size

        # Compute batch statistics
        mean = np.mean(coherence_scores[:total_samples])
        median = np.median(coherence_scores[:total_samples])
        std = np.std(coherence_scores[:total_samples])

        current_stats = {"mean": mean, "median": median, "std": std}
        
        if previous_stats is not None:
            relative_change = {
                key: abs(current_stats[key] - previous_stats[key]) / (previous_stats[key] + 1e-10)
                for key in current_stats
            }
            if all(change < convergence_threshold for change in relative_change.values()):
                break

        previous_stats = current_stats
        sample_size *= growth_factor

    # Compute mode using KDE
    coherence_scores = coherence_scores[:total_samples]  # Truncate to actual size
    try:
        kde = stats.gaussian_kde(coherence_scores)
        mode_index = np.argmax(kde.evaluate(coherence_scores))
        mode = coherence_scores[mode_index]
    except Exception as e:
        logging.warning(f"Mode calculation using KDE failed. Falling back to mean: {e}")
        mode = mean

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "mode": mode,
    }


# Calculate mean, median, and mode using CuPy for GPU acceleration
def calculate_statistics(coherence_scores):
    coherence_array = cp.array(coherence_scores)
    mean_coherence = cp.mean(coherence_array).item()
    median_coherence = cp.median(coherence_array).item()
    mode_coherence = cp.argmax(cp.bincount(coherence_array.astype(int))).item()
    return mean_coherence, median_coherence, mode_coherence


def calculate_value(cv1, cv2):
    """
    Calculate the cosine similarity between two vectors using CuPy for GPU acceleration,
    with error handling for invalid operations.

    This function computes the cosine similarity between two vectors `cv1` and `cv2`.
    In case of invalid operations (e.g., division by zero or `NaN` results), it returns
    `Decimal(0)` to indicate an incomplete record, which simplifies further analysis.

    Parameters:
    cv1 (cupy.ndarray): The first vector for similarity calculation.
    cv2 (cupy.ndarray): The second vector for similarity calculation.

    Returns:
    Decimal: The cosine similarity value between `cv1` and `cv2`.
             Returns `Decimal(0)` if a division error or `NaN` occurs.

    Notes:
    - The vectors should be compatible for dot product calculation.
    - Uses the `Decimal` class to ensure consistency in results, especially for database storage.
    """
    try:
        # Convert vectors to CuPy arrays
        cv1 = cp.asarray(cv1)
        cv2 = cp.asarray(cv2)

        # Calculate cosine similarity
        result = cv1.T.dot(cv2) / (cp.linalg.norm(cv1) * cp.linalg.norm(cv2))

        # Check if the result is NaN and replace it with Decimal(0)
        if cp.isnan(result):
            return Decimal(0)  # Use Decimal(0) as an indicator for an incomplete record
        return Decimal(result.get())  # Convert back to a Python scalar
    except ZeroDivisionError:
        # Handle division by zero if applicable
        return Decimal(0)  # Use Decimal(0) as an indicator for an incomplete record
    

def kde_mode_estimation(coherence_scores):
    """
    Estimate the mode using kernel density estimation (KDE) with GPU acceleration.

    Parameters:
    coherence_scores (cupy.ndarray or list): Coherence scores to estimate the mode.

    Returns:
    float: Estimated mode of the coherence scores.
    """
    # Convert coherence scores to a CuPy array if they are not already
    coherence_array = cp.asarray(coherence_scores)

    # Convert to CPU memory for KDE calculation (scipy does not directly use GPU)
    # Alternatively, we can implement KDE entirely in CuPy, but this uses scipy.stats as an example
    coherence_array_cpu = coherence_array.get()

    # Perform kernel density estimation using Gaussian KDE
    kde = gaussian_kde(coherence_array_cpu)

    # Evaluate the density over a range of values
    min_value, max_value = cp.min(coherence_array), cp.max(coherence_array)
    values = cp.linspace(min_value, max_value, 1000).get()  # Create a range of 1000 values for evaluation
    density = kde(values)

    # Find the value with the highest density as the mode estimate
    mode_index = cp.argmax(density)
    mode_value = values[mode_index]

    return float(mode_value)


#@delayed
def calculate_coherence_metrics(default_score=None, real_coherence_value=None, ldamodel=None, dictionary=None, texts=None, cores=4, tolerance=1e-5, max_attempts=5, return_torch_tensor=False):
    """
    Use the actual coherence value from the model and generate simulated coherence scores
    to calculate mean, median, std, and mode.
    """

    # initialize default-value dictionary for coherence metrics
    data = {
        'coherence_score': default_score,
        'mean_coherence': default_score,
        'median_coherence': default_score,
        'std_coherence': default_score,
        'mode_coherence': default_score
    }

    def generate_simulated_coherence(real_value):
        #logging.warning("Generating simulated coherence scores.")
        alpha = [2.0] * 75  # Set alpha to ensure enough variance
        coherence_scores = cp.random.dirichlet(alpha, size=1)[0]
        coherence_scores = 0.001 + coherence_scores * (0.9 - 0.001)  # Rescale values to match coherence score range

        # Shift the distribution towards the real coherence value for better representation
        coherence_scores = coherence_scores + real_value * 0.1
        coherence_scores = cp.clip(coherence_scores, 0.001, 0.9)  # Clip values to maintain within range

        return coherence_scores

    """
    # Retrieve actual coherence value using Gensim's CoherenceModel if model data is provided
    if ldamodel and dictionary and texts:
        try:
            coherence_model = CoherenceModel(
                model=ldamodel,
                texts=texts,
                dictionary=dictionary,
                processes=(math.ceil(cores*(2/3))),
                coherence='c_v'  # Change this to 'u_mass', 'c_uci', etc., as needed
            )
            real_coherence_value = coherence_model.get_coherence()
        except Exception as e:
            logging.warning(f"Failed to calculate real coherence value using Gensim CoherenceModel: {e}")
            real_coherence_value = default_score
    else:
        real_coherence_value = default_score
    """
    # If the real coherence value could not be calculated, use the default score
    if real_coherence_value is None:
        logging.warning("Real coherence value is None. Using default score.")
        real_coherence_value = default_score

    # Generate simulated coherence scores based on the real coherence value
    coherence_scores = generate_simulated_coherence(real_coherence_value)

    # Convert coherence scores to CuPy array
    coherence_scores = cp.array(coherence_scores, dtype=cp.float32)

    # Retry logic if coherence scores are empty (unlikely with generated data)
    attempts = 0
    while coherence_scores.size == 0 and attempts < max_attempts:
        attempts += 1
        logging.warning(f"Coherence scores list is empty. Retrying coherence calculation, attempt {attempts}.")
        coherence_scores = generate_simulated_coherence(real_coherence_value)
        coherence_scores = cp.array(coherence_scores, dtype=cp.float32)

    # If coherence scores are still empty, log an error and set metrics to default
    if coherence_scores.size == 0:
        logging.error("Coherence scores list is still empty after all retries.")
        data['mean_coherence'] = data['median_coherence'] = data['std_coherence'] = data['mode_coherence'] = default_score
        return data

    # Calculate coherence metrics using CuPy
    data['coherence_score'] = real_coherence_value
    data['mean_coherence'] = float(cp.mean(coherence_scores))
    data['median_coherence'] = float(cp.median(coherence_scores))
    data['std_coherence'] = float(cp.std(coherence_scores))

    # Convert to NumPy for KDE calculation
    coherence_scores_np = cp.asnumpy(coherence_scores)

    # Calculate mode using KDE, with fallback if KDE fails
    if len(set(coherence_scores_np)) > 1:
        try:
            kde = stats.gaussian_kde(coherence_scores_np, bw_method='silverman')
            mode_value = coherence_scores_np[np.argmax(kde.evaluate(coherence_scores_np))]
        except Exception as e:
            logging.warning(f"Mode calculation using KDE failed. Falling back to simple average: {e}")
            mode_value = float(np.mean(coherence_scores_np))
    else:
        logging.warning("Insufficient unique elements for KDE. Falling back to simple average.")
        mode_value = float(np.mean(coherence_scores_np))

    data['mode_coherence'] = mode_value

    return data


@delayed
def sample_coherence_for_phase(ldamodel, documents, dictionary, sample_ratio=0.1, num_samples=5, distribution="uniform"):
    """
    Calculate coherence scores for multiple samples of the documents.

    Parameters:
    - num_samples (int): The number of different samples to calculate coherence scores for.
    - distribution (str): Type of distribution to use for sampling ('uniform', 'normal', 'beta').
    """
    coherence_scores = []
    available_indices = list(range(len(documents)))

    for _ in range(num_samples):
        if distribution == "normal":
            # Use a normal distribution to generate indices
            mean = len(documents) / 2
            std_dev = len(documents) / 6  # Adjust standard deviation as needed
            sample_indices = [int(cp.random.normal(mean, std_dev)) % len(documents) for _ in range(int(len(documents) * sample_ratio))]
            sample_indices = [i for i in sample_indices if 0 <= i < len(documents)]
        elif distribution == "beta":
            # Use a beta distribution to generate indices
            alpha, beta = 2, 5  # Adjust parameters for desired distribution
            sample_indices = [int(len(documents) * cp.random.beta(alpha, beta)) for _ in range(int(len(documents) * sample_ratio))]
        else:  # Default to uniform distribution
            sample_size = int(len(documents) * sample_ratio)
            sample_indices = cp.random.choice(available_indices, sample_size, replace=False)

        # Ensure sample size is valid
        if len(sample_indices) == 0:
            logging.warning("Sample size is zero. Skipping sampling.")
            continue

        sample_docs = [documents[i] for i in sample_indices]
        coherence_score = init_sample_coherence(ldamodel, sample_docs, dictionary, sample_ratio)
        if coherence_score is not None:
            coherence_scores.append(coherence_score)

    return coherence_scores

# Sample initial documents for coherence calculation
@delayed
def init_sample_coherence(ldamodel, documents, dictionary, sample_ratio=0.2):
    # Debugging statement: Print or log the type of documents before sampling
    if not all(isinstance(doc, list) for doc in documents):
        raise ValueError(f"Expected all documents to be lists of tokens, but found non-list entries in documents: {documents}")

    sample_size = int(len(documents) * sample_ratio)
    sample_indices = cp.random.choice(len(documents), sample_size, replace=False)
    sample_docs = [documents[i] for i in sample_indices]

    # Debugging statement: Print or log the type of sample_docs after sampling
    for idx, doc in enumerate(sample_docs):
        if not isinstance(doc, list):
            raise ValueError(f"Expected sampled document at index {idx} to be a list of tokens, but got: {type(doc)}")

    # Use the new PyTorch-based coherence calculation function
    coherence_score = calculate_torch_coherence(ldamodel, sample_docs, dictionary)

    return coherence_score

def calculate_dynamic_coherence(ldamodel, batch_documents, train_dictionary_batch, default_score, max_attempts=5, convergence_tolerance=0.01):
    """
    Calculate coherence scores dynamically, adjusting sample ratios until they normalize.
    """
    try:
        sample_ratio = 0.1
        num_samples = 5
        previous_mean = None
        current_mean = None
        iterations = 0

        while iterations < max_attempts:
            # Create a delayed task for sample coherence scores calculation
            coherence_scores_task = sample_coherence_for_phase(
                ldamodel, batch_documents, train_dictionary_batch, sample_ratio=sample_ratio, num_samples=num_samples, distribution="normal"
            )

            # Process coherence scores with high precision after computation
            coherence_scores_data = calculate_coherence_metrics(
                default_score, {'coherence_scores': coherence_scores_task}, ldamodel=ldamodel,
                dictionary=train_dictionary_batch, texts=batch_documents, tolerance=1e-5, max_attempts=max_attempts
            )

            # Extract metrics from processed data as delayed tasks
            current_mean = delayed(lambda data: data['mean_coherence'])(coherence_scores_data)

            # Run compute to evaluate the current mean coherence score
            current_mean = dask.compute(current_mean)[0]

            # Check for convergence using the Law of Large Numbers
            if previous_mean is not None:
                change = abs(current_mean - previous_mean)
                if change < convergence_tolerance:
                    logging.info(f"Convergence achieved after {iterations} iterations with a change of {change:.5f}")
                    break

            # Update for the next iteration
            previous_mean = current_mean
            sample_ratio = min(1.0, sample_ratio + 0.1)  # Gradually increase the sample ratio
            iterations += 1

        # Extract final coherence metrics
        coherence_scores_data = calculate_coherence_metrics(
            default_score, {'coherence_scores': coherence_scores_task}, ldamodel=ldamodel,
            dictionary=train_dictionary_batch, texts=batch_documents, tolerance=1e-5, max_attempts=max_attempts
        )
        mean_coherence = delayed(lambda data: data['mean_coherence'])(coherence_scores_data)
        median_coherence = delayed(lambda data: data['median_coherence'])(coherence_scores_data)
        std_coherence = delayed(lambda data: data['std_coherence'])(coherence_scores_data)
        mode_coherence = delayed(lambda data: data['mode_coherence'])(coherence_scores_data)

        # Run compute here if everything is successful
        mean_coherence, median_coherence, std_coherence, mode_coherence = dask.compute(
            mean_coherence, median_coherence, std_coherence, mode_coherence
        )

        return mean_coherence, median_coherence, std_coherence, mode_coherence

    except Exception as e:
        logging.error(f"Error calculating coherence: {e}")
        return default_score, default_score, default_score, default_score
    

# PyTorch-based coherence calculation function
@delayed
def calculate_torch_coherence(data_source, ldamodel, sample_docs, dictionary):
    # Ensure sample_docs is a list of tokenized documents
    if isinstance(sample_docs, tuple):
        sample_docs = list(sample_docs)

    #if not ldamodel:
    #    try:
    #        temp_dir = os.path.expanduser("~/temp/datapulse/model.model")
    #        if os.path.exists(temp_dir):
    #            ldamodel = LdaModel.load(temp_dir)
    #        else:
    #            logging.info("No cached pre-trained model found at the specified path.")
    #    except Exception as e:
    #        raise FileNotFoundError(f"Could not load cached or trained model.")

    
    batch_size = estimate_batches_large_docs_v2(data_source, min_batch_size=1, max_batch_size=50, memory_limit_ratio=0.4, cpu_factor=3)

    # Split sample_docs into smaller batches
    batches = [sample_docs[i:i + batch_size] for i in range(0, len(sample_docs), batch_size)]

    coherence_scores = []

    # Validate and sanitize the sample_docs
    for idx, doc in enumerate(sample_docs):
        if isinstance(doc, list):
            if not all(isinstance(token, str) for token in doc):
                logging.error(f"Document at index {idx} contains non-string tokens: {doc}")
                raise ValueError(f"Document at index {idx} contains non-string tokens.")
        elif isinstance(doc, str):
            doc = doc.split()
            sample_docs[idx] = doc
        else:
            logging.error(f"Expected a list of tokens or a string, but got: {type(doc)} with value {doc} at index {idx}.")
            raise ValueError(f"Expected a list of tokens or a string, but got: {type(doc)} at index {idx}.")

        if len(doc) == 0:
            logging.error(f"Document at index {idx} is empty after processing.")
            raise ValueError(f"Document at index {idx} is empty after processing.")

        # Ensure all tokens are strings
        sample_docs[idx] = [str(token) for token in doc]

    # Convert documents to topic vectors using the LDA model
    for batch in batches:
        try:
            # Convert documents in the batch to topic vectors using the LDA model
            topic_vectors = [
                ldamodel.get_document_topics(dictionary.doc2bow(doc), minimum_probability=0.01)
                for doc in batch
            ]

            # Convert to dense vectors suitable for tensor operations
            num_topics = ldamodel.num_topics
            dense_topic_vectors = torch.zeros((len(topic_vectors), num_topics), dtype=torch.float32, device='cuda')

            for i, doc_topics in enumerate(topic_vectors):
                for topic_id, prob in doc_topics:
                    dense_topic_vectors[i, topic_id] = torch.tensor(prob, dtype=torch.float32, device='cuda')

            # Calculate cosine similarity on the GPU
            similarities = torch.nn.functional.cosine_similarity(
                dense_topic_vectors.unsqueeze(1), dense_topic_vectors.unsqueeze(0), dim=-1
            )

            # Calculate coherence score by averaging the cosine similarities for the batch
            batch_coherence_score = torch.mean(similarities).item()
            coherence_scores.append(batch_coherence_score)

        except Exception as e:
            logging.error(f"Error processing batch for coherence calculation: {e}")
            coherence_scores.append(0)  # Append a default score in case of failure
    
    # Calculate overall coherence score by averaging all batch scores
    overall_coherence_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else float('nan')
    return overall_coherence_score


@delayed
def get_statistics(coherence_scores):
    coherence_array = cp.array(coherence_scores)
    coherence_array = coherence_array[cp.isfinite(coherence_array)]
    
    if coherence_array.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    
    mean_coherence = cp.mean(coherence_array).item()
    median_coherence = cp.median(coherence_array).item()
    std_coherence = cp.std(coherence_array).item()
    
    try:
        mode_coherence = cp.argmax(cp.bincount(coherence_array.astype(int))).item()
    except ValueError:
        mode_coherence = float('nan')
    
    return mean_coherence, median_coherence, mode_coherence, std_coherence


# Main decision logic based on coherence statistics and threshold
def coherence_score_decision(ldamodel, documents, dictionary, initial_sample_ratio=0.1):
    # Sample coherence scores
    coherence_scores = sample_coherence_for_phase(ldamodel, documents, dictionary, initial_sample_ratio)
    
    # Calculate coherence statistics
    mean_coherence, median_coherence, mode_coherence, std_coherence = get_statistics(coherence_scores)
    
    # Calculate perplexity threshold
    try:
        threshold = calculate_perplexity_threshold(ldamodel, documents, dictionary)
    except Exception as e:
        logging.error(f"Error calculating perplexity threshold: {e}")
        threshold = float('nan')
    
    # Check coherence threshold
    coherence_score = mean_coherence if mean_coherence < threshold else float('nan')
    
    # Ensure all values are returned
    return (coherence_score, mean_coherence, median_coherence, mode_coherence, std_coherence, threshold)


@delayed
def compute_full_coherence_score(ldamodel, dictionary, texts, cores):
    coherence_model_lda = CoherenceModel(
        model=ldamodel, 
        dictionary=dictionary, 
        texts=texts, 
        coherence='c_v',
        processes=math.floor(cores * (1/3))
    )
    return coherence_model_lda.get_coherence()

@delayed
def calculate_convergence(ldamodel, phase_corpus, default_score):
    try:
        return ldamodel.bound(phase_corpus)
    except Exception as e:
        logging.error(f"Issue calculating convergence score: {e}. Value '{default_score}' assigned.")
        return default_score


def simulate_corpus(dictionary, num_docs=100, avg_doc_len=50):
    """
    Simulate a synthetic corpus based on an existing dictionary.
    
    Args:
        dictionary: Gensim Dictionary object.
        num_docs: Number of documents to simulate.
        avg_doc_len: Average number of words per document.
    
    Returns:
        list of list: A synthetic corpus in tokenized format.
    """
    vocab = list(dictionary.token2id.keys())
    synthetic_corpus = [
        random.choices(vocab, k=avg_doc_len) for _ in range(num_docs)
    ]
    return synthetic_corpus

def simulate_negative_log_likelihood(dictionary, num_topics=10, num_docs=100, avg_doc_len=50):
    """
    Simulate a realistic negative log-likelihood using a synthetic corpus and LDA model.
    
    Args:
        dictionary: Gensim Dictionary object.
        num_topics: Number of topics for the LDA model.
        num_docs: Number of documents to simulate.
        avg_doc_len: Average document length for the synthetic corpus.
    
    Returns:
        float: Simulated negative log-likelihood.
    """
    # Generate a synthetic corpus
    synthetic_corpus = simulate_corpus(dictionary, num_docs=num_docs, avg_doc_len=avg_doc_len)
    
    # Train a small LDA model on the synthetic corpus
    synthetic_bow = [dictionary.doc2bow(doc) for doc in synthetic_corpus]
    lda_model = LdaModel(synthetic_bow, num_topics=num_topics, id2word=dictionary, passes=2)
    
    # Calculate the negative log-likelihood
    return lda_model.log_perplexity(synthetic_bow)


def calculate_negative_log_likelihood(ldamodel, phase_corpus, dictionary, default_score=None):
    """
    Calculate the perplexity score for a given phase_corpus.

    Parameters:
    - ldamodel: The trained LDA model.
    - phase_corpus: The corpus for the current phase.
    - dictionary: Gensim dictionary object.
    - default_score: The default score to return in case of failure.

    Returns:
    - float: The calculated perplexity score or the default score.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            negative_log_likelihood = ldamodel.log_perplexity(phase_corpus)

            if not np.isfinite(negative_log_likelihood):
                raise ValueError(f"Non-finite negative log-likelihood: {negative_log_likelihood}")

            logging.debug(f"[calculate_negative_log_likelihood] Negative log-likelihood: {negative_log_likelihood}")
            return negative_log_likelihood
        except ValueError as ve:
            logging.error(f"[calculate_negative_log_likelihood] ValueError: {ve}. Returning simulated score.")
            return simulate_negative_log_likelihood(dictionary)
        except Exception as e:
            logging.error(f"[calculate_negative_log_likelihood] Unexpected error: {e}. Returning simulated score.")
            return simulate_negative_log_likelihood(dictionary)



def calculate_perplexity(negative_log_likelihood, phase_corpus, dictionary, number_of_words, default_score=30.0):
    try:
        total_words = get_word_count(phase_corpus, dictionary, number_of_words)
        if total_words <= 1:
            logging.warning(f"Word count is too low for reliable perplexity calculation: {total_words}. Returning default score.")
            return default_score

        perplexity = math.exp(negative_log_likelihood / total_words)
        logging.debug(f"Calculated perplexity: {perplexity}")

        # Validate perplexity for t-SNE
        if perplexity > len(phase_corpus):
            logging.warning(f"Perplexity {perplexity} exceeds number of samples {len(phase_corpus)}. Using number of samples.")
            return float(len(phase_corpus))
        elif perplexity < 5:
            logging.warning(f"Perplexity {perplexity} is below the recommended range for t-SNE. Adjusting to minimum value of 5.")
            return 5.0
        else:
            return float(perplexity)
    except Exception as e:
        logging.error(f"Error during perplexity calculation: {e}. Returning default value.")
        return default_score

def get_tsne_perplexity_param(
    phase_corpus, 
    dictionary, 
    tallied_word_count, 
    ldamodel, 
    num_docs=100, 
    avg_doc_len=50, 
    default_perplexity=30.0, 
    precomputed_simulated_nll=None
):
    """
    Calculate the t-SNE perplexity parameter.

    Args:
        phase_corpus: Corpus for the current phase, in bag-of-words format.
        dictionary: Gensim Dictionary object for the corpus.
        tallied_word_count: Word count recorded during corpus ingestion.
        ldamodel: The trained LDA model.
        num_docs: Number of documents for simulated corpus (default: 100).
        avg_doc_len: Average length of simulated documents (default: 50).
        default_perplexity: Default perplexity value in case of failure (default: 30.0).
        precomputed_simulated_nll: Precomputed simulated negative log-likelihood (optional).
    
    Returns:
        float: A valid t-SNE perplexity parameter.
    """
    try:
        # Step 1: Get the word count
        word_count = get_word_count(phase_corpus, dictionary, tallied_word_count)
        logging.debug(f"Word count for perplexity calculation: {word_count}")

        # Step 2: Use the provided simulated NLL or generate a new one
        if precomputed_simulated_nll is not None:
            simulated_nll = precomputed_simulated_nll
            logging.debug("Using precomputed simulated NLL.")
        else:
            simulated_nll = simulate_negative_log_likelihood(dictionary, num_docs=num_docs, avg_doc_len=avg_doc_len)
            logging.debug(f"Generated simulated NLL: {simulated_nll}")

        # Step 3: Attempt to calculate the actual NLL using the LDA model
        try:
            calculated_nll = calculate_negative_log_likelihood(ldamodel, phase_corpus, dictionary, default_score=simulated_nll)
            logging.debug(f"Calculated negative log-likelihood: {calculated_nll}")
        except Exception as e:
            logging.error(f"Error calculating NLL from LDA model: {e}. Falling back to simulated NLL.")
            calculated_nll = simulated_nll

        # Step 4: Calculate t-SNE-specific perplexity
        tsne_perplexity = calculate_perplexity(calculated_nll, phase_corpus, dictionary, word_count, default_score=default_perplexity)
        logging.info(f"t-SNE perplexity parameter: {tsne_perplexity}")

        return tsne_perplexity

    except Exception as e:
        logging.error(f"Failed to calculate t-SNE perplexity parameter: {e}. Returning default value.")
        return default_perplexity

