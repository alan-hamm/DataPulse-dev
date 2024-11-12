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


import cupy as cp
# Set up memory pooling for CuPy
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


from dask import delayed
from dask.distributed import get_client
import logging
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np
import math
from decimal import Decimal, InvalidOperation
from scipy.stats import gaussian_kde
import random


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


# Calculate mean, median, and mode using CuPy for GPU acceleration
def calculate_statistics(coherence_scores):
    coherence_array = cp.array(coherence_scores)
    mean_coherence = cp.mean(coherence_array).item()
    median_coherence = cp.median(coherence_array).item()
    mode_coherence = cp.argmax(cp.bincount(coherence_array.astype(int))).item()
    return mean_coherence, median_coherence, mode_coherence

# Sample initial documents for coherence calculation
def sample_coherence(ldamodel, documents, dictionary, sample_ratio=0.2):
    sample_size = int(len(documents) * sample_ratio)
    sample_indices = np.random.choice(len(documents), sample_size, replace=False)
    sample_docs = [documents[i] for i in sample_indices]
    coherence_model = CoherenceModel(model=ldamodel, texts=sample_docs, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence_per_topic()

@delayed
def sample_coherence_for_phase(ldamodel, documents, dictionary, sample_ratio=0.1):
    return sample_coherence(ldamodel, documents, dictionary, sample_ratio)

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

def calculate_perplexity(negative_log_likelihood, num_words):
    """
    Calculate perplexity from negative log-likelihood.

    Parameters:
    - negative_log_likelihood (float): The negative log-likelihood from log_perplexity().
    - num_words (int): The total number of words in the corpus.

    Returns:
    - float: The perplexity score.
    """
    if num_words == 0:
        raise ValueError("Number of words must be greater than zero to calculate perplexity.")
    
    return math.exp(negative_log_likelihood / num_words)

# Calculate perplexity-based threshold
def calculate_perplexity_threshold(ldamodel, documents, default_score):
    if not documents:  # Check if documents list is empty
        return default_score  # Return a default score if there are no documents
    
    perplexity = ldamodel.log_perplexity(documents)
    threshold = 0.8 * perplexity  # Adjust this multiplier based on observed correlation
    return threshold

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

# Function to replace NaN with interpolated values
def replace_nan_with_interpolated(data):
    # Calculate mean values for replacement where possible
    if math.isnan(data.get('mean_coherence', float('nan'))):
        data['mean_coherence'] = sum(data.get('coherence_scores', [])) / max(len(data.get('coherence_scores', [])), 1)
        
    if math.isnan(data.get('median_coherence', float('nan'))):
        coherence_scores = sorted(data.get('coherence_scores', []))
        mid = len(coherence_scores) // 2
        data['median_coherence'] = (coherence_scores[mid] + coherence_scores[~mid]) / 2 if coherence_scores else data['mean_coherence']
        
    if math.isnan(data.get('std_coherence', float('nan'))):
        if len(data.get('coherence_scores', [])) > 1:
            mean = data['mean_coherence']
            std_dev = (sum((x - mean) ** 2 for x in data['coherence_scores']) / (len(data['coherence_scores']) - 1)) ** 0.5
            data['std_coherence'] = std_dev
        else:
            data['std_coherence'] = 0  # Standard deviation of a single value is 0
        
    return data


# Function to handle NaN replacement with high precision and calculate coherence metrics
@delayed
def replace_nan_with_high_precision(default_score, data, ldamodel=None, dictionary=None, texts=None, tolerance=1e-5, retry_attempts=2):
    """
    Handle NaN values by recalculating coherence scores with different sample sizes if empty.
    Uses CuPy to ensure high-precision calculations and retries coherence calculation if needed.

    Parameters:
    - default_score (Decimal): Default value to use if calculation fails.
    - data (dict): Dictionary containing coherence scores and other metrics.
    - ldamodel (LdaModel): The LDA model for recalculating coherence if needed.
    - dictionary (Dictionary): The Gensim dictionary used for topic modeling.
    - texts (list): Tokenized documents for recalculating coherence if needed.
    - tolerance (float): Tolerance to filter near-zero or near-one coherence scores.
    - retry_attempts (int): Number of attempts to retry coherence calculation if scores are empty.

    Returns:
    - dict: Updated data dictionary with high-precision coherence metrics.
    """
    # Retrieve coherence scores list for calculations and convert it to a CuPy array
    coherence_scores = cp.array(data.get('coherence_scores', []), dtype=cp.float32)
    
    # Filter out values near zero or one based on tolerance
    coherence_scores = coherence_scores[~cp.isclose(coherence_scores, 0, atol=tolerance)]
    coherence_scores = coherence_scores[~cp.isclose(coherence_scores, 1, atol=tolerance)]
    
    # Retry logic if coherence scores are empty after filtering
    attempts = 0
    while coherence_scores.size == 0 and attempts < retry_attempts:
        attempts += 1
        logging.warning(f"Coherence scores list is empty after filtering. Retrying coherence calculation, attempt {attempts}.")
        
        # If ldamodel, dictionary, and texts are provided, retry coherence calculation with a larger sample ratio
        if ldamodel and dictionary and texts:
            try:
                new_sample_ratio = min(1.0, 0.2 + random.uniform(0.1, 0.3) * attempts)  # Increase the sample ratio
                coherence_scores = cp.array(sample_coherence(ldamodel, texts, dictionary, sample_ratio=new_sample_ratio), dtype=cp.float32)
                
                # Filter again after recalculating coherence scores
                coherence_scores = coherence_scores[~cp.isclose(coherence_scores, 0, atol=tolerance)]
                coherence_scores = coherence_scores[~cp.isclose(coherence_scores, 1, atol=tolerance)]
            except Exception as e:
                logging.error(f"Retry coherence calculation failed on attempt {attempts}: {e}")

    # If coherence scores are still empty, log an error and set metrics to default
    if coherence_scores.size == 0:
        logging.error("Coherence scores list is still empty after all retries.")
        data['mean_coherence'] = data['median_coherence'] = data['std_coherence'] = data['mode_coherence'] = default_score
        return data  # Return early if no coherence scores to process

    # Calculate and replace NaNs with high-precision values
    data['mean_coherence'] = float(cp.mean(coherence_scores).get())
    data['median_coherence'] = float(cp.median(coherence_scores).get())
    data['std_coherence'] = float(cp.std(coherence_scores).get())
    
    # Calculate mode, with fallback if bincount fails
    try:
        mode_value = float(cp.argmax(cp.bincount(coherence_scores.astype(int))).get())
    except ValueError:
        logging.warning("Mode calculation using bincount failed. Falling back to KDE-based mode estimation.")
        # Use KDE-based mode estimation if bincount fails
        mode_value = kde_mode_estimation(coherence_scores)

    data['mode_coherence'] = mode_value

    return data


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

@delayed
def calculate_perplexity_score(ldamodel, phase_corpus, num_words, default_score):
    try:
        negative_log_likelihood = ldamodel.log_perplexity(phase_corpus)
        return calculate_perplexity(negative_log_likelihood, num_words)
    except RuntimeWarning as e:
        logging.info(f"Issue calculating perplexity score: {e}. Value '{default_score}' assigned.")
        return default_score