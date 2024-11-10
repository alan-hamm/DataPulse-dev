import cupy as cp
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np

# Sample initial documents for coherence calculation
def sample_coherence(ldamodel, documents, dictionary, sample_ratio=0.1):
    sample_size = int(len(documents) * sample_ratio)
    sample_indices = np.random.choice(len(documents), sample_size, replace=False)
    sample_docs = [documents[i] for i in sample_indices]
    coherence_model = CoherenceModel(model=ldamodel, texts=sample_docs, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence_per_topic()

# Calculate mean, median, and mode using CuPy for GPU acceleration
def calculate_statistics(coherence_scores):
    coherence_array = cp.array(coherence_scores)
    mean_coherence = cp.mean(coherence_array).item()
    median_coherence = cp.median(coherence_array).item()
    mode_coherence = cp.argmax(cp.bincount(coherence_array.astype(int))).item()
    return mean_coherence, median_coherence, mode_coherence

# Wrapper to sample coherence scores for a specific training phase
def sample_coherence_for_phase(ldamodel: LdaModel, documents: list, dictionary: Dictionary, sample_ratio: float):
    return sample_coherence(ldamodel, documents, dictionary, sample_ratio)

# Wrapper to get coherence statistics using GPU
def get_statistics(coherence_scores):
    # Convert coherence scores to a CuPy array
    coherence_array = cp.array(coherence_scores)
    
    # Filter out non-finite values (e.g., NaN, inf) from the array
    coherence_array = coherence_array[cp.isfinite(coherence_array)]
    
    if coherence_array.size == 0:
        # Return default values if all scores are invalid
        return float('nan'), float('nan'), float('nan')
    
    # Calculate mean, median, and mode using CuPy
    mean_coherence = cp.mean(coherence_array).item()
    median_coherence = cp.median(coherence_array).item()
    
    try:
        mode_coherence = cp.argmax(cp.bincount(coherence_array.astype(int))).item()
    except ValueError:
        # Handle case where bincount fails due to empty or invalid input
        mode_coherence = float('nan')
    
    return mean_coherence, median_coherence, mode_coherence


# Calculate perplexity-based threshold
def calculate_perplexity_threshold(ldamodel: LdaModel, documents):
    perplexity = ldamodel.log_perplexity(documents)
    threshold = 0.8 * perplexity  # Adjust this multiplier based on observed correlation
    return threshold

# Main decision logic based on coherence statistics and threshold
def coherence_score_decision(ldamodel, documents, dictionary, initial_sample_ratio=0.1):
    # Initial coherence calculation
    coherence_scores = sample_coherence_for_phase(ldamodel, documents, dictionary, initial_sample_ratio)
    mean_coherence, median_coherence, mode_coherence = get_statistics(coherence_scores)
    
    # Calculate threshold
    threshold = calculate_perplexity_threshold(ldamodel, documents)
    
    # Decision based on threshold
    if mean_coherence < threshold:
        coherence_score = mean_coherence
    else:
        # Increase sample size iteratively if coherence is above the threshold
        sample_ratio = 0.2  # Increase ratio for further coherence checks
        coherence_scores = sample_coherence_for_phase(ldamodel, documents, dictionary, sample_ratio)
        mean_coherence, median_coherence, mode_coherence = get_statistics(coherence_scores)
        coherence_score = mean_coherence  # Final coherence after expanded sample

    return coherence_score, mean_coherence, median_coherence, mode_coherence, threshold
