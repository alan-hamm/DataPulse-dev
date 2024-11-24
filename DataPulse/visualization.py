# visualization.py - SpectraSync: Immersive Visualization Engine for Topic Modeling
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This module powers SpectraSync's visualization capabilities, transforming data into a sensory experience that bridges
# the analytical with the interactive. Equipped with both interactive and static plotting functions, it lets users explore
# the thematic architecture of topic models as if walking through a digital landscape. Interactive visualizations reveal 
# connections and topic coherence through pyLDAvis, while static plots generated with matplotlib serve as snapshots of 
# SpectraSync's multi-dimensional insights.
#
# Functions:
# - Interactive visualization: Deploys pyLDAvis to create immersive, interactive topic maps for in-depth exploration.
# - Static plotting: Crafts static topic plots with matplotlib, enabling users to capture the thematic distribution at a glance.
#
# Dependencies:
# - Python libraries: os, numpy, logging, pickle
# - Visualization libraries: pyLDAvis, matplotlib
#
# Developed with AI assistance to deliver SpectraSync’s visually engaging, data-driven experiences.

import torch
import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim  # Library for interactive topic model visualization
import pyLDAvis.gensim_models as gensimvis
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dask import delayed
from dask.distributed import performance_report, wait, get_client
import matplotlib
import json
import pprint as pp
import re
from decimal import Decimal
import math
from collections.abc import Iterable
import numbers
from .utils import garbage_collection
import os
import logging
import torch
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import dask
import dask.array as da
from dask import delayed, compute
from dask.distributed import get_client

matplotlib.use('Agg')


# Set max_open_warning to 0 to suppress the warning
plt.rcParams['figure.max_open_warning'] = 0 # suppress memory warning msgs re too many plots being open simultaneously

@delayed
def process_row(row, num_topics, threshold=0.001):
    if isinstance(row, list) and row:
        # Apply smoothing only to values below a specific threshold
        processed_row = [
            value if value > threshold else threshold
            for value in row if isinstance(value, (int, float))
        ]

        # Ensure the processed row has the expected number of topics
        if len(processed_row) < num_topics:
            processed_row.extend([threshold] * (num_topics - len(processed_row)))

        # Normalize the row to ensure it sums to 1
        total = sum(processed_row)
        if total > 0:
            processed_row = [value / total for value in processed_row]

        # Find the dominant topic label
        dominant_topic_label = f"Topic {processed_row.index(max(processed_row))}"
    else:
        # Default to assigning 'No Topic' if the row is not valid
        processed_row = [threshold] * num_topics
        dominant_topic_label = "No Topic"

    return processed_row, dominant_topic_label

@delayed
def process_row_v2(row, num_topics):
    """
    Process a single row of document-topic distributions to extract the dominant topic.

    Parameters:
    - row (list): List of topic probabilities for a document.
    - num_topics (int): Number of topics in the model.

    Returns:
    - tuple: (processed_row, dominant_topic_label)
    """
    try:
        # Ensure the row is valid
        if not isinstance(row, list) or len(row) != num_topics:
            logging.warning(f"Invalid row: {row}")
            return [0] * num_topics, "No Topic"

        # Normalize probabilities (ensure they sum to 1)
        total = sum(row)
        if total > 0:
            processed_row = [value / total for value in row]
        else:
            processed_row = [0] * num_topics

        # Determine the dominant topic
        max_index = processed_row.index(max(processed_row))
        dominant_topic_label = f"Topic {max_index + 1}"  # Label topics as "Topic 1", "Topic 2", etc.

        return processed_row, dominant_topic_label
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return [0] * num_topics, "No Topic"



def get_document_topics(ldamodel, bow_doc):
    try:
        topics = ldamodel.get_document_topics(bow_doc, minimum_probability=0)
        if not topics:
            logging.warning(f"No significant topics found for document: {bow_doc}")
            return [{"topic_id": None, "probability": 0}]
        return topics
    except Exception as e:
        logging.error(f"Error getting document topics for document {bow_doc}: {e}")
        return [{"topic_id": None, "probability": 0}]



def fill_distribution_matrix(ldaModel, corpus, num_topics):
    """
    Constructs a topic distribution matrix for each document in the corpus.

    This function creates a matrix where each row represents a document, and each column 
    represents a topic. The values in the matrix indicate the probability distribution of 
    topics across documents as assigned by the LDA model. The function does not use Numba 
    for optimization due to the complex object types involved in the LDA topic data.

    Parameters:
    - ldaModel: The trained LDA model used to obtain topic distributions.
    - corpus: The corpus of documents to analyze, where each document is represented as a bag-of-words.
    - num_topics: The total number of topics in the LDA model.

    Returns:
    - distributions_matrix: A NumPy array with shape (num_documents, num_topics), where each entry 
      represents the topic probability for a document.
    """
    distributions_matrix = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        doc_topics = ldaModel.get_document_topics(doc, minimum_probability=0)
        for topic_num, prob in doc_topics:
            distributions_matrix[i, topic_num] = prob
    return distributions_matrix

# The create_vis_pcoa function utilizes Principal Coordinate Analysis (PCoA) to visualize topic
# distributions based on Jensen-Shannon divergence. PCoA is beneficial for capturing complex
# distances, which can reveal nuanced topic relationships. However, PCoA can be computationally
# intensive on large datasets, so PCA may be used in certain cases for efficiency.
def create_vis_pcoa(ldaModel, corpus, topics, phase_name, filename, time_key, PCOA_DIR):
    """
    Generates a Principal Coordinate Analysis (PCoA) visualization for topic distributions.

    This function uses Jensen-Shannon divergence and Principal Coordinate Analysis (Classical 
    Multidimensional Scaling) to create a 2D visualization of topic distributions for the 
    provided LDA model and corpus. It saves the visualization as a JPEG image in the specified directory.

    Parameters:
    - ldaModel: Serialized LDA model used for topic extraction.
    - corpus: Serialized corpus of documents to analyze with the LDA model.
    - topics: Number of topics in the LDA model.
    - phase_name: Name of the analysis phase (e.g., "train" or "test") for directory organization.
    - filename: Name of the output image file.
    - time_key: Unique identifier to track timing or phase.
    - PCOA_DIR: Root directory to save PCoA visualizations.

    Returns:
    - Tuple (time_key, create_pcoa): 
        - time_key: Provided identifier to track the operation's timing or phase.
        - create_pcoa: Boolean indicating if the visualization was successfully created.
    """
    create_pcoa = None
    PCoAfilename = filename

    try:
        PCOA_DIR = os.path.join(PCOA_DIR, phase_name, f"number_of_topics-{topics.compute()}")
        os.makedirs(PCOA_DIR, exist_ok=True)
        #if os.path.exists(PCOA_DIR):
        #    logging.info(f"Confirmed that directory exists: {PCOA_DIR}")
        #else:
        #    logging.error(f"Directory creation failed for: {PCOA_DIR}")

        PCoAIMAGEFILE = os.path.join(PCOA_DIR, PCoAfilename)
    except Exception as e:
         logging.error(f"Couldn't create PCoA file: {e}")

    # try Jensen-Shannon Divergence & Principal Coordinate Analysis (aka Classical Multidimensional Scaling)
    topic_labels = [] # list to store topic labels

    ldaModel = ldaModel
    topic_distributions = [ldaModel.get_document_topics(doc, minimum_probability=0) for doc in corpus]

    # Ensure all topics are represented even if their probability is 0
    num_topics = ldaModel.num_topics
    distributions_matrix = np.zeros((len(corpus), num_topics))

    # apply topic labels and extract distribution matrix
    for i, doc_topics in enumerate(topic_distributions):
        topic_labels.append(f"Topic {max(doc_topics, key=lambda x: x[1])[0]}")
        for topic_num, prob in doc_topics:
            distributions_matrix[i, topic_num] = prob
    
    try: 
        pcoa_results = pyLDAvis.js_PCoA(distributions_matrix) 

        # Assuming pcoa_results is a NumPy array with shape (n_dists, 2)
        x = pcoa_results[:, 0]  # X-coordinates
        y = pcoa_results[:, 1]  # Y-coordinates

        # Create a figure and an axes instance
        fig, ax = plt.subplots(figsize=(10, 10))

        # Generate unique colors for each topic label using a colormap
        unique_labels = list(set(topic_labels))
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
        
        # Create a mapping from topic labels to colors
        label_to_color = dict(zip(unique_labels, colors))

        # Plot each point and assign it the color based on its label.
        scatter_plots = {}
        
        for i in range(len(x)):
            if topic_labels[i] not in scatter_plots:
                scatter_plots[topic_labels[i]] = ax.scatter(x[i], y[i], color=label_to_color[topic_labels[i]], label=topic_labels[i])
            else:
                ax.scatter(x[i], y[i], color=label_to_color[topic_labels[i]])

        # Set title and labels for axes
        ax.set_title('PCoA Results')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Add legend outside the plot to avoid covering data points.
        ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

        # Save the figure as an image file with additional padding to accommodate the legend.
        fig.savefig(f'{PCoAIMAGEFILE}.jpg', bbox_inches='tight')

        # Close the figure to free up memory
        plt.close(fig)

        create_pcoa = True

    except Exception as e: 
        logging.error(f"An error occurred during PCoA transformation: {e}")
        create_pcoa=False

    #garbage_collection(False,"create_vis(...)")
    return (time_key, create_pcoa)


#@delayed
def create_vis_pca(ldaModel, corpus, topics, phase_name, filename, time_key, PCOA_DIR):
    """
    Generates a 2D Principal Component Analysis (PCA) visualization for topic distributions.
    """
    create_pcoa = None
    PCAfilename = filename

    # Set up directory and output file paths
    try:
        # Build the output directory
        PCOA_DIR = os.path.join(PCOA_DIR, phase_name, f"number_of_topics-{topics}")
        os.makedirs(PCOA_DIR, exist_ok=True)
        PCAIMAGEFILE = os.path.join(PCOA_DIR, PCAfilename)
    except Exception as e:
        logging.error(f"Couldn't create PCoA file: {e}")
        return time_key, False

    num_topics = ldaModel.num_topics

    # Fill the topic distribution matrix
    try:
        distributions_matrix = fill_distribution_matrix(ldaModel, corpus, num_topics)
    except Exception as e:
        logging.error(f"Error filling distribution matrix: {e}")
        return time_key, False

    # Perform dimensionality reduction with PCA
    try:
        pcoa_results = PCA(n_components=2).fit_transform(distributions_matrix)
        x, y = pcoa_results[:, 0], pcoa_results[:, 1]
    except Exception as e:
        logging.error(f"An error occurred during PCA transformation: {e}")
        return time_key, False

    # Plotting and visualization
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        topic_labels = [f"Topic {max(doc_topics, key=lambda x: x[1])[0]}" for doc_topics in ldaModel.get_document_topics(corpus, minimum_probability=0)]
        unique_labels = list(set(topic_labels))
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors))

        # Scatter plot for each topic with assigned colors
        scatter_plots = {}
        for i in range(len(x)):
            if topic_labels[i] not in scatter_plots:
                scatter_plots[topic_labels[i]] = ax.scatter(x[i], y[i], color=label_to_color[topic_labels[i]], label=topic_labels[i])
            else:
                ax.scatter(x[i], y[i], color=label_to_color[topic_labels[i]])

        # Title, labels, and legend setup
        ax.set_title('PCoA Results')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

        # Attempt to save the figure
        try:
            fig.savefig(f'{PCAIMAGEFILE}.jpg', bbox_inches='tight')
            logging.info(f"Figure saved successfully to {PCAIMAGEFILE}.jpg")
        except Exception as save_error:
            logging.error(f"Failed to save figure: {save_error}")
        finally:
            plt.close(fig)

        create_pcoa = True

    except Exception as e:
        logging.error(f"Error during PCA visualization creation: {e}")
        create_pcoa = False

    return (time_key, create_pcoa)



def create_tsne_plot(document_topic_distributions_json, perplexity_score, mode_coherence, phase_name, topics, filename, time_key, pca_dir, title="tSNE Topic Distribution"):

    tSNEfilename = filename

    # Set up directory and output file paths
    try:
        pca_dir = os.path.join(pca_dir, phase_name, f"number_of_topics-{topics}")
        os.makedirs(pca_dir, exist_ok=True)
        tSNEIMAGEFILE = os.path.join(pca_dir, tSNEfilename)
    except Exception as e:
        logging.error(f"Couldn't create tSNE file: {e}")
        return time_key, False

    # initialize num_topics
    num_topics = topics

    # Initialize GPU tensor directly and prepare labels (optimized for efficiency)
    try:
        try:
            # Deserialize document topic distributions
            document_topic_distributions = json.loads(document_topic_distributions_json)
            logging.debug(f"Deserialized document_topic_distributions: {document_topic_distributions[:5]}")
        except Exception as e:
            logging.error(f"Failed to deserialize document_topic_distribution: {e}", exc_info=True)
            return time_key, False
        
        if not isinstance(document_topic_distributions, list) or not all(isinstance(row, list) for row in document_topic_distributions):
            logging.error("Deserialized input is not a list of lists.")
            return time_key, False

        if not document_topic_distributions:
            logging.error("Document topic distributions are empty after deserialization.")
            return time_key, False
     
        try:
            delayed_results = [process_row_v2(row, num_topics) for row in document_topic_distributions]
            results = compute(*delayed_results)
            processed_distributions, dominant_topics_labels = zip(*results)

            logging.debug(f"Dominant topic labels before filtering: {dominant_topics_labels[:10]}")
        except Exception as e:
            logging.error(f"Error processing document_topic_distributions: {e}", exc_info=True)
            return time_key, False
            #raise  # Stops here and provides a traceback        

        # Calculate basic statistics from the processed distributions
        all_std_devs = [np.std(dist) for dist in processed_distributions]

        # Set the coherence threshold dynamically based on data characteristics
        # Use the median standard deviation to set a relative threshold
        coherence_threshold = np.median(all_std_devs)

        valid_distributions = []
        valid_labels = []
        for dist, label in zip(processed_distributions, dominant_topics_labels):
            if sum(dist) > 0:
                coherence_score = np.std(dist)
                #if coherence_score >= coherence_threshold and coherence_score >= mode_coherence:
                if coherence_score >= coherence_threshold:
                    valid_distributions.append(dist)
                    valid_labels.append(label)

        logging.debug(f"Number of valid distributions after filtering: {len(valid_distributions)}")
        logging.debug(f"Valid labels after filtering: {valid_labels[:10]}")

        # If all rows still have insufficient coherence or variance, adjust with a fallback approach
        if not valid_distributions:
            logging.warning("All rows filtered out; falling back to the first 10 rows.")
            valid_distributions = processed_distributions[:min(10, len(processed_distributions))]
            valid_labels = dominant_topics_labels[:min(10, len(dominant_topics_labels))]

        # Convert processed distributions to a GPU tensor
        distributions_tensor = torch.tensor(valid_distributions, device='cuda', dtype=torch.float32)
        logging.debug("Successfully created GPU tensor for document distributions.")

    except Exception as e:
        logging.error(f"Failed to create GPU tensor: {e}")
        return time_key, False

    # get number of topics from deserialized JSON. This count can be different from the
    # hyperparameter number of topics due to Dask error in train_model_v2 line 388.
    num_topics = len(document_topic_distributions)
    if num_topics != topics:
        logging.warning(f"The hyperparameter topics, {topics}, does not equal the number JSON document topics {num_topics}")
        logging.warning("Using the JSON document topic count for tSNE plot generation.")

    try:
        if distributions_tensor.numel() == 0:
            logging.error("Distributions tensor is empty. Cannot proceed with variance check.")
            return time_key, False
        
        logging.debug(f"Variance before noise: {distributions_tensor.var(dim=0)}")

        # Add small Gaussian noise
        noise = torch.normal(mean=0, std=0.01, size=distributions_tensor.shape, device='cuda')
        distributions_tensor += noise

        # Add synthetic variation to avoid uniformity
        synthetic_variation = torch.rand_like(distributions_tensor) * 0.05
        distributions_tensor += synthetic_variation

        # Check variance to ensure tSNE can proceed
        variance = distributions_tensor.var(dim=0)
        logging.debug(f"Variance after noise addition: {variance}")
        logging.debug(f"Variance threshold: {variance.mean() * 0.01}")


        high_variance_columns = variance > (variance.mean() * 0.01)  # Lowered variance threshold to be less strict
        logging.debug(f"Number of high-variance columns: {high_variance_columns.sum()}")

        if high_variance_columns.sum() == 0:
            logging.warning("All columns have low variance; using fallback columns.")
            high_variance_columns = torch.ones(distributions_tensor.shape[1], dtype=torch.bool, device='cuda')
            logging.debug("Fallback columns selected for t-SNE.")

        distributions_tensor = distributions_tensor[:, high_variance_columns]
        logging.debug(f"Shape after filtering low-variance columns: {distributions_tensor.shape}")
    
        # Validate filtered tensor
        if distributions_tensor.shape[1] == 0:
            logging.error("All columns filtered out. Cannot proceed with t-SNE.")
            return time_key, False
    except Exception as e:
        logging.error(f"Failed noise addition and variance check: {e}")
        return time_key, False

    # Move tensor to CPU for TSNE processing
    distributions_tensor_cpu = distributions_tensor.cpu()
    logging.debug("Distributions tensor successfully moved to CPU for t-SNE processing.")

    # Perform tSNE and visualization
    try:
        # Ensure perplexity is less than n_samples and within a reasonable range for t-SNE
        n_samples = distributions_tensor_cpu.shape[0]
        # Use the actual perplexity score if available, otherwise default to a typical value (e.g., 30)
        actual_perplexity = max(5, min(perplexity_score, 50))  # Assuming perplexity_score is calculated from your model
        perplexity = min(actual_perplexity, n_samples - 1)  # Ensure perplexity is valid for t-SNE

        logging.debug(f"Tensor shape: {distributions_tensor_cpu.shape}")
        logging.debug(f"Variance of tensor: {distributions_tensor_cpu.var(axis=0)}")

        # Use GPU-accelerated TSNE for dimensionality reduction
        tsne_result = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42).fit_transform(distributions_tensor_cpu)
        logging.debug(f"tSNE result shape: {tsne_result.shape}")
        logging.debug(f"tSNE result sample: {tsne_result[:5]}")

        # Create DataFrame for visualization
        df = pd.DataFrame({
            'TSNE1': tsne_result[:, 0],
            'TSNE2': tsne_result[:, 1],
            'Dominant_Topic': valid_labels[:len(tsne_result)]
        })
        logging.debug(f"Sample DataFrame rows:\n{df.head()}")
        logging.debug(f"Valid labels: {valid_labels[:10]}")

        # Generate unique colors for each topic label using a colormap
        unique_labels = list(set(valid_labels))
        logging.debug(f"Number of unique topic labels: {len(unique_labels)}")
        if len(unique_labels) <= 1:
            logging.warning("Only one unique topic label found. Plot may lack color diversity.")
        
        # colormap for distinct colors for each unique topic label
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

        # Create a mapping from topic labels to colors
        label_to_color = dict(zip(unique_labels, colors))
        logging.debug(f"Label-to-color mapping: {label_to_color}")

        # Map colors to the DataFrame
        df['Color'] = df['Dominant_Topic'].map(label_to_color)

        # Create and save interactive plot
        fig = px.scatter(df, x='TSNE1', y='TSNE2', color='Dominant_Topic', hover_data={'TSNE1': False, 'TSNE2': False}, title=title)
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.write_html(f'{tSNEIMAGEFILE}.html')
        logging.info(f"Figure saved successfully to {tSNEIMAGEFILE}.html")
   
    except Exception as e:
        logging.error(f"Failed during tSNE or visualization: {e}")
        return time_key, False

    return time_key, True






#@delayed
def create_vis_pylda(ldaModel, corpus, dictionary, topics, phase_name, filename, CORES, time_key, PYLDA_DIR):
    """
    Generates an interactive HTML visualization of LDA topic distributions using pyLDAvis.
    """
    create_pylda = None

    # Set up directory and file path
    try:
        PYLDA_DIR = os.path.join(PYLDA_DIR, phase_name, f"number_of_topics-{topics}")
        os.makedirs(PYLDA_DIR, exist_ok=True)
        IMAGEFILE = os.path.join(PYLDA_DIR, f"{filename}.html")
    except Exception as e:
        logging.error(f"Couldn't create the pyLDA file: {e}")
        return time_key, False

    # Ensure data is deserialized (remove pickle.loads if data is already deserialized)
    try:
        ldaModel =ldaModel if isinstance(ldaModel, str) else ldaModel
        corpus = corpus if isinstance(corpus, str) else corpus
        dictionary =dictionary if isinstance(dictionary, str) else dictionary
    except Exception as e:
        logging.error(f"Deserialization error: {e}")
        return time_key, False
    except Exception as e:
        logging.error(f"The pyLDAvis HTML could not be saved: {e}")
        create_pylda = False

    try:
        try:
            # Generate visualization
            vis = pyLDAvis.gensim.prepare(
                ldaModel, corpus, dictionary, mds='mmds', n_jobs=int(CORES * (2 / 3)), sort_topics=False
            )
        except Exception as e:
            logging.error(f"There was an error with Gensim prepare(): {e}")
        
        try:
            # Save using pyLDAvis' standard save_html
            with open(IMAGEFILE, 'w') as f:
                f.write(pyLDAvis.prepared_data_to_html(vis))
            create_pylda = True
            logging.info(f"pyLDAvis HTML saved successfully at {IMAGEFILE}")
        except Exception as e:
            logging.error(f"There was an error saving the pyLDAvis object.")

    except Exception as e:
        logging.error(f"Error during pyLDAvis visualization creation: {e}")
        create_pylda = False

    return (time_key, create_pylda)

def process_visualizations(phase_results, phase_name, performance_log, n_topics, cores, pylda_dir, pca_dir, pca_gpu_dir):
    """
    Submits and processes visualization tasks for LDA model outputs using Dask, generating 
    interactive pyLDAvis and PCoA visualizations in parallel.

    This function iterates over LDA model results and submits separate tasks for pyLDAvis and 
    PCoA visualizations to the Dask client. Visualization tasks are executed concurrently, and 
    the function waits for their completion before gathering and returning the results. 
    Performance is logged to track resource usage during the visualization phase.

    Parameters:
    - client: Dask client used to submit and manage parallel tasks.
    - phase_results: List of dictionaries containing LDA model outputs and metadata for each document.
    - phase_name: Name of the phase (e.g., "train" or "test") to label the visualization output.
    - performance_log: Path to the log file for tracking performance metrics.
    - cores: Number of CPU cores allocated for pyLDAvis.
    - pylda_dir: Directory to save pyLDAvis HTML visualizations.
    - pcoa_dir: Directory to save PCoA image visualizations.

    Returns:
    - completed_pylda_vis: List of completed pyLDAvis visualization results.
    - completed_pcoa_vis: List of completed PCoA visualization results.

    Notes:
    - The function logs warnings if any visualization tasks are incomplete and tracks execution
      time with a performance report.
    - pyLDAvis and PCoA visualizations are handled as separate futures, allowing for 
      concurrent execution.
    """
    client = get_client()
    with performance_report(filename=performance_log):
        logging.info(f"Processing {phase_name} visualizations.")

        visualization_futures_pylda = []
        visualization_futures_pca = []
        visualization_futures_tsne = []

        processed_results = set()  # Track processed result hashes

        for result_dict in phase_results:
                # Process as usual
                unique_id = result_dict.get('time_key')
                if unique_id is None:
                    print("Warning: 'time_key' is missing in result_dict.")
                    continue
                
                try:
                    vis_future_pylda = client.submit(
                        create_vis_pylda,
                        pickle.loads(result_dict['lda_model']),
                        pickle.loads(result_dict['corpus']),
                        pickle.loads(result_dict['dictionary']),
                        n_topics, "VALIDATION", result_dict['text_md5'], cores,
                        result_dict['time_key'], pylda_dir,pure=False, retries=3
                    )
                    visualization_futures_pylda.append(vis_future_pylda)
                except Exception as e:
                    logging.error(f"Error in create_vis_pylda() Dask operation: {e}")
                    logging.error(f"TYPE: pyLDA -- MD5: {result_dict['text_md5']}")
                
                try:
                    train_tsne_vis = client.submit(
                        create_tsne_plot,
                        result_dict['validation_result'], 
                        result_dict['perplexity'],
                        result_dict['mode_coherence'],
                        phase_name,
                        n_topics,
                        result_dict['text_md5'],
                        result_dict['time_key'], 
                        pca_gpu_dir,pure=False, retries=3
                    )
                    visualization_futures_tsne.append(train_tsne_vis)
                except Exception as e:
                    logging.error(f"Error in create_tsne_plot() Dask operation: {e}")
                    logging.error("Traceback: ", exc_info=True)
                    logging.error(f"TYPE: tsNE -- MD5: {result_dict['text_md5']}") 
                    logging.debug(f"document_topic_distributions: {result_dict['document_topic_distributions']}")
                    logging.debug(f"mode_coherence: {result_dict['mode_coherence']}")
                    logging.debug(f"perplexity_score: {result_dict['perplexity_score']}")   
                    raise        
                try:
                    vis_future_pca = client.submit(
                        create_vis_pca,
                        pickle.loads(result_dict['lda_model']),
                        pickle.loads(result_dict['corpus']),
                        n_topics, phase_name, result_dict['text_md5'],
                        result_dict['time_key'], pca_dir,pure=False, retries=3
                    )
                    visualization_futures_pca.append(vis_future_pca)
                except Exception as e:
                    logging.error(f"Error in create_vis_pcoa() Dask operation: {e}")
                    logging.error(f"TYPE: PCoA -- MD5: {result_dict['text_md5']}")

        # Wait for all visualization tasks to complete
        logging.info(f"Execute WAIT on {phase_name} pyLDA visualizations: {len(visualization_futures_pylda)} futures.")
        done_viz_futures_pylda, not_done_viz_futures_pylda = wait(visualization_futures_pylda, timeout=None)
        
        logging.info(f"Execute WAIT on {phase_name} PCoA visualizations: {len(visualization_futures_pca)} futures.")
        done_viz_futures_pca, not_done_viz_futures_pca = wait(visualization_futures_pca, timeout=None)

        logging.info(f"Execute WAIT on {phase_name} PCoA visualizations: {len(visualization_futures_tsne)} futures.")
        done_viz_futures_tsne, not_done_viz_futures_tsne = wait(visualization_futures_tsne, timeout=None)

        if len(not_done_viz_futures_pylda) > 0:
            logging.error(f"Some {phase_name} pyLDA visualizations were not created: {len(not_done_viz_futures_pylda)}.")
        if len(not_done_viz_futures_pca) > 0:
            logging.error(f"Some {phase_name} PCA visualizations were not created: {len(not_done_viz_futures_pca)}.")
        if len(not_done_viz_futures_tsne) > 0:
            logging.error(f"Some {phase_name} tSNE GPU visualizations were not created: {len(not_done_viz_futures_tsne)}.")

        # resolve results from completed visualization tasks
        completed_pylda_vis = [future.result(timeout=240) for future in done_viz_futures_pylda]
        completed_pca_vis = [future.result(timeout=240) for future in done_viz_futures_pca]
        completed_tsne_vis = [future.result(timeout=240) for future in done_viz_futures_tsne]

        logging.info(f"Completed resolving {len(completed_pylda_vis) + len(completed_pca_vis) +len(completed_tsne_vis)} {phase_name} visualization futures.")

    return completed_pylda_vis, completed_pca_vis, completed_tsne_vis

