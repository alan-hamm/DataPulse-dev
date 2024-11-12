# visualization.py - Visualization Tools for SLIF Topic Modeling
# Author: Alan Hamm
# Date: April 2024
#
# Description:
# This script provides visualization tools for exploring and analyzing topic models generated within the
# Unified Topic Modeling and Analysis (UTMA). It includes functions for interactive visualizations and static
# plotting, leveraging pyLDAvis and matplotlib.
#
# Functions:
# - Interactive visualization: Uses pyLDAvis for interactive exploration of LDA topics.
# - Static plotting: Configures and manages matplotlib plots for visual representation of topics.
#
# Dependencies:
# - Python libraries: os, numpy, logging, pickle
# - Visualization libraries: pyLDAvis, matplotlib
#
# Developed with AI assistance.

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim  # Library for interactive topic model visualization
import pyLDAvis.gensim_models as gensimvis
from sklearn.decomposition import PCA
from dask import delayed
from dask.distributed import performance_report, wait, get_client
import matplotlib
from .utils import garbage_collection

matplotlib.use('Agg')


# Set max_open_warning to 0 to suppress the warning
plt.rcParams['figure.max_open_warning'] = 0 # suppress memory warning msgs re too many plots being open simultaneously

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

    ldaModel = pickle.loads(ldaModel)
    topic_distributions = [ldaModel.get_document_topics(doc, minimum_probability=0) for doc in pickle.loads(corpus)]

    # Ensure all topics are represented even if their probability is 0
    num_topics = ldaModel.num_topics
    distributions_matrix = np.zeros((len(pickle.loads(corpus)), num_topics))

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

@delayed
def create_vis_pca(ldaModel, corpus, topics, phase_name, filename, time_key, PCOA_DIR):
    """
    Generates a 2D Principal Component Analysis (PCA) visualization for topic distributions.
    """
    create_pcoa = None
    PCoAfilename = filename

    # Set up directory and output file paths
    try:
        # Build the output directory
        PCOA_DIR = os.path.join(PCOA_DIR, phase_name, f"number_of_topics-{topics}")
        os.makedirs(PCOA_DIR, exist_ok=True)
        PCoAIMAGEFILE = os.path.join(PCOA_DIR, PCoAfilename)
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
            fig.savefig(f'{PCoAIMAGEFILE}.jpg', bbox_inches='tight')
            logging.info(f"Figure saved successfully to {PCoAIMAGEFILE}.jpg")
        except Exception as save_error:
            logging.error(f"Failed to save figure: {save_error}")
        finally:
            plt.close(fig)

        create_pcoa = True

    except Exception as e:
        logging.error(f"Error during PCA visualization creation: {e}")
        create_pcoa = False

    return (time_key, create_pcoa)


@delayed
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
        ldaModel = pickle.loads(ldaModel) if isinstance(ldaModel, bytes) else ldaModel
        corpus = pickle.loads(corpus) if isinstance(corpus, bytes) else corpus
        dictionary = pickle.loads(dictionary) if isinstance(dictionary, bytes) else dictionary
    except Exception as e:
        logging.error(f"Deserialization error: {e}")
        return time_key, False
    except Exception as e:
        logging.error(f"The pyLDAvis HTML could not be saved: {e}")
        create_pylda = False

    try:
        # Generate visualization
        vis = pyLDAvis.gensim.prepare(
            ldaModel, corpus, dictionary, mds='mmds', n_jobs=int(CORES * (2 / 3)), sort_topics=False
        )

        # Save using pyLDAvis' standard save_html
        with open(IMAGEFILE, 'w') as f:
            f.write(pyLDAvis.prepared_data_to_html(vis))
        create_pylda = True
        logging.info(f"pyLDAvis HTML saved successfully at {IMAGEFILE}")

    except Exception as e:
        logging.error(f"Error during pyLDAvis visualization creation: {e}")
        create_pylda = False

    return (time_key, create_pylda)

def process_visualizations(client, phase_results, phase_name, performance_log, cores, pylda_dir, pcoa_dir):
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
    with performance_report(filename=performance_log):
        logging.info(f"Processing {phase_name} visualizations.")

        visualization_futures_pylda = []
        visualization_futures_pcoa = []

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
                        result_dict['lda_model'],
                        result_dict['corpus'],
                        result_dict['dictionary'],
                        result_dict['topics'],
                        phase_name,
                        result_dict['text_md5'],  # filename
                        cores,
                        result_dict['time_key'],
                        pylda_dir
                    )
                    visualization_futures_pylda.append(vis_future_pylda)
                except Exception as e:
                    logging.error(f"Error in create_vis_pylda() Dask operation: {e}")
                    logging.error(f"TYPE: pyLDA -- MD5: {result_dict['text_md5']}")
                
                try:
                    vis_future_pcoa = client.submit(
                        create_vis_pca,
                        result_dict['lda_model'],
                        result_dict['corpus'],
                        result_dict['topics'],  # f'number_of_topics-{topics}'
                        phase_name,
                        result_dict['text_md5'],  # filename
                        result_dict['time_key'],
                        pcoa_dir
                    )
                    visualization_futures_pcoa.append(vis_future_pcoa)
                except Exception as e:
                            logging.error(f"Error in create_vis_pcoa() Dask operation: {e}")
                            logging.error(f"TYPE: PCoA -- MD5: {result_dict['text_md5']}")

        # Wait for all visualization tasks to complete
        logging.info(f"Executing WAIT on {phase_name} pyLDA visualizations: {len(visualization_futures_pylda)} futures.")
        done_viz_futures_pylda, not_done_viz_futures_pylda = wait(visualization_futures_pylda)
        
        logging.info(f"Executing WAIT on {phase_name} PCoA visualizations: {len(visualization_futures_pcoa)} futures.")
        done_viz_futures_pcoa, not_done_viz_futures_pcoa = wait(visualization_futures_pcoa)

        if len(not_done_viz_futures_pylda) > 0:
            logging.error(f"Some {phase_name} pyLDA visualizations were not created: {len(not_done_viz_futures_pylda)}.")
        if len(not_done_viz_futures_pcoa) > 0:
            logging.error(f"Some {phase_name} PCoA visualizations were not created: {len(not_done_viz_futures_pcoa)}.")

        # Gather results from completed visualization tasks
        completed_pylda_vis = [future.result() for future in done_viz_futures_pylda]
        completed_pcoa_vis = [future.result() for future in done_viz_futures_pcoa]

        logging.info(f"Completed gathering {len(completed_pylda_vis) + len(completed_pcoa_vis)} {phase_name} visualization results.")

    return completed_pylda_vis, completed_pcoa_vis

