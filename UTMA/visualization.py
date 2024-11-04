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


from .utils import garbage_collection
import os 
import numpy as np
import pyLDAvis
import pyLDAvis.gensim  # Library for interactive topic model visualization
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import matplotlib
import pickle
from dask.distributed import performance_report, wait
matplotlib.use('Agg')

import logging
# Set max_open_warning to 0 to suppress the warning
plt.rcParams['figure.max_open_warning'] = 0 # suppress memory warning msgs re too many plots being open simultaneously


def create_vis_pylda(ldaModel, corpus, dictionary, topics, phase_name, filename, CORES, PYLDA_DIR):
    create_pylda = None
    #print("We are inside Create Vis.")
    try:
        PYLDA_DIR = os.path.join(PYLDA_DIR, phase_name, f"number_of_topics-{topics}")
        os.makedirs(PYLDA_DIR, exist_ok=True)
        if os.path.exists(PYLDA_DIR):
            logging.info(f"Confirmed that directory exists: {PYLDA_DIR}")
        else:
            logging.error(f"Directory creation failed for: {PYLDA_DIR}")

        IMAGEFILE = os.path.join(PYLDA_DIR,f"{filename}.html")
    except Exception as e:
         logging.error(f"Couldn't create the pyLDA file: {e}")

    # Prepare the visualization data.
    # Note: sort_topics=False will prevent reordering topics after training.
    try:
        # ERROR: Object of type complex128 is not JSON serializable
        # https://github.com/bmabey/pyLDAvis/issues/69#issuecomment-311337191
        # as mentioned in the forum, use mds='mmds' instead of default js_PCoA
        # https://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepare
        ldaModel = pickle.loads(ldaModel)
        corpus = pickle.loads(corpus)
        dictionary = pickle.loads(dictionary)
        vis = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary,  mds='mmds', n_jobs=int(CORES*(2/3)), sort_topics=False)

        pyLDAvis.save_html(vis, IMAGEFILE)
        create_pylda = True

    except Exception as e:
        logging.error(f"The pyLDAvis HTML could not be saved: {e}")
        create_pylda = False

    #garbage_collection(False,"create_vis(...)")
    return (filename, create_pylda)


def create_vis_pcoa(ldaModel, corpus, topics, phase_name, filename, PCOA_DIR):
    create_pcoa = None
    PCoAfilename = filename

    try:
        PCOA_DIR = os.path.join(PCOA_DIR, phase_name, f"number_of_topics-{topics}")
        os.makedirs(PCOA_DIR, exist_ok=True)
        if os.path.exists(PCOA_DIR):
            logging.info(f"Confirmed that directory exists: {PCOA_DIR}")
        else:
            logging.error(f"Directory creation failed for: {PCOA_DIR}")

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
    return (filename, create_pcoa)


def process_visualizations(client, phase_results, phase_name, performance_log, cores, pylda_dir, pcoa_dir):
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
                        pylda_dir
                    )
                    visualization_futures_pylda.append(vis_future_pylda)
                except Exception as e:
                    logging.error(f"Error in create_vis_pylda() Dask operation: {e}")
                    logging.error(f"TYPE: pyLDA -- MD5: {result_dict['text_md5']}")

                try:
                    vis_future_pcoa = client.submit(
                        create_vis_pcoa,
                        result_dict['lda_model'],
                        result_dict['corpus'],
                        result_dict['topics'],  # f'number_of_topics-{topics}'
                        phase_name,
                        result_dict['text_md5'],  # filename
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
