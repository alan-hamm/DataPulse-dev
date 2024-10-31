# developed traditionally with addition of AI assistance
from .utils import garbage_collection
import os 
import numpy as np
import pyLDAvis
import pyLDAvis.gensim  # Library for interactive topic model visualization
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('Agg')

import logging
# Set max_open_warning to 0 to suppress the warning
plt.rcParams['figure.max_open_warning'] = 0 # suppress memory warning msgs re too many plots being open simultaneously


def create_vis_pylda(ldaModel, corpus, dictionary, topics, filename, CORES, vis_root, PYLDA_DIR):
    create_pylda = None
    #print("We are inside Create Vis.")
    
    topics_dir = os.path.join(PYLDA_DIR, f"number_of_topics-{topics}")
    #PYLDA_DIR = os.path.join(topics_dir,vis_root)
    PYLDA_DIR = os.path.join(topics_dir,vis_root)
    #PYLDA_DIR = os.path.join(PYLDA_DIR,vis_root)
    os.makedirs(PYLDA_DIR, exist_ok=True)
    IMAGEFILE = os.path.join(PYLDA_DIR,f"{filename}.html")

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


def create_vis_pcoa(ldaModel, corpus, topics, filename, vis_root, PCOA_DIR):
    create_pcoa = None
    PCoAfilename = filename

    topics_dir = os.path.join(PCOA_DIR, f"number_of_topics-{topics}")
    #PCOA_DIR = os.path.join(topics_dir, vis_root)
    PCOA_DIR = os.path.join(topics_dir, vis_root)
    #PCOA_DIR = os.path.join(PCOA_DIR, vis_root)
    os.makedirs(PCOA_DIR, exist_ok=True)
    PCoAIMAGEFILE = os.path.join(PCOA_DIR, PCoAfilename)

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
