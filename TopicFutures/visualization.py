# written with pair programming
import os 
import numpy as np
import pyLDAvis
import pyLDAvis.gensim  # Library for interactive topic model visualization
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import logging
# Set max_open_warning to 0 to suppress the warning
plt.rcParams['figure.max_open_warning'] = 0 # suppress memory warning msgs re too many plots being open simultaneously

def create_vis(ldaModel, corpus, dictionary, filename, CORES, PYLDA_DIR, PCOA_DIR):
    create_pylda = None
    create_pcoa = None
    PCoAfilename = filename
    #print("We are inside Create Vis.")
    

    IMAGEFILE = os.path.join(PYLDA_DIR,f"{filename}.html")
    PCoAIMAGEFILE = os.path.join(PCOA_DIR, PCoAfilename)

    # Disable notebook mode since we're saving to HTML.
    #pyLDAvis.disable_notebook()
    
    # Prepare the visualization data.
    # Note: sort_topics=False will prevent reordering topics after training.
    try:
        vis = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary,  n_jobs=int(CORES*(2/3)), sort_topics=False)

        pyLDAvis.save_html(vis, IMAGEFILE)
        create_pylda = True

    except Exception as e:
        logging.error(f"The pyLDAvis HTML could not be saved: {e}")
        create_pylda = False


    # try Jensen-Shannon Divergence & Principal Coordinate Analysis (aka Classical Multidimensional Scaling)
    topic_labels = [] # list to store topic labels

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


    return filename, create_pylda, create_pcoa