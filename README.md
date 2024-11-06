# Unified Topic Modeling and Analysis (UTMA)

### Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Example Preprocessing: *Moby Dick*](#example-preprocessing-moby-dick)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Optimization](#optimization)

### Overview
**Unified Topic Modeling and Analysis (UTMA)** is a flexible and scalable framework designed for comprehensive topic modeling and analysis. Beyond Latent Dirichlet Allocation (LDA), UTMA integrates adaptive resource management, dynamic topic modeling, and advanced analytical capabilities to support large-scale, distributed computations and diachronic analysis. This framework is ideal for tracking topic evolution over time, making it suitable for a variety of document types and research needs.

### Key Features
- **Adaptive Resource Management**: UTMA leverages Dask for process-based distributed parallelization, harnessing multiple cores and avoiding the limitations of the GIL while dynamically adjusting resources to efficiently handle large datasets and high computational loads.
- **Comprehensive Topic Modeling**: Supports customizable LDA model configurations and hyperparameter tuning, enabling detailed exploration of topics in diverse document collections.
- **Diachronic Analysis(_Pending_)**: Facilitates tracking and analyzing topic shifts over time, particularly useful for examining historical changes or comparing topics across decades.
- **Detailed Metadata Tracking**: Records extensive metadata for each batch, including dynamic core counts, model parameters, and evaluation scores, ensuring complete reproducibility and transparency.
- **Support for Multiple Document Types**: Designed to handle diverse document sources, from medical journals and newspapers to novels, UTMA can analyze any textual dataset requiring topic-based insights.
- **Integrated Data Persistence and Storage**: Metadata and model outputs are stored in a PostgreSQL database, supporting complex queries and retrieval for downstream analysis.

### Example Preprocessing: *Moby Dick*
As an example of how data must be preprocessed for UTMA, consider analyzing *Moby Dick* by Herman Melville. Each paragraph of the novel can be treated as an individual document within the corpus, requiring tokenization and conversion to a suitable format, such as a bag-of-words model. By segmenting the text in this way, UTMA can uncover recurring themes and track the evolution of topics like "whaling," "obsession," and "fate" across the paragraphs. Preprocessing each chapter separately prepares the text for topic modeling and diachronic analysis(_in development_).

   ```json
   [
     ["precisely", "little", "particular", "interest", "think", "little", "watery", "drive", "spleen", "regulate", "circulation", "grow", "whenever", "drizzly", "involuntarily", "pause", "coffin", "warehouse", "bring", "funeral", "especially", "require", "strong", "principle", "prevent", "deliberately", "step", "street", "methodically", "knock", "people", "account", "substitute", "pistol", "philosophical", "flourish", "throw", "quietly", "surprising", "almost", "degree", "cherish", "nearly", "feeling"],
     ["insular", "belt", "wharf", "indian", "commerce", "surround", "street", "waterward", "extreme", "downtown", "battery", "wash", "cool", "breeze", "previous", "crowd", "gazer"],
     ["dreamy", "afternoon", "thence", "northward", "see?—poste", "silent", "sentinel", "thousand", "thousand", "mortal", "reverie", "lean", "spile", "seat", "look", "bulwark", "rigging", "strive", "well", "seaward", "landsman", "plaster", "counter", "nail", "bench", "clinch", "field"]
   ]
   ```
<sub>*Source: [Project Gutenberg](https://www.gutenberg.org/files/2701/2701-h/2701-h.htm)  
Release Date: June, 2001 [eBook #2701]  
Most recently updated: August 18, 2021*</sub>



### Project Structure
The project is composed of modular scripts, each dedicated to a specific aspect of the pipeline, including:

### Prerequisites
- **Python** 3.12.0
- **Anaconda** (recommended for dependency management)
- **PostgreSQL** (required for data storage and integration)

- **Data Preprocessing**: Handles initial data preparation, including tokenization and formatting, for ingestion by the modeling pipeline.
- **Dynamic Topic Model Training** (`topic_model_trainer.py`): Manages the LDA model training, evaluation, and metadata generation, with adaptive scaling to optimize resource usage.
- **Visualization and Analysis**: Generates and saves visualizations (e.g., topic coherence plots) for exploring model outputs interactively.
- **Diachronic Analysis (_Planned_)**: A dedicated module for analyzing and visualizing how topics evolve over time.
- **Database Integration**: Stores all metadata and modeling outputs in a PostgreSQL database for easy access and persistence.

--- 

### Installation
To install UTMA, clone the repository and set up the required Python environment:

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Set Up the Environment**
Using Anaconda is recommended for managing dependencies:
   ```bash
   conda create -n utma_env python=3.12.0
   conda activate utma_env
   ```

3. **Install Dependencies**
   
   - Open your terminal and run the following command to install the required packages listed in `requirements.txt`:

     ```bash
     $ pip install -r requirements.txt
     ```

   - For Dask's distributed functionality, ensure `dask[distributed]` is installed:

     ```bash
     $ pip install dask[distributed]=="2024.8.2"
     ```

4. **Set Up PostgreSQL Database**
Ensure PostgreSQL is installed and running. Create a new database to store UTMA data, and update the connection settings in the project configuration files.

5. **Run the Application**

### Usage
After setup, run the main script to start the UTMA framework. Here’s an example command:

   ```bash
   python utma.py \
       --username "postgres" \
       --password "admin" \
       --database "UTMA" \
       --corpus_label "moby_dick" \
       --data_source "/path/to/your/data/tokenized-documents/moby_dick.json" \
       --start_topics 20 \
       --end_topics 60 \
       --step_size 5 \
       --num_workers 10 \
       --max_workers 14 \
       --num_threads 8 \
       --max_memory 5 \
       --mem_threshold 4 \
       --max_cpu 110 \
       --futures_batches 75 \
       --base_batch_size 100 \
       --max_batch_size 120 \
       --log_dir "/path/to/your/log/" \
       --root_dir "/path/to/your/root/" \
       2>"/path/to/your/log/terminal_output.txt"
   ```
This command manages the distribution of resources, saves model outputs, and logs metadata directly to the database.

### **Optimization** 

   Optimizing Future, Base, and Maximum Batch Size for utma.py

   Configuring `futures_batches`, `base_batch_size`, and `max_batch_size` is critical to balancing resource utilization and achieving efficient processing times, especially on high-performance systems.

   ### **Guidlines for Setting Key Batch Size Parameter**
   -  `--futures_batches`: Defines the maximum number of future tasks that can be scheduled in Dask.
   -  `--base_batch_size`: Sets the base number of documents per batch for each phase (training, validation, and test).
   -  `--max_batch_size`: Sets the maximum number of documents in a batch, providing flexibility for adaptive batching.


   ### 1. **Importance of `--futures_batches`**
   The  `--futures_batches` parameter directly affects processing performance by controlling task concurrency:

   -  **Performance Tuning:** Adjusting `--futures_batches` manages resource allocation. Higher values increase parallelism but may exhaust memory, while lower values conserve resources.

   -  **Dynamic Adaptation:** Optimize based on document structure and system resources, beginning with `--futures_batches=3` and increasing gradually to find balance.

   ### 2.  **Setting Minimum and Maximum Batch Sizes**
   -  **Consider System Resources:** Set batch sizes according to CPU and memory capacity. High-memory, multi-core setups (e.g., 128 GB RAM) support larger batch sizes.

   **Balance Task Complexity and Resource Use:**

   -  **Complex Tasks:** Use larger batches to minimize task overhead and reduce inter-process communication.
   Light Tasks: Smaller batches maximize CPU utilization in low-memory environments.

   -  **Light Tasks:** Smaller batches maximize CPU utilization in low-memory environments.

   ### 3. **Starting Points and Adjustments**
   -  **Initial Defaults:** Try base_batch_size=100-200 and max_batch_size=120-220.
   -  **Monitor and Adjust:** Use Dask’s dashboard to refine settings based on observed performance and resource utilization.

   ### **Example Configurations:**
   -  **High-Memory Systems:** `--futures_batches=5`(_value determined by size of preprocessed corpus_), `--base_batch_size=200`, `--max_batch_size=220`.
   -  **Low-Memory Systems:** Use `--base_batch_size` with valueus between 50 and 100 to prevent memory overload.

   Testing and Adjustment:
   -  Start with sample data to test batch sizes, adjusting iteratively to balance performance and resource limits.

   **Monitoring Performance**
   After configuring batch sizes, use the Dask dashboard to observe task distribution, resource utilization, and memory usage per worker. Adjust batch sizes further if tasks are not distributed evenly or if memory usage approaches system limits.

<sub>_Last updated: 2024-11-02_</sub>

