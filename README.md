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
- **Adaptive Resource Management**: UTMA leverages Dask for distributed parallel processing, dynamically adjusting resources to efficiently handle large datasets and high computational loads.
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
       --futures_batches 100 \
       --base_batch_size 100 \
       --max_batch_size 120 \
       --log_dir "/path/to/your/log/" \
       --root_dir "/path/to/your/root/" \
       2>"/path/to/your/log/terminal_output.txt"
   ```
This command manages the distribution of resources, saves model outputs, and logs metadata directly to the database.

### **Optimization** 

   Optimizing Batch Size for utma.py

   Configuring batch sizes is critical to balancing resource utilization and achieving efficient processing times, especially on high-performance systems.

   ## **Key Batch Size Parameters**
   -  futures_batches: Defines the maximum number of future tasks that can be scheduled in Dask.
   -  base_batch_size: Sets the base number of documents per batch for each phase (training, validation, and test).
   -  max_batch_size: Sets the maximum number of documents in a batch, providing flexibility for adaptive batching.

   ## **Guidelines for Setting Batch Sizes**

   1. **Consider System Resources:**
         -  Determine batch sizes based on the available CPU cores, memory, and disk I/O capacity. Larger batch sizes are typically feasible on systems with high memory (e.g., 128 GB RAM or more) and multiple CPU cores, as they allow each Dask worker to handle more documents without exceeding memory limits.

   2. **Balance Task Complexity and Resource Utilization**:

      -  **Complex or Intensive Tasks** (e.g., high-dimensional topic modeling): Larger batch sizes reduce the number of tasks submitted to Dask, decreasing the overhead of task management and inter-process communication.
      -  **Simple or Lightweight Tasks:** Smaller batch sizes allow more tasks to be processed in parallel, maximizing CPU utilization for low-memory operations. Start with Reasonable Defaults and Refine:

   3. **Start with Reasonable Defaults and Refine:**
      -  Initial Recommendation: For typical workloads, set base_batch_size to around 100-200 and max_batch_size slightly above (e.g., 120-220).
      -  Monitor performance and refine the batch size based on observed processing times and Dask’s resource utilization (accessible via the Dask dashboard).

   4. **Example Configurations:**
      -  High-Memory, Multi-Core Systems: Set futures_batches=200, base_batch_size=200, and max_batch_size=220.
      -  Low-Memory or Limited-Core Systems: Reduce base_batch_size to 50-100, depending on available memory, to prevent memory overflow on each worker.

   5. **Testing and Adjustment:**
      -  Run a test with smaller data samples to verify the chosen batch size. Adjust futures_batches, base_batch_size, and max_batch_size iteratively, as needed, to achieve optimal processing time.

   **Monitoring Performance**
   After configuring batch sizes, use the Dask dashboard to observe task distribution, resource utilization, and memory usage per worker. Adjust futures_batches and batch sizes further if tasks are not distributed evenly or if memory usage approaches system limits.

   Additional guidance on optimizing UTMA performance can be found in the project documentation. Use the Dask dashboard for real-time monitoring and to identify any bottlenecks in processing.

<sub>_Last updated: 2024-11-02_</sub>

