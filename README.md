# Unified Topic Modeling and Analysis (UTMA)

### Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Preprocessing: CDC's Morbidity and Mortality Weekly Report Journals](#cdcs-mmwr-2015---2019)
- [Optimization](#optimization)

### Overview
**Unified Topic Modeling and Analysis (UTMA)** is a versatile and scalable framework for advanced topic modeling and analysis. Going beyond simple text mining, UTMA incorporates adaptive resource management, dynamic topic modeling, and sophisticated analytical capabilities for large-scale, distributed computations and diachronic (time-based) analysis. UTMA is ideal for tracking topic evolution over time and is well-suited to various document types and research applications.

### Key Features
- **Adaptive Resource Management**: UTMA leverages [Dask](https://www.dask.org/) for process-based distributed parallelization, harnessing multiple cores and avoiding the limitations of the GIL while dynamically adjusting resources to efficiently handle large datasets and high computational loads.
- **Concurrency, Parallelization, and Multithreading**: The system utilizes a hybrid model of concurrency through Dask’s distributed tasks, allowing concurrent execution, with multiprocessing for resource-heavy operations and multithreading within Dask workers for efficient I/O-bound task management. This design enhances both performance and scalability across different hardware configurations.
- **Comprehensive Machine Learning Pipeline**: The framework includes a robust machine learning pipeline that handles data preprocessing, model training, evaluation, and hyperparameter tuning, designed to optimize model performance for diverse text corpora.
- **Diachronic Analysis (_Pending_)**: Facilitates tracking and analyzing topic shifts over time, particularly useful for examining historical changes or comparing topics across decades.
- **Detailed Metadata Tracking**: Records extensive metadata for each batch, including dynamic core counts, model parameters, and evaluation scores, ensuring complete reproducibility and transparency.
- **Support for Multiple Document Types**: Designed to handle diverse document sources, from [thousands of free, open access to academic outputs and resources](https://v2.sherpa.ac.uk/opendoar/) and [newspapers](https://en.wikipedia.org/wiki/List_of_free_daily_newspapers) to [novels](https://www.gutenberg.org/), UTMA can analyze any textual dataset requiring topic-based insights.
- **Integrated Data Persistence and Storage**: Metadata and model outputs are stored in a PostgreSQL database, supporting complex queries and retrieval for downstream analysis.

### Project Structure
The project is composed of modular scripts, each dedicated to a specific aspect of the pipeline, including:

- **Dynamic Topic Model Training** (`topic_model_trainer.py`): Manages the LDA model training, evaluation, and metadata generation, with adaptive scaling to optimize resource usage using both multiprocessing and multithreading to enhance performance.
- **Visualization and Analysis**(`visualization.py`): Generates and saves visualizations (e.g., topic coherence plots) for exploring model outputs interactively, utilizing concurrent processing capabilities.
- **Database Integration**(`write_to_postgres.py`): Stores all metadata and modeling outputs in a PostgreSQL database for easy access and persistence.
- **Diachronic Analysis (_Planned_)**: A dedicated module for analyzing and visualizing how topics evolve over time.

---

For further configuration tips and performance monitoring, Dask provides a [dashboard](https://docs.dask.org/en/stable/) for tracking concurrent task execution, resource utilization, and multithreading activity in real-time.

### Prerequisites
- **Python** 3.12.0
- **Anaconda** (recommended for dependency management)
- **PostgreSQL** (required for data storage and integration)
- **Data Preprocessing(_script coming soon_)**: Handle initial data preparation, including tokenization and formatting, for ingestion by the modeling pipeline. See [example](#cdcs-mmwr-2015---2019)
- **Dynamic Topic Model Training** (`topic_model_trainer.py`): Manages the LDA model training, evaluation, and metadata generation, with adaptive scaling to optimize resource usage.
- **Visualization and Analysis**(`visualization.py`): Generates and saves visualizations (e.g., topic coherence plots) for exploring model outputs interactively.
- **Diachronic Analysis (_Planned_)**: A dedicated module for analyzing and visualizing how topics evolve over time.
- **Database Integration**(`write_to_postgres.py`): Stores all metadata and modeling outputs in a PostgreSQL database for easy access and persistence.

--- 

### Installation
To install UTMA, clone the repository and set up the required Python environment. Follow these instructions based on your data's format:

1. **If Preprocessing is Required**: Use the `DocumentParser.ipynb` notebook to prepare your documents according to UTMA's standard format, as outlined in the [CDC's MMWR example](#cdcs-mmwr-2015---2019). Once preprocessing is completed, proceed with the installation steps below.
   
2. **If Preprocessing is Not Required**: Skip to the installation steps directly.

3. **If Both Steps are Needed**: First, preprocess the data, then follow the installation instructions.

#### Installation Steps

1. **Clone the Repository**
 ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
  ```

2. **Set Up the Environment** Using Anaconda is recommended for managing dependencies:
   ```bash
      conda create -n utma_env python=3.12.0
      conda activate utma_env
   ```
3. **Install Dependencies** Run the following commands to install the required packages:
   ```bash
   pip install -r requirements.txt
   pip install dask[distributed]=="2024.8.2"
   ```
4. **Set Up PostgreSQL Database** Ensure PostgreSQL is installed and running. Create a new database to store UTMA data, and update the connection settings in the project configuration files.

## Preprocessing with DocumentParser.ipynb
Use DocumentParser.ipynb if your documents are not in UTMA's expected format. The notebook:
-  Parses JSON and HTML files, ensuring clean and structured text.
-  Performs tokenization, lemmatization, and bigram extraction.
-  Produces JSON and JSONL formatted outputs that align with UTMA’s standards.

Documents must be in the following format to bypass preprocessing:
```json
   [
      ["example", "tokenized", "sentence", "one"],
      ["another", "example", "sentence", "two"],
      ["and", "so", "on"]
   ]
```

## Steps to Run the Notebook
1. Open DocumentParser.ipynb, set corpus_path to the folder containing your documents, and adjust file paths for logging and outputs.
2. Execute each cell to process and save the documents in the required format.
3. Move the processed files to the designated input directory.

## Steps to Run the UTMA.py Script
After setup, run the main script to start the UTMA framework. Here’s an example command:

   ```bash
   python utma.py \
       --username "postgres" \
       --password "admin" \
       --database "UTMA" \
       --corpus_label "mmwr" \
       --data_source "/path/to/your/data/tokenized-documents/mmwr.json" \
       --start_topics 20 \
       --end_topics 60 \
       --step_size 5 \
       --num_workers 10 \
       --max_workers 14 \
       --num_threads 2 \
       --max_memory 5 \
       --mem_threshold 4 \
       --max_cpu 110 \
       --futures_batches 75 \
       --base_batch_size 100 \
       --max_batch_size 120 \
       --log_dir "/path/to/your/log/" \
       2>"/path/to/your/log/terminal_output.txt"
   ```
This command manages the distribution of resources, saves model outputs, and logs metadata directly to the database.

#### CDC's MMWR 2015 - 2019
A real-world application of UTMA’s data preprocessing capabilities can be seen in analyzing the [MMWR Journals](https://www.cdc.gov/mmwr/), extracted from the [CDC text corpora for learners](https://github.com/cmheilig/harvest-cdc-journals/). Each report in these journals is treated as a standalone document and requires specific preprocessing steps to align with UTMA's standards, including tokenization and formatting as a bag-of-words model.

By organizing and structuring the text data in this format, UTMA can identify recurring themes and track the evolution of key public health topics, such as "infection control," "vaccine efficacy," and "disease prevention." This structured approach allows UTMA to perform diachronic analyses of topic shifts over time, revealing insights into public health trends and topic persistence. Preprocessing each document in this way prepares it for the advanced topic modeling and analysis that UTMA provides.

**Excerpt:**
   ```json
   [
      ["prevalence", "abstinence", "months", "enrollment", "confidence", "interval", "certificate", "confidence_interval"], 
      ["groups", "contributed", "modification", "national", "infection", "prevention", "control", "strategy", "incorporate", "community", "awareness"], 
      ["effectiveness", "seasonal", "influenza", "vaccine", "depends", "vaccine", "viruses", "circulating", "influenza", "viruses"], 
      ["investigators", "determined", "likely", "factors", "transmission", "included", "bottles", "shared", "football", "players"], 
      ["collaboration", "agencies", "overseas", "vaccination", "intended", "reduce", "disease", "outbreaks", "ensuring", "refugees", "arrive", "protected"]
   ]
   ```
   This format aligns with the project’s requirements, enabling UTMA to analyze the thematic structure and evolution of health topics in CDC reports.

   The project supports preprocessing for a range of CDC’s journal content, including _Emerging Infectious Diseases_([EID](https://wwwnc.cdc.gov/eid)) and _Preventing Chronic Disease_([PCD](https://www.cdc.gov/pcd)). Available resources include CDC documents, spanning 42 years: [HTML Mirrors of MMWR, EID, and PCD](https://data.cdc.gov/National-Center-for-State-Tribal-Local-and-Territo/CDC-Text-Corpora-for-Learners-HTML-Mirrors-of-MMWR/ut5n-bmc3/about_data) and associated [Corpus Metadata](https://data.cdc.gov/National-Center-for-State-Tribal-Local-and-Territo/CDC-Text-Corpora-for-Learners-MMWR-EID-and-PCD-Art/7rih-tqi5/about_data).

### **Optimization** 

   Configuring `futures_batches`, `base_batch_size`, and `max_batch_size` is critical to balancing resource utilization and achieving efficient processing times, especially on high-performance systems.

   ### **Guidlines for Setting Key Batch Size Parameter**
   -  `--futures_batches`: Defines the maximum number of future tasks that can be scheduled in Dask.
   -  `--base_batch_size`: Sets the base number of documents per batch for each phase (training, validation, and test).
   -  `--max_batch_size`: Sets the maximum number of documents in a batch, providing flexibility for adaptive batching.


   ### 1. **Importance of the futures_batches Parameter**
   The --futures_batches parameter plays a critical role in processing performance by controlling how many tasks run concurrently. Adjusting this parameter allows you to balance resource use and processing efficiency based on your system’s capacity.

   -  Performance Tuning: Higher values for --futures_batches increase parallel task processing, which can speed up execution but may also push memory limits. Lower values, on the other hand, reduce memory usage but may slow down execution by limiting parallelism.

   -  Dynamic Adaptation: Start with a conservative value for --futures_batches, such as 3–10, depending on your system’s memory and processing power. Gradually increase this value to find the optimal balance between parallelism and resource availability.

   ### 2. Setting Minimum and Maximum Batch Sizes
   Choosing the right batch sizes is essential for balancing load and maximizing efficiency based on task complexity.

   -  Complex Tasks: For resource-intensive tasks (e.g., large-scale modeling), set a larger batch size to reduce the overhead associated with task scheduling and communication between processes.

   -  Lightweight Tasks: For simpler, less memory-intensive tasks, use smaller batches. This allows more tasks to be processed concurrently, maximizing CPU utilization, especially in environments with limited memory.

      **Example Configuration:** `--futures_batches=75`, `--base_batch_size=200`, `--max_batch_size=220`

   ### 3. **Core Count and Thread Configuration**
   For UTMA, while memory is essential for handling large datasets, **the number of cores and threads available significantly impacts performance**.
   -  **Higher Core Counts:** Increasing the number of cores for training and inference improves performance, especially on multi-threaded systems.
   -  **Thread Utilization:** Configuring an optimal number of threads per worker improves processing time while managing memory efficiency.

      #### **Example Configurations:**
      -  **High-Core Count Systems:** `--num_workers=10`, `--max_workers 14`, `--num_threads=2`.
      -  **Low-Core Count Systems:** `--num_workers=4`, `--max_workers 6`, `--num_threads=1`.

      Profiling parallel code can be challenging, but Dask's distributed scheduler offers an interactive [dashboard](https://docs.dask.org/en/latest/dashboard.html) for diagnostics that simplifies real-time computation monitoring. Built with Bokeh, the dashboard is available upon starting the scheduler and provides a user-specified link(_e.g._ http://localhost:8787/status) to track task progress and resource usage according to your Dask configuration.

      See **[Dask documentation](https://docs.dask.org/en/stable/)**

   Testing and Adjustment:
   -  Start with sample data to test batch sizes, adjusting iteratively to balance performance and resource limits.

   **Monitoring Performance**
   After configuring batch sizes, use the Dask dashboard to observe task distribution, resource utilization, and memory usage per worker. Adjust batch sizes further if tasks are not distributed evenly or if memory usage approaches system limits.

<sub>_Last updated: 2024-11-02_</sub>

