# Unified Topic Modeling and Analysis (UTMA)

### Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Example Preprocessing: CDC's Morbidity and Mortality Weekly Report Journals](#cdcs-mmwr-2015---2019)
- [Optimization](#optimization)

### Overview
**Unified Topic Modeling and Analysis (UTMA)** is a versatile and scalable framework for advanced topic modeling and analysis. Going beyond simple text mining, UTMA incorporates adaptive resource management, dynamic topic modeling, and sophisticated analytical capabilities for large-scale, distributed computations and diachronic (time-based) analysis. UTMA is ideal for tracking topic evolution over time and is well-suited to various document types and research applications.

### Key Features
- **Adaptive Resource Management**: UTMA leverages [Dask](https://www.dask.org/) for process-based distributed parallelization, harnessing multiple cores and avoiding the limitations of the GIL while dynamically adjusting resources to efficiently handle large datasets and high computational loads.
-  **Dask Delayed Integration**: UTMA makes extensive use of Dask’s `@delayed` decorator to transform functions and tasks into Dask-delayed objects, enabling the lazy evaluation of computational tasks. This design optimizes resource use by building up a task graph before executing it in parallel, allowing UTMA to handle large, complex workflows more efficiently.
- **Concurrency, Parallelization, and Multithreading**: The system utilizes a hybrid model of concurrency through Dask’s distributed tasks, allowing concurrent execution, with multiprocessing for resource-heavy operations and multithreading within Dask workers for efficient I/O-bound task management. This design enhances both performance and scalability across different hardware configurations.
-  [**CUDA Acceleration**](https://developer.nvidia.com/about-cuda): UTMA leverages **CUDA** through **CuPy** to accelerate computationally intensive coherence metric calculations. CUDA is specifically applied to mean, median, and standard deviation calculations of coherence scores across large datasets, significantly reducing processing time. By offloading these calculations to the GPU, UTMA can handle high-volume computations more efficiently, maximizing the potential of systems equipped with NVIDIA GPUs.
- **Comprehensive Machine Learning Pipeline**: The framework includes a robust machine learning pipeline that handles data preprocessing, model training, evaluation, and hyperparameter tuning, designed to optimize model performance for diverse text corpora.
- **Diachronic Analysis (_Pending_)**: Facilitates tracking and analyzing topic shifts over time, particularly useful for examining historical changes or comparing topics across decades.
- **Detailed Metadata Tracking**: Records extensive metadata for each batch, including dynamic core counts, model parameters, and evaluation scores, ensuring complete reproducibility and transparency.
- **Support for Multiple Document Types**: Designed to handle diverse document sources, from [free, open access to academic outputs and resources](https://v2.sherpa.ac.uk/opendoar/) and [newspapers](https://en.wikipedia.org/wiki/List_of_free_daily_newspapers) to [novels](https://www.gutenberg.org/), UTMA can analyze any textual dataset requiring topic-based insights.
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
- [**CUDA**](https://developer.nvidia.com/cuda-zone): Required for GPU acceleration. CUDA enables the use of CuPy to accelerate certain computations, particularly useful for handling large datasets and complex calculations in the UTMA framework.
   - **CuPy**: Used in coherence metric calculations to enable CUDA-based acceleration, leveraging GPU resources for high-precision and high-performance computations. CuPy enhances the speed of computations, especially for coherence and convergence metrics, by offloading tasks to the GPU.
- **Data Preprocessing** (`DocumentParser Notebook`): Handles initial data preparation, including tokenization and formatting, for ingestion by the modeling pipeline. See [example](#cdcs-mmwr-2015---2019)
- **Dynamic Topic Model Training** (`topic_model_trainer.py`): Manages the LDA model training, evaluation, and metadata generation, with adaptive scaling to optimize resource usage.
- **Visualization and Analysis** (`visualization.py`): Generates and saves visualizations (e.g., topic coherence plots) for exploring model outputs interactively.
- **Diachronic Analysis (_Pending_)**: A dedicated module for analyzing and visualizing how topics evolve over time.
- **Database Integration** (`write_to_postgres.py`): Stores all metadata and modeling outputs in a PostgreSQL database for easy access and persistence.
 **Dask Distributed Configuration**: UTMA includes a custom `distributed.yaml` file in the `config/` directory with recommended settings for Dask performance and resource management tailored to UTMA’s processing requirements.

--- 

### Installation
To install UTMA, clone the repository and set up the required Python environment. Follow these steps for a smooth setup:

#### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/alan-hamm/Unified-Topic-Modeling-and-Analysis.git
   cd your-directory-name
   ```

2. **Set Up the Environment** 
      (Using Anaconda is recommended for managing dependencies)
      ```bash
      conda env create -f config/environment.yaml
      conda activate utma
      ```

3. **Set Up PostgreSQL Database** Ensure PostgreSQL is installed and running. Create a new database to store UTMA data, and update the connection settings in the project configuration files to point to your database.

#### Data Processing 

If you have documents that require preprocessing, use the following guidelines to prepare your data before proceeding with UTMA analysis:

1. **If Preprocessing is Required**: Use the `DocumentParser` notebook to prepare your documents according to UTMA's standard format, as outlined in the [CDC's MMWR example](#cdcs-mmwr-2015---2019). **IMPORTANT** Make sure the Python environment has been set up(_i.e._, run `environment.yaml`) before using the notebook.
   
2. **If Preprocessing is Not Required**: If your documents are already in the required format and you've set up the Python environment with `environment.yaml`, you can skip the preprocessing step and proceed directly to running UTMA.

3. **Combining Installation and Preprocessing**: **Combining Installation and Preprocessing** If both installation and preprocessing are needed, first complete the [installation steps](#installation-steps) above, then preprocess the data using the `DocumentParser` notebook.


### Distributed Configuration

   **_By default, the settings in `distributed.yaml` are optimized for high-performance processing with Dask on systems with significant CPU and memory resources. Adjust as needed to suit your environment._**

This project includes a custom `distributed.yaml` file for configuring Dask. The `distributed.yaml` file is located in the [`config/`](https://github.com/alan-hamm/Unified-Topic-Modeling-and-Analysis/tree/main/config) directory and contains recommended settings for Dask performance and resource management tailored for UTMA's processing requirements.

To ensure your Dask environment is correctly configured, follow these steps:

   1. **Review the `distributed.yaml` File**  
      Examine the `config/distributed.yaml` file to understand its settings, especially if you need to adjust resource limits based on your system’s specifications.

   2. **Customize if Necessary**  
      Depending on your hardware and workload, you may want to customize certain values (e.g., memory limits, CPU thresholds) in the `distributed.yaml` file.

   3. **Refer to Setup Instructions**  
      For more detailed instructions on configuring the Dask dashboard and securing it for local access, see the `Dask_Dashboard_Setup_Instructions.txt` file in the `config/` directory.

## Preprocessing with DocumentParser Notebook
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

## Steps to Run the UTMA Script
After setup, run the main script to start the UTMA framework. Here’s an example command:

   ```bash
   python utma.py \
       --username "postgres" \
       --password "admin" \
       --database "UTMA" \
       --corpus_label "mmwr" \
       --data_source "/path/to/your/data/tokenized-documents/mmwr_year_2015.json" \
       --start_topics 20 \
       --end_topics 60 \
       --step_size 5 \
       --num_workers 10 \
       --max_workers 12 \
       --num_threads 2 \
       --max_memory 15 \
       --mem_threshold 14 \
       --max_cpu 110 \
       --futures_batches 275 \
       --base_batch_size 275 \
       --max_batch_size 300 \
       --log_dir "/path/to/your/log/" \
       2>"/path/to/your/log/terminal_output.txt"
   ```
This command manages the distribution of resources, saves model outputs, and logs metadata directly to the database.

#### CDC's MMWR [2015 - 2019](https://github.com/alan-hamm/Unified-Topic-Modeling-and-Analysis/tree/main/examples)
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


## **Optimization** 

   Configuring `futures_batches`, `base_batch_size`, and `max_batch_size` is critical to balancing resource utilization and achieving efficient processing times, especially on high-performance systems. The script `batch_estimation.py` is provided for adaptive batch size estimation based on document complexity, memory, and CPU limits. This script is recommended for anyone running UTMA on datasets with varying document sizes or on systems with constrained resources.


   ### **Guidlines for Setting Key Batch Size Parameter**
   
   1. **Understanding Batch Size Impact**

      -  **Base Batch Size**: Setting an appropriate base batch size is crucial. A batch size too small will increase scheduling overhead, while a batch size too large can exhaust memory resources, leading to performance degradation. For large documents or complex tasks, use larger batch sizes to optimize resource use and reduce scheduling overhead. For smaller tasks, use smaller batch sizes to increase task concurrency and CPU utilization.
   
      -  **Max Batch Size**: Defines the upper limit for document processing per batch. Adaptive batching helps to manage tasks dynamically based on resource availability. Setting this value appropriately helps UTMA adapt to different document types without exhausting memory.

   2. **Batch Calculation and System Resource Balance**

      Batch size should be calculated to balance memory usage and task efficiency. The `batch_estimation.py` script automates this process by analyzing document complexity, system memory, and CPU limits to suggest an optimal batch size for both standard and large documents. This script is highly recommended for fine-tuning `futures_batches`, `base_batch_size`, and `max_batch_size` based on empirical testing.

         **Example Usage of `batch_estimation.py`**:
         ```python
         from batch_estimation import estimate_futures_batches
         optimal_batch_size = estimate_futures_batches(document="path/to/document.json")
         ```

   3. **Optimal Futures Batches**

      The `futures_batches` parameter is essential for parallel task processing. Setting this to a higher value allows more concurrent tasks but may increase memory usage. For initial configurations, a conservative value (e.g., 3–10) is recommended, then adjust based on system performance. Higher values can improve speed but risk memory overflow.

   4. **Benefits of Adaptive Batch Sizes**

      Adaptive batch sizes calculated by 'batch_estimation.py' allow the UTMA framework to better handle document variability and optimize resource usage. This approach reduces memory-related issues, as batch sizes are optimized for current system capacity and workload, ensuring smooth execution without overwhelming resources.

   5. **Monitoring and Iterative Adjustment**

      Use the [Dask dashboard](https://docs.dask.org/en/latest/dashboard.html) to observe task distribution, memory usage, and performance metrics. Monitor the impact of changes in batch size on system utilization, and adjust batch sizes if memory or CPU usage approaches system thresholds.

   6. **RAM Allocation and Management**

      UTMA is memory-intensive, especially when handling large datasets or high batch sizes. Setting a high memory_limit in the Dask LocalCluster configuration is recommended if system RAM allows. For optimal memory usage:

      -  Adjust memory_limit based on available system RAM and the expected load. As a rule of thumb, ensure that memory_limit per worker is balanced with the total number of workers to avoid exceeding system memory.
      -  Monitor RAM usage in the Dask dashboard. If you notice frequent memory spills or high memory consumption, consider reducing base_batch_size or max_batch_size.
      -  Use Adaptive Scaling to optimize worker utilization without overloading RAM. Configure min_workers and max_workers according to your system's capabilities. For instance, setting min_workers=10 and max_workers=14 can dynamically scale tasks without overwhelming available memory.

   7. **Core and Thread Configuration**

      Adjust 'num_workers', 'max_workers', and 'num_threads' based on the core count of your system. Higher core counts improve model training speed, while thread configuration impacts memory efficiency. Example configurations:

      #### **Example Configurations:**
      -  **High-Core Count Systems:** `--num_workers=10`, `--max_workers 14`, `--num_threads=2`.
      -  **Low-Core Count Systems:** `--num_workers=4`, `--max_workers 6`, `--num_threads=1`.

   Profiling parallel code can be challenging, but Dask's distributed scheduler offers an [interactived dashboard](https://docs.dask.org/en/latest/dashboard.html) for diagnostics that simplifies real-time computation monitoring. Built with Bokeh, the dashboard is available upon starting the scheduler and provides a user-specified link(_e.g._ http://localhost:8787/status) to track task progress and resource usage according to your Dask configuration.

   See [How to diagnose performance](https://distributed.dask.org/en/latest/diagnosing-performance.html)\,  [Diagnostics(local)](https://docs.dask.org/en/stable/diagnostics-local.html)\,  and [Diagnostics(distributed)](https://docs.dask.org/en/stable/diagnostics-distributed.html)

   **Monitoring Performance**
   After configuring batch sizes, use the Dask dashboard to observe task distribution, resource utilization, and memory usage per worker. Adjust batch sizes further if tasks are not distributed evenly or if memory usage approaches system limits.

<sub>_Last updated: 2024-11-11_</sub>

