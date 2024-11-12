
# SpectraSync Installation Guide

Welcome to **SpectraSync**! Follow these steps to set up your environment and get started with multi-dimensional topic analysis.

---

### Table of Contents
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Installation Steps](#installation-steps)
- [Testing the Installation](#testing-the-installation)
- [Configuration](#configuration)
- [Optimization](#optimization)
---

### System Requirements
To run SpectraSync smoothly, ensure your system meets the following requirements:
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 16GB RAM (32GB+ recommended for large datasets)
- **Python**: Version 3.12.0
- **GPU Support**: Optional (CUDA required for GPU acceleration)
- **PostgreSQL**

### Environment Setup
SpectraSync works best with a dedicated virtual environment to avoid conflicts with other packages. We recommend using **Anaconda** or **virtualenv** for environment management.

#### Using Anaconda (recommended)
1. **Create a new environment**:
   ```bash
   conda create -n spectrasync python=3.12.0
   ```
2. **Activate the environment**:
   ```bash
   conda activate spectrasync
   ```

#### Using Virtualenv
1. **Create a new virtual environment**:
   ```bash
   python3 -m venv spectrasync_env
   ```
2. **Activate the environment**:
   - **Linux/macOS**:
     ```bash
     source spectrasync_env/bin/activate
     ```
   - **Windows**:
     ```bash
     spectrasync_env\Scripts\activate
     ```

### Installation Steps
Once your environment is set up, follow these steps to install SpectraSync and its dependencies.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/SpectraSync.git
   cd SpectraSync
   ```

2. **Install dependencies**:
   ```bash
   conda env create -f config/environment.yaml
   conda activate utma
   ```
   For CUDA support(**_SpectraSync requires CUDA_**):
      [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
      [macOS](https://docs.nvidia.com/cuda/archive/9.1/cuda-installation-guide-mac-os-x/index.html)


### Testing the Installation
Once installation is complete, verify that everything works by running the following commands:

1. **Run Tests**: Use the provided tests to verify correct installation and setup.
   ```bash
   python -m unittest discover -s tests
   ```

2. **Example Run**: Test a sample data file to ensure all dependencies are functioning.
   ```bash
   python spectrasync.py --input example_data.json --output results/
   ```
## Steps to Run the Notebook
   1. Open DocumentParser.ipynb, set corpus_path to the folder containing your documents, and adjust file paths for logging and outputs.
   2. Execute each cell to process and save the documents in the required format.
   3. Move the processed files to the designated input directory.

### Example Preprocessing: CDC's Morbidity and Mortality Weekly Report Journals
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

## **Example CLI Run**:
    ```bash
    python spectrasync.py \
       --username "postgres" \
       --password "admin" \
       --database "UTMA" \
       --corpus_label "mmwr" \
       --data_source "/path/to/your/data/preprocessed-documents/data.json" \
       --start_topics 20 \
       --end_topics 60 \
       --step_size 5 \
       --num_workers 10 \
       --max_workers 12 \
       --num_threads 1 \
       --max_memory 10 \
       --mem_threshold 9 \
       --max_cpu 110 \
       --futures_batches 30 \
       --base_batch_size 200 \
       --max_batch_size 300 \
       --log_dir "/path/to/your/log/" \
       2>"/path/to/your/log/terminal_output.txt"
      ```
You’re all set! Dive into SpectraSync and start exploring the hidden connections within your data. If you encounter issues, check our GitHub Wiki or open an issue in the repository.

---

### Configuration

Review the config/environment.yaml

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

### Optimization

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

#### Guidelines for Setting Key Batch Size Parameters

1. **Understanding Batch Size Impact**
   - **Base Batch Size**: Setting an appropriate base batch size is crucial. A batch size too small increases scheduling overhead, while a batch size too large can exhaust memory resources, leading to performance degradation. For large documents or complex tasks, use larger batch sizes to optimize resource use and reduce scheduling overhead.

---
