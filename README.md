
# Scalable LDA Insights Framework (SLIF)

The **Scalable LDA Insights Framework (SLIF)** is a comprehensive Python package designed to facilitate the analysis and interpretation of large-scale datasets using Latent Dirichlet Allocation (LDA). Developed through extensive research and enhanced with AI-assisted programming, SLIF harnesses robust tools for topic modeling, offering a reliable and scalable framework for transforming unstructured text data into meaningful insights.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Configuration](#configuration)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Overview

SLIF provides a flexible, end-to-end solution for large-scale topic modeling, combining powerful libraries and techniques in Python to streamline the entire workflow. At its core, SLIF includes advanced functions for calculating symmetric and asymmetric `alpha` and `beta` parameters (e.g., `calculate_numeric_alpha` and `calculate_numeric_beta`), leveraging the `Decimal` library for precision. These components ensure accuracy and fine-tuning for LDA, making it suitable for complex data analysis tasks.

SLIF’s development foundation rests on extensive research, grounded in systematic reviews of online resources, official documentation, community insights, and textbooks. This rigorous approach is further supported by real-time AI assistance, which contributes to coding, debugging, and templating, adding an intelligent "second pair" in pair programming. The combination of research-driven methodology with AI guidance ensures SLIF’s features are reliable and finely tuned to best practices in topic modeling and software development.

## Features

- **Scalable LDA Modeling**: Provides a high-performance `train_model` function for training LDA models with extensive configuration options.
- **Parallel & Concurrent Processing**: SLIF leverages Dask to support multithreading and distributed processing for large datasets, making parallel and concurrent processing a core feature.
- **Precise Parameter Calculation**: Calculates symmetric and asymmetric parameters with `Decimal` precision.
- **Comprehensive Data I/O**: Facilitates efficient handling of JSON-formatted text data through `get_num_records` and `futures_create_lda_datasets`.
- **Modular Structure**: Organized into standalone modules for data handling, modeling, visualization, and logging.
- **Advanced Visualization**: Offers `create_vis`, using pyLDAvis for interactive topic visualizations and Principal Coordinate Analysis (PCoA) for dimensionality reduction.
- **Detailed Logging**: Logs each step of the process, storing errors and runtime feedback for simplified debugging, with real-time monitoring available through Dask’s dashboard.

## Installation

### Prerequisites

- Python 3.12.0
- Anaconda (recommended for dependency management)
- PostgreSQL (required for data storage and integration)

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SLIF.git
   cd SLIF
   ```

2. Create and activate a new conda environment:
   ```bash
   conda create -n slif_env python=3.12.0
   conda activate slif_env
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up PostgreSQL:
   - Install PostgreSQL and create a database instance.
   - Configure database connection details in SLIF’s configuration to enable database writing and data persistence.

## Usage

1. **Prepare your data**: Ensure that your dataset is preprocessed and formatted as a list of lists, where each inner list represents a tokenized document (e.g., a paragraph or sentence). This structure is essential for SLIF's LDA training functions to process the data correctly. Common preprocessing steps include tokenization, removing stop words, and lemmatization.

   ### Example Data Format
   Here’s a sample structure from *Moby Dick* where each paragraph is tokenized as a list of words:
   ```json
   [
     ["precisely", "little", "particular", "interest", "think", "little", "watery", "drive", "spleen", "regulate", "circulation", "grow", "whenever", "drizzly", "involuntarily", "pause", "coffin", "warehouse", "bring", "funeral", "especially", "require", "strong", "principle", "prevent", "deliberately", "step", "street", "methodically", "knock", "people", "account", "substitute", "pistol", "philosophical", "flourish", "throw", "quietly", "surprising", "almost", "degree", "cherish", "nearly", "feeling"],
     ["insular", "belt", "wharf", "indian", "commerce", "surround", "street", "waterward", "extreme", "downtown", "battery", "wash", "cool", "breeze", "previous", "crowd", "gazer"],
     ["dreamy", "afternoon", "thence", "northward", "see?—poste", "silent", "sentinel", "thousand", "thousand", "mortal", "reverie", "lean", "spile", "seat", "look", "bulwark", "rigging", "strive", "well", "seaward", "landsman", "plaster", "counter", "nail", "bench", "clinch", "field"]
   ]
   ```

2. **Adjust configuration settings**: Customize any necessary settings in the configuration file to match your environment.

3. **Run the main script**: Execute `topic_analysis.py` with desired parameters:
   ```bash
   python topic_analysis.py --params <parameter_values>
   ```

### Script Parameters

SLIF’s main script allows for dynamic parameter configuration to customize LDA training, data processing, and visualization. Refer to each script’s documentation for specific parameters.

## Components

The SLIF framework consists of well-defined modules that each handle a specific aspect of the LDA workflow:

### 1. **topic_analysis.py**

   Orchestrates the entire LDA pipeline from data processing to model training and visualization. It manages error handling and coordinates various framework components.

### 2. **alpha_eta.py**

   Contains functions for calculating symmetric and asymmetric `alpha` and `beta` parameters, crucial for optimizing LDA model topic distribution. The use of the `Decimal` library enhances precision in these calculations.

### 3. **data_io.py**

   Manages data I/O operations, with functions for loading, preprocessing, and structuring data. It supports JSON format and includes utilities like `get_num_records` for efficient data handling.

### 4. **logging_helper.py**

   Implements a logging framework to capture runtime information and errors. It redirects errors to `stderr_out.txt`, allowing users to troubleshoot issues efficiently.

### 5. **model.py**

   Houses the LDA model training process using Gensim’s LDA module. The `train_model` function is highly customizable, supporting corpus creation, dictionary generation, and coherence scoring.

### 6. **process_futures.py**

   Facilitates concurrent processing with `concurrent.futures`, enabling parallel LDA training on large datasets. It supports multiprocessing and distributed computing through Dask, making the framework more scalable.

### 7. **utils.py**

   Provides general-purpose utility functions that support various components of the framework, streamlining tasks such as data transformations and helper functions.

### 8. **visualization.py**

   Generates visualizations using pyLDAvis and PCoA, creating interactive and informative representations of LDA topics, including term relevance and topic distributions.

### 9. **WriteToPostgres.py**

   Integrates with PostgreSQL, a necessary component for SLIF’s database functions. It enables users to store model outputs and metadata, ensuring model results can be further analyzed and persistently stored.

## Configuration

Configuration is designed to be easily modifiable. Key settings for LDA parameters, file paths, and database connections are located in the configuration section of `topic_analysis.py`. SLIF allows users to dynamically set these parameters using environment variables, making it flexible for different environments and workflows.

## Logging

SLIF incorporates detailed logging throughout the modeling and data processing pipeline. The logs are stored in `stderr_out.txt`, capturing errors and providing status updates that aid in debugging and monitoring. For real-time monitoring and diagnostics, Dask’s **Dashboard** offers a comprehensive view of SLIF's processing tasks, including task progress, CPU usage, memory load, and other performance metrics. This Dashboard enables users to track and manage computational workloads as they run, providing insights into system resource usage and helping to optimize performance in real time.

### Example of Logging Setup in `topic_analysis.py`:

```python
sys.stderr = open(f"{LOG_DIRECTORY}/stderr_out.txt", "w")
```

This approach ensures that any issues are logged systematically, providing users with insights into each step of the modeling process, while Dask’s Dashboard gives a powerful visual interface for real-time monitoring.

## Contributing

We welcome contributions! Fork the repository, create a new branch, and submit a pull request. Please follow our contribution guidelines and adhere to the code style standards for consistency.

## License

This project is licensed under the MIT License.

--- 
