# The Scalable LDA Insights Framework (SLIF)

The Scalable LDA Insights Framework (SLIF) is a comprehensive Python package designed to facilitate the analysis and interpretation of large-scale datasets through Latent Dirichlet Allocation (LDA). Developed with a blend of AI-assisted programming and research-driven methodologies, SLIF combines robust tools for topic modeling with advanced resource management, extensive PostgreSQL integration, and real-time feedback features, making it an exceptional tool for researchers and data practitioners.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Options](#configuration-options)
- [Logging and Monitoring](#logging-and-monitoring)
- [Contributing](#contributing)
- [License](#license)

## Overview

SLIF enables users to conduct scalable, high-precision topic modeling on large datasets. By leveraging Dask for distributed computing, Gensim for LDA model training, and PostgreSQL for managing model metadata, SLIF provides a powerful environment for performing intensive text analysis at scale. Its development harnessed AI as an advanced 'second pair' in pair programming, aiding with code structure, debugging, and research integrations. This collaborative approach, combined with deep exploration of best practices from Stack Exchange, official documentation, and systematic online research, ensures SLIF's robustness.

### Core Components
- **AI-Assisted Programming**: Real-time coding support, intelligent snippets, and debugging.
- **Integrated Research**: Best practices and knowledge from documentation and forums enhance SLIF's reliability.
- **Data Utilities**: Efficient handling of JSON-formatted text data for topic modeling.

## Key Features

- **Train LDA Models with Precision**: SLIF's `train_model` function empowers users to train LDA models with diverse configurations while managing corpus creation, dictionary generation, and coherence score computation. This function is designed to support parallel and concurrent processing, distributed computing, hyperthreading, and batch processing.
  
- **Advanced Parameter Control**: SLIF includes functions to calculate symmetric and asymmetric alpha and beta parameters, `calculate_numeric_alpha` and `calculate_numeric_beta`, providing flexibility and precision with the Decimal library.
  
- **Efficient Data I/O Operations**: Utilities like `get_num_records` and `futures_create_lda_datasets` enable streamlined data handling for large JSON datasets.
  
- **PostgreSQL Integration for Metadata**: SLIF efficiently logs model metadata and performance metrics in PostgreSQL, facilitating easy retrieval and analysis. The framework includes:
  - **Dynamic Table Generation**: Using the `create_dynamic_table_class` function, SLIF dynamically generates SQLAlchemy model classes for custom PostgreSQL tables.
  - **Metadata Storage**: SLIF's `add_model_data_to_database` function handles dynamic data storage in PostgreSQL, including efficient zipping of large text bodies before insertion.
  
- **Comprehensive Logging**: SLIF features detailed logging at every stage, giving users a clear view of the modeling process and making troubleshooting straightforward. 

- **Adaptive Resource Management**: Adjusts batch sizes and worker allocation based on CPU and memory utilization, maximizing processing efficiency.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/slif_repository.git
   cd slif_repository
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up PostgreSQL:
   - Install PostgreSQL and create a new database (e.g., `SLIF`).
   - Configure the PostgreSQL URI (username, password, host, port, and database name) in the application settings.

4. (Optional) Set up the Dask dashboard for real-time monitoring.

## Usage

SLIF is designed to support flexible configurations for topic modeling tasks. To start an analysis, use the following command structure:

### Example Commands

```bash
# Example 1: Basic configuration
python topic_analysis.py --corpus_label "sample" --data_source "/data/sample.json" --start_topics 20 --end_topics 100 --step_size 5

# Example 2: Advanced configuration with memory and CPU utilization settings
python topic_analysis.py --corpus_label "advanced" --data_source "/data/advanced.json" --start_topics 50 --end_topics 500 --step_size 10 --max_memory "10GB" --max_cpu 85
```

## Configuration Options

Below is a list of configurable parameters:

- `--corpus_label`: Specifies the corpus label.
- `--data_source`: Path to the JSON file containing tokenized sentences.
- `--start_topics`: Minimum number of topics to model.
- `--end_topics`: Maximum number of topics to model.
- `--step_size`: Incremental step for the topic range.
- `--num_workers`: Minimum cores for Dask workers.
- `--max_workers`: Maximum cores for Dask workers.
- `--max_memory`: Maximum memory allocation per worker.
- `--max_cpu`: Maximum CPU utilization threshold.
- `--log_dir`: Directory for logging outputs.
- `--root_dir`: Root directory for model outputs and data.
- **PostgreSQL URI**: Configure the URI to connect to the PostgreSQL database, which will store metadata and model performance information.

## Logging and Monitoring

SLIF's logging framework offers detailed feedback on every stage of the topic modeling process, with integration for PostgreSQL and the Dask dashboard:

- **PostgreSQL Logging**: Logs model metadata, performance, and visualization paths to a specified PostgreSQL database.
- **Logging Levels**: Adjustable verbosity levels to support both detailed debugging and streamlined output.
- **Performance Reports**: Generates HTML performance reports for batch processes, aiding in performance monitoring.
- **Dask Dashboard**: Real-time monitoring of Dask cluster at `http://localhost:8787`.

## License

SLIF is licensed under the MIT License. 

---

By leveraging SLIF's comprehensive AI-assisted programming features, rich research integration, and PostgreSQL database support, users gain access to a robust framework for efficient, scalable topic modeling and in-depth analysis of large-scale datasets.