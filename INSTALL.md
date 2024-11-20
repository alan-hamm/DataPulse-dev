# INSTALL.md

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation Instructions](#installation-instructions)
  - [PostgreSQL Installation](#postgresql-installation)
  - [Create a Conda Environment (Recommended)](#create-a-conda-environment-recommended)
  - [Create a Virtual Environment using Python's venv (Alternative)](#create-a-virtual-environment-using-pythons-venv-alternative)
- [PyTorch and CUDA Version Compatibility](#pytorch-and-cuda-version-compatibility)
- [Testing the Installation](#testing-the-installation)
  - [Testing PyTorch Installation](#testing-pytorch-installation)
  - [Testing GPU for CUDA Support](#testing-gpu-for-cuda-support)

## Prerequisites

This project demands specific tools and packages for optimal functionality. CuPy, CUDA, and PyTorch are foundational for DataPulse's processing power, while PostgreSQL is its neural core—without it, the system cannot run. PostgreSQL serves as DataPulse's memory and database engine, seamlessly storing, structuring, and recalling the intricate layers of topics and models generated across dimensions. Here’s how to get PostgreSQL set up as a prerequisite for DataPulse:

- **PyTorch**: Essential for efficient, GPU-accelerated tensor operations, PyTorch facilitates the rapid computation of coherence metrics and other complex numerical tasks. By harnessing GPU hardware, PyTorch accelerates processing times significantly, making it ideal for handling the demands of large-scale topic modeling within DataPulse. Its optimized handling of tensor operations ensures that even the most intensive calculations are performed with minimal latency, which is crucial for maintaining high-speed coherence assessments and model evaluations.

- **CuPy**: CuPy supports high-performance numerical calculations, designed to leverage CUDA-enabled GPUs for accelerated processing. Acting as a GPU-powered counterpart to NumPy, CuPy excels at handling large-scale matrix operations and other tasks that benefit from parallel computation on GPU hardware. This makes it particularly effective for DataPulse's coherence sampling and statistical functions, where rapid, large-scale numerical processing is essential.

**Note**: It is critical to ensure that the versions of PyTorch and CUDA match appropriately for your hardware. Refer to [PyTorch's installation guide](https://pytorch.org/get-started/locally/) for a detailed compatibility matrix.

**Python Version**: The package was developed using Python 3.12.0. It is recommended to use this version or ensure compatibility if using a different Python version.

## System Requirements

To run this project efficiently, your system should meet the following requirements:

- **Operating System**:

  - Linux (Ubuntu 18.04 or newer recommended)
  - macOS (10.15 or newer)
  - Windows 10 or later

- **Hardware**:

  - **CPU**: Multi-core processor (Intel i5 or AMD equivalent, or better).
  - **GPU**: CUDA-enabled GPU with at least 6GB VRAM for GPU acceleration. This is particularly important if you plan to utilize PyTorch and CuPy for performance gains. An NVIDIA GPU with CUDA compute capability 3.5 or higher is recommended.
  - **Memory**: At least 16GB RAM is recommended for efficient processing, particularly when working with large datasets.
  - **Storage**: At least 20GB of free storage for software installations and intermediate files.

- **Software**:

  - **Python**: Version 3.8 or newer (developed using Python 3.12.0).
  - **Conda**: Recommended for managing the environment and dependencies.
  - **CUDA Toolkit**: Required for GPU acceleration. Make sure the version matches the PyTorch version.
  - **NVIDIA Drivers**: Must be up to date to ensure compatibility with the CUDA toolkit.
  - **PyTorch and CuPy**: Required for GPU-accelerated tensor operations and numerical computations.

## Installation Instructions

### PostgreSQL Installation

1. **Install PostgreSQL**: Download and install PostgreSQL from the official site: [PostgreSQL Downloads](https://www.postgresql.org/download/).

2. **Setup a Dedicated Database**:

   -  After installation, create a database specifically for DataPulse.
   -  Make note of the database name, username, and password—these credentials will be required for configuration.

3. **Configure Database Access**:

   -  Ensure PostgreSQL is running and accessible. You may want to adjust settings to allow remote connections if DataPulse will be deployed in a distributed environment.  
   -  Ensure that your PostgreSQL user has adequate privileges for creating tables, writing data, and managing resources as DataPulse builds its multi-dimensional topic models.

4. **Verify Connection**:

   -  Test the connection to ensure PostgreSQL is configured correctly. This can be done within the DataPulse setup process or by running a quick script to confirm connectivity.

### Create a Conda Environment (Recommended)

Using `conda` is highly recommended due to its robust environment management and ability to resolve dependencies easily, including GPU libraries like PyTorch and CuPy.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/alan-hamm/SpectraSync.git
   cd DataPulse
   ```

2. **Create and Activate Conda Environment**: Use the provided `environment.yaml` file to create the environment.

   ```bash
   conda env create -f environment.yaml
   conda activate DataPulse
   ```

   **Why `environment.yaml` is Best**: The `environment.yaml` file captures the exact package specifications, ensuring all dependencies (including versions of PyTorch and CUDA) are installed consistently. This is crucial for avoiding compatibility issues, particularly with GPU-related libraries. Update the `prefix: /path/to/.conda/envs/DataPulse` to point to your DataPulse environment location.

3. **Install Additional Dependencies** (if needed):

   ```bash
   conda install <additional-package>
   ```

### Create a Virtual Environment using Python's venv (Alternative)

If you prefer to use a Python virtual environment (`venv`), follow these steps. Note that additional steps may be required to manually handle CUDA dependencies and library versions.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/alan-hamm/SpectraSync.git
   cd DataPulse
   ```

2. **Create and Activate Python Virtual Environment**:

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies**: You can either use `requirements.txt` or `environment.yaml`.

    - **Using environment.yaml (Preferred Option)**: For optimal compatibility, especially with CUDA-related dependencies, use environment.yaml with Conda:
    ```bash
    conda env create -f environment.yaml
    ```
    This will ensure the environment includes all Conda and pip dependencies specified in environment.yaml.

    - **Using requirements.txt (Alternative Option)**: If you prefer or need to install dependencies using requirements.txt, or if the environment.yaml includes dependencies that are incompatible outside Conda, you can set up a virtual environment (e.g., using venv) or install directly with pip:
    ```bash
    pip install -r requirements.txt
    ```

    - **Updating** requirements.txt from environment.yaml: If you want to generate a requirements.txt from environment.yaml (for use with pip in environments where Conda is unavailable):
    ```bash
    conda list --export > requirements.txt
    ```
     
   **Note**: Installing via `requirements.txt` does not guarantee version compatibility as strictly as `conda` with `environment.yaml` does, especially for CUDA-related dependencies.

## PyTorch and CUDA Version Compatibility

When installing PyTorch, you must ensure that the version of CUDA installed matches the version required by PyTorch. Using incompatible versions may lead to runtime errors or inefficient use of GPU resources. To determine the appropriate versions for your system:

1. Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).
2. Select your OS, package (Conda or pip), and desired CUDA version.
3. Follow the installation command provided to ensure compatibility.

_For example_, to install PyTorch with CUDA 11.7 (this is an example; you must consult the official documentation to ensure compatibility with your system and hardware):

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch
```

Ensure your GPU drivers are also up to date and compatible with your chosen version of CUDA.


## Testing the Installation

### Testing PyTorch Installation

To verify that PyTorch has been successfully installed and is configured for GPU support, run the following command:

```python
import torch
print(torch.cuda.is_available())
```

If `True` is returned, PyTorch is able to access the GPU. If `False` is returned, double-check your CUDA installation and compatibility.

### Testing GPU for CUDA Support

To check if your GPU supports CUDA and that your system is configured correctly, use the following command:

```bash
nvcc --version
```

This command will display the version of CUDA installed. Make sure that your GPU drivers are up to date and that the CUDA version is compatible with your hardware.

You can also use Python to verify CUDA compatibility with PyTorch:

```python
import torch
print(torch.cuda.get_device_name(0))
```

This will print the name of your GPU if it is CUDA compatible.

## Testing the Installation

After setting up the environment and installing all necessary dependencies, you can test if everything is working properly by running the following:

```bash
python -m unittest discover tests/
```

This command will run the unit tests included in the `tests/` directory to ensure that all components are functioning as expected.

---

You’re all set! Dive into DataPulse and start exploring the hidden connections within your data. If you encounter issues, check our GitHub Wiki or open an issue in the repository.

