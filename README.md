
# SpectraSync: Neural Intelligence Meets Multi-Dimensional Topic Analysis

### Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Visualization](#visualization)
- [Machine Learning and Analysis](#machine-learning-and-analysis)
- [Optimization with GPU Acceleration](#optimization-with-gpu-acceleration)
- [Distributed Configuration](#distributed-configuration)
- [Batch Configuration](#batch-configuration)
- [Example Preprocessing](#example-preprocessing-cdcs-morbidity-and-mortality-weekly-report-journals)

---

### Overview
In a world where information flows like neon-lit rain over a vast, electric metropolis, SpectraSync emerges as the ultimate conduit, channeling torrents of data into discernible pulses. This isn’t merely about analyzing text; it’s about orchestrating the digital symphony of topics across dimensions, decades, and digital realms. SpectraSync stands as a sentinel of cognitive insight, bridging neural intelligence with multi-dimensional topic analysis, illuminating shifts in language, and resonating with the evolving patterns embedded in the written word.

Designed for those who operate on the cutting edge, SpectraSync doesn't just passively process—it's alive with the potential to capture, track, and synchronize the underlying threads woven into the corpus of data. Each session within SpectraSync reveals a dynamic spectrum, an unbroken sequence of thematic waves, morphing and re-aligning like the thoughtforms of an advanced intelligence. This platform isn’t your traditional toolkit—it’s a neural-inspired, data-driven powerhouse with a singular mission: to bring coherence to the chaos of information.

### Key Features
- **Adaptive Resource Management**: SpectraSync harnesses the formidable power of [Dask](https://www.dask.org/) for distributed parallelization. This ensures a seamless orchestration of resources across processors, dynamically adjusting to tackle vast data landscapes and high computational demands without skipping a beat. The system adapts, self-modulates, and optimizes, deploying cores and threads in perfect synchrony to handle even the heaviest data streams with precision.

- **Multi-Phase Topic Analysis**: Far from the confines of linear processing, SpectraSync performs a tri-phased exploration—train, validation, and test—that keeps models pristine and refined. By treating each phase as a unique dataset, it preserves the sanctity of unbiased learning, diving deep into intricate data patterns. Each model builds upon an evolving dictionary of terms, maintaining distinct corpora for each phase to deliver a thorough, multi-dimensional perspective.

- **Diachronic Topic Tracking**: SpectraSync traverses time itself, tracking the shifts in language and evolving terminologies. Users can trace topics across years, even decades, capturing emergent themes and the twilight of others. By mapping how concepts morph, persist, or disappear over time, it uncovers the narrative threads running through historical and modern text alike.

- **Precision Metrics**: With coherence, convergence, and perplexity metrics in hand, SpectraSync doesn’t leave quality to chance. Each metric is tuned with algorithmic precision, fine-tuned across myriad parameters to capture relevance, thematic clarity, and linguistic structure. A spectrum of scoring metrics ensures that every model reflects a refined, accurate portrayal of the data’s hidden dimensions.

### Visualization
Visualization in SpectraSync is an immersive experience, pushing the boundaries of interaction in the digital realm. Each visualization is a portal into the unseen, rendering complex datasets into intuitively graspable maps. Bokeh and pyLDAvis power the platform’s visual dimensions, creating an environment where data doesn’t just speak—it resonates.

- **2D and 3D Topic Mapping**: SpectraSync brings your data into vivid relief, visualizing topics in two or three dimensions, allowing you to explore the intricate networks of ideas that link one document to another. It’s not just about seeing data; it’s about inhabiting it.

- **Temporal Topic Flow**: As topics shift and reform across timelines, SpectraSync captures this dynamic evolution, letting you witness how language trends and persists. It becomes a chronicle of change, a digital archive of thought made manifest in visual form.

- **Interactive Model Visualization**: With SpectraSync, you don’t just view models—you engage with them. Each visualization offers an interactive portal, inviting you to dissect topics and understand the underlying themes, creating a space where exploration leads to revelation.

### Machine Learning and Analysis
SpectraSync is more than just a machine learning engine; it’s a digital mind, configured to dissect, explore, and evolve through data. Its machine learning core thrives on advanced algorithms that go beyond simple clustering, instead capturing the full spectrum of thematic evolution. Using Gensim’s LDA (Latent Dirichlet Allocation) model, SpectraSync delivers an analysis that is not only multi-layered but dynamically optimized.

- **Hyperparameter Tuning & Adaptive Model Selection**: SpectraSync applies a rigorous methodology to find the most resonant model configurations. Hyperparameters are fine-tuned in a ceaseless pursuit of coherence and perplexity optimization, ensuring models yield insights of the highest clarity and relevance.

- **Dynamic Topic Allocation**: The architecture of SpectraSync allows it to shift and recalibrate in real time, making dynamic adjustments that tailor-fit each data structure. This adaptability enables SpectraSync to capture even the most nuanced patterns, providing a level of analytical depth that traditional models simply cannot achieve.

- **High-Speed Convergence Tracking**: Speed is of the essence. SpectraSync’s convergence tracking allows it to rapidly navigate through the topic space, minimizing computational delays while maximizing insight—a neural engine that never sleeps.

### Optimization with GPU Acceleration

SpectraSync leverages GPU acceleration for efficient processing of large datasets, using the following tools:

- **CUDA**: CUDA is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows software developers to use NVIDIA GPUs for general-purpose processing (an approach known as GPGPU). CUDA is necessary for leveraging GPU hardware and enabling acceleration for both PyTorch and CuPy.

- **PyTorch**: PyTorch is used for deep learning and tensor operations, enabling significant speed improvements in training and evaluation processes by utilizing GPU acceleration. The calculations related to coherence metrics, such as cosine similarity, are performed using PyTorch tensors, which can be processed much faster on a GPU compared to a CPU.

- **CuPy**: CuPy provides an interface similar to NumPy, but all array computations are executed on the GPU, resulting in considerable speed improvements for numerical calculations. This project uses CuPy to accelerate matrix operations and other numerical tasks that are computationally intensive.

By using GPU-accelerated libraries like PyTorch and CuPy, this project achieves significant performance gains compared to CPU-only execution. Users should pay close attention to the compatibility between the versions of CUDA, PyTorch, and their GPU drivers to fully utilize the GPU's capabilities and avoid runtime errors.


---


### Distributed Configuration

   **_By default, the settings in `distributed.yaml` are optimized for high-performance processing with Dask on systems with significant CPU and memory resources. Adjust as needed to suit your environment._**

This project includes a custom `distributed.yaml` file for configuring Dask. The `distributed.yaml` file is located in the [`config/`](https://github.com/alan-hamm/SpectraSync/tree/main/config) directory and contains recommended settings for Dask performance and resource management tailored for SpectraSync's processing requirements.

To ensure your Dask environment is correctly configured, follow these steps:

   1. **Review the `distributed.yaml` File**  
      Examine the `config/distributed.yaml` file to understand its settings, especially if you need to adjust resource limits based on your system’s specifications.

   2. **Customize if Necessary**  
      Depending on your hardware and workload, you may want to customize certain values (e.g., memory limits, CPU thresholds) in the `distributed.yaml` file.

   3. **Refer to Setup Instructions**  
      For more detailed instructions on configuring the Dask dashboard and securing it for local access, see the `Dask_Dashboard_Setup_Instructions.txt` file in the `config/` directory.

### Batch Configuration

   Configuring `futures_batches`, `base_batch_size`, and `max_batch_size` is critical to balancing resource utilization and achieving efficient processing times, especially on high-performance systems. The script `batch_estimation.py` is provided for adaptive batch size estimation based on document complexity, memory, and CPU limits. This script is recommended for anyone running SpectraSync on datasets with varying document sizes or on systems with constrained resources.


   ### Guidlines for Setting Key Batch Size Parameter
   
   1. **Understanding Batch Size Impact**

      -  **Base Batch Size**: Setting an appropriate base batch size is crucial. A batch size too small will increase scheduling overhead, while a batch size too large can exhaust memory resources, leading to performance degradation. For large documents or complex tasks, use larger batch sizes to optimize resource use and reduce scheduling overhead. For smaller tasks, use smaller batch sizes to increase task concurrency and CPU utilization.
   
      -  **Max Batch Size**: Defines the upper limit for document processing per batch. Adaptive batching helps to manage tasks dynamically based on resource availability. Setting this value appropriately helps SpectraSync adapt to different document types without exhausting memory.

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

      Adaptive batch sizes calculated by 'batch_estimation.py' allow the SpectraSync framework to better handle document variability and optimize resource usage. This approach reduces memory-related issues, as batch sizes are optimized for current system capacity and workload, ensuring smooth execution without overwhelming resources.

   5. **Monitoring and Iterative Adjustment**

      Use the [Dask dashboard](https://docs.dask.org/en/latest/dashboard.html) to observe task distribution, memory usage, and performance metrics. Monitor the impact of changes in batch size on system utilization, and adjust batch sizes if memory or CPU usage approaches system thresholds.

   6. **RAM Allocation and Management**

      SpectraSync is memory-intensive, especially when handling large datasets or high batch sizes. Setting a high memory_limit in the Dask LocalCluster configuration is recommended if system RAM allows. For optimal memory usage:

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

---


### Example Preprocessing: CDC's Morbidity and Mortality Weekly Report Journals
A real-world application of SpectraSync’s data preprocessing capabilities can be seen in analyzing the [MMWR Journals](https://www.cdc.gov/mmwr/), extracted from the [CDC text corpora for learners](https://github.com/cmheilig/harvest-cdc-journals/). Each report in these journals is treated as a standalone document and requires specific preprocessing steps to align with SpectraSync's standards, including tokenization and formatting as a bag-of-words model.

By organizing and structuring the text data in this format, SpectraSync can identify recurring themes and track the evolution of key public health topics, such as "infection control," "vaccine efficacy," and "disease prevention." This structured approach allows SpectraSync to perform diachronic analyses of topic shifts over time, revealing insights into public health trends and topic persistence. Preprocessing each document in this way prepares it for the advanced topic modeling and analysis that SpectraSync provides.

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
   This format aligns with the project’s requirements, enabling SpectraSync to analyze the thematic structure and evolution of health topics in CDC reports.

   The project supports preprocessing for a range of CDC’s journal content, including _Emerging Infectious Diseases_([EID](https://wwwnc.cdc.gov/eid)) and _Preventing Chronic Disease_([PCD](https://www.cdc.gov/pcd)). Available resources include CDC documents, spanning 42 years: [HTML Mirrors of MMWR, EID, and PCD](https://data.cdc.gov/National-Center-for-State-Tribal-Local-and-Territo/CDC-Text-Corpora-for-Learners-HTML-Mirrors-of-MMWR/ut5n-bmc3/about_data) and associated [Corpus Metadata](https://data.cdc.gov/National-Center-for-State-Tribal-Local-and-Territo/CDC-Text-Corpora-for-Learners-MMWR-EID-and-PCD-Art/7rih-tqi5/about_data).

## **Example CLI Run**:
    ```bash
    python spectrasync.py \
       --username "postgres" \
       --password "admin" \
       --database "SpectraSync" \
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

---

SpectraSync stands at the intersection of neural intelligence and advanced data analysis, ready to uncover the spectral layers within the fabric of language. Step into a world where insights pulse to life, patterns converge, and knowledge flows like an electric current through the digital landscape. Welcome to the future of multi-dimensional topic analysis.
