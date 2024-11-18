#!/usr/bin/env python
# coding: utf-8

# # **ETL for SpectraSync**
# 
# ### **Author**: [Your Name]  
# ### **Date**: [Date of Notebook Creation]
# 
# ---
# 
# ## **Notebook Overview**
# 
# This notebook is part of the **Unified Topic Modeling Analysis (UTMA)** project, focusing on the **Extract, Transform, Load (ETL)** process for preparing textual data from various sources—including medical journals, novels, and general public health documents—for topic modeling and diachronic analysis.
# 
# ### **1. Objectives of the ETL Process**
# The primary purpose of this ETL notebook is to preprocess textual data from multiple time periods and sources in a manner that ensures high variability in the resulting corpora, which is crucial for effective topic modeling and subsequent diachronic analysis. Specifically, this notebook will:
# - **Combine Data from Multiple Time Periods**: Expand the temporal scope by merging different corpora (e.g., CDC MMWR publications from 2010-2014, 2014-2019, and 2020-2024) into a unified text to increase diversity.
# - **Balanced Sampling and Temporal Tagging**: Ensure that each document is tagged with its respective time period, enabling balanced representation across temporal spans during training.
# - **Diversity-Oriented Batch Preparation**: Implement strategies to diversify the data, such as weighted sampling and clustering, to mitigate the limited variance that impacts model quality.
# 
# ### **2. Data Sources**
# - **CDC MMWR Journals** (2010-2024): Split into three segments (2010-2014, 2014-2019, 2020-2024) to allow for **temporal analysis** and facilitate the understanding of changes in public health discourse.
# - **Additional Texts (Future Expansion)**: General-purpose pipeline to support documents such as novels, newspaper articles, and other public health records.
# 
# ### **3. Key Preprocessing Steps**
# This notebook will perform the following steps to ensure that the final output is prepared for topic modeling in a robust and variance-rich manner:
# - **Text Extraction**:
#   - Extract raw paragraphs from input documents and corpora, ensuring all parts of the data are captured.
#   
# - **Content Transformation**:
#   - **Tokenization**: Utilize spaCy’s NLP pipeline to tokenize the paragraphs.
#   - **Stop Word Handling**: Evaluate and experiment with retaining or removing stop words, considering their role in providing context and facilitating diachronic analysis.
#   - **POS Filtering and Character Length Adjustment**: Initially process only content-rich words (nouns, adjectives, verbs, adverbs) with 5+ characters, but allow flexibility to include other parts of speech or adjust the threshold based on analysis needs.
#   - **Temporal Tagging**: Tag paragraphs with their respective time period to retain temporal information for balanced sampling and diachronic analysis.
# 
# ### **4. Strategies for Variability Enhancement**
# Given the challenges of **limited variance** within homogenous corpora (e.g., CDC MMWR journals), this notebook employs multiple strategies to ensure variability:
# - **Balanced Sampling Across Time Periods**: Incorporate a balanced mix of paragraphs from all three time segments in each training batch.
# - **Diversity-Based Batch Formation and Weighted Sampling**: Use techniques like **TF-IDF-based clustering** and **weighted sampling** to diversify the batches, ensuring maximum differentiation between topics.
#   
# ### **5. Data Outputs**
# - **Unified Corpus**: A unified text, comprising paragraphs from all three temporal segments, tagged accordingly.
# - **Segmented Corpora**: Individual segments for each time period, maintaining temporal integrity for diachronic analysis.
# - **Batch Preparation for Training**: Diverse and balanced batches of paragraphs ready for input into the **Gensim LDA model**, ensuring that variability is maximized to improve coherence, perplexity, and topic differentiation.
# 
# ### **6. Challenges Addressed in ETL**
# - **Limited Corpus Variance**: By expanding the data scope and employing balanced sampling and batch diversification, the goal is to address and mitigate the uniformity issues that limit the richness of LDA topics.
# - **Preparation for Future Analysis**: This notebook not only sets the foundation for robust LDA modeling but also prepares the data for **dynamic topic modeling (DTM)** and **hierarchical LDA (hLDA)**, to be implemented in subsequent stages for a richer understanding of topic evolution over time.
# 
# ### **7. Future Considerations**
# - **Data Augmentation**: Consider data augmentation techniques such as synonym replacement or back-translation to further enhance variability in future iterations.
# - **Advanced Modeling Approaches**: This ETL process will feed into subsequent modeling efforts, including experimenting with **Non-Negative Matrix Factorization (NMF)** and **hierarchical topic modeling**, as well as exploring **adaptive thresholding** for coherence metrics.
# 
# ---
# 
# ### **Notebook Flow**
# 1. **Setup and Initialization**: Import necessary libraries, set up paths, and initialize logging.
# 2. **Data Extraction**: Load and preview the data from the three corpora (2010-2014, 2014-2019, 2020-2024).
# 3. **Text Preprocessing**: Tokenize, filter, and tag data for each paragraph.
# 4. **Batch Formation**: Implement batch formation strategies (balanced sampling, diversity-based clustering).
# 5. **Output Generation**: Save the processed data for further use in topic modeling.
# 
# **Note**: This notebook is designed to be modular and adaptive, supporting different types of textual data as we continue to refine the topic modeling pipeline.
# 
# --- 
# 
# **Let’s get started by setting up the environment and extracting the data.** 
# 

# In[ ]:
# Standard Library Imports
import os                  # Provides functions for interacting with the operating system, e.g., file handling.
import sys                 # Provides system-specific parameters and functions.
import re                  # Supports regular expressions for text manipulation and pattern matching.
import csv                 # Facilitates reading from and writing to CSV files.
import json                # Enables reading from and writing to JSON files, often used for structured data.
import gzip                # Provides functions for working with gzip-compressed files.
import codecs              # Handles different text encodings, important for text data processing.
import shutil              # Provides functions for high-level file operations, e.g., copying and moving files.
import pickle              # Serializes and deserializes Python objects.
import hashlib             # Provides hashing functions such as MD5, useful for generating unique IDs or checking data integrity.
import threading           # Supports concurrent threads of execution within a process.
import socket              # Provides low-level networking interfaces.
import itertools           # Implements efficient looping and combinatorial tools, e.g., cartesian products.
import argparse            # Parses command-line options, arguments, and subcommands.
import random              # Implements pseudo-random number generators.
import math                # Provides mathematical functions like ceil, floor, sqrt, etc.
import datetime            # Provides classes for manipulating dates and times.
from time import time, sleep  # Allows timing operations and creating delays.
from collections import defaultdict  # Implements dictionaries with default values, useful for tracking counts.
import statistics          # Provides statistical functions like mean, median, stdev.
import pprint as pp

# Logging Imports
import logging             # Provides logging functionalities to monitor code execution.
import logging.config      # Configures the logging library with a configuration file.
import logging.handlers    # Provides additional handlers for logging, e.g., rotating file handler.
import yaml                # Used for loading and parsing configuration files in YAML format.

# Encoding and Parsing Imports
import chardet             # Detects character encoding of text files, allowing for accurate reading of various encodings.
import unicodedata         # Handles Unicode character information, useful for identifying and removing non-printable characters.
from bs4 import BeautifulSoup  # Parses HTML content, enabling extraction of specific tags (e.g., <p> tags) for processing.
import html5lib            # Parser for HTML5, used by BeautifulSoup for web scraping.
import gzip                # Handles gzip-compressed files for reading and writing.

# NLTK Imports
import nltk                                # Natural Language Toolkit, a suite for text processing.
from nltk.corpus import stopwords          # Provides lists of stop words to remove from text.
from nltk.corpus.reader.api import CorpusReader  # Base class for reading and structuring corpora.
from nltk import sent_tokenize, pos_tag, wordpunct_tokenize  # Tokenizers and POS tagger for sentence processing.
stop_words = stopwords.words('english')    # Initializing English stop words list for filtering out common words.

# Gensim Imports
import gensim                              # Library for topic modeling and word vector creation.
from gensim.models import Word2Vec, ldamulticore  # Word2Vec for word embeddings, ldamulticore for topic modeling.
from gensim.models.phrases import Phrases, Phraser  # Constructs multi-word phrases (e.g., bigrams) from tokenized text.
import gensim.corpora as corpora           # Handles creation of dictionaries and corpora from text data.
from gensim.utils import simple_preprocess  # Preprocesses text into a list of tokens.
from gensim.corpora import Dictionary      # Handles the creation of a dictionary from tokenized text.

# SpaCy Import (specific model)
import en_core_web_lg                      # SpaCy's large English NLP model for advanced text processing.
nlp = en_core_web_lg.load(disable=['parser', 'ner'])  # Loads the model, with parsing and named entity recognition disabled for efficiency.

# Readability Import
from readability.readability import Unparseable  # Exception handling for parsing errors in HTML.
from readability.readability import Document as Paper  # Extracts readable content from HTML, discarding noise.

# Data Processing and Scientific Libraries
import numpy as np                         # Supports efficient numerical operations on large arrays and matrices.
import pandas as pd                        # Data analysis library for handling structured data (e.g., DataFrames).
import dask                                # Parallel computing library for handling large computations across multiple cores.
from dask.distributed import get_client, Client, LocalCluster, performance_report, wait  # Distributed Dask functionalities for parallel processing.
from distributed import Future             # Represents a future result from a Dask computation.
from sklearn.manifold import TSNE          # Dimensionality reduction for visualizing high-dimensional data.
from matplotlib import pyplot as plt       # Plotting library for creating visualizations.
from tqdm import tqdm                      # Adds progress bars to loops, useful for monitoring lengthy operations.


# SQL and ORM Libraries
import sqlalchemy                          # SQL toolkit and Object Relational Mapper for accessing SQL databases.

# Tornado and Bokeh Imports
import tornado                             # Asynchronous networking library, used by Dask and distributed systems.
from tornado.iostream import StreamClosedError  # Exception for closed streams, used in Dask.
from bokeh.util.deprecation import BokehDeprecationWarning  # Suppresses warnings from deprecated Bokeh functionality.

# Warnings and Suppression
import warnings                            # Provides functions to filter or suppress warnings.
from numpy import ComplexWarning           # Represents warnings related to complex numbers in numpy operations.

# General Warning Suppressions
import warnings
from numpy import ComplexWarning
from bokeh.util.deprecation import BokehDeprecationWarning
from tornado.iostream import StreamClosedError
from dask import delayed

################################

DOC_ID = r'.*[\d\w\-.]+\.(html|json)$'  # Regular expression pattern to identify document filenames ending in .html or .json.
#DOC_ID = r'^.*/cleaned/[^/]+\.(html|json)$'

DOC_TYPE = 'json'
DOC_FOLDER = '2010_2014'

TAGS = ['p', 'i']  # List of HTML tags to extract content from; 'p' is commonly used to denote paragraphs in HTML.


##################################
class UnifiedParser(CorpusReader):
       
    def __init__(self, root, tags=TAGS, fileids=DOC_ID, **kwargs):
        CorpusReader.__init__(self, root, fileids)
        self.tags = tags
        # Statistics tracking
        self.error_character_stats = defaultdict(int)
        self.parsing_errors_stats = defaultdict(int)
        self.token_frequency_stats = defaultdict(int)
        self.foreign_sentence_count = 0
        self.mixed_language_sentence_count = 0
        self.paragraph_language_counts = defaultdict(int)
        self.sentence_length_stats = []
        self.paragraph_length_stats = []
        self.unique_tokens = set()
        self.special_character_usage = defaultdict(int)
        self.html_tag_counts = defaultdict(int)

    def resolve(self, fileids=None):
        return fileids 

    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        return chardet.detect(raw_data)['encoding']
    
    def process_content(self, content, non_html_log_file):
        content = unicodedata.normalize('NFKC', content.replace("\n", "\n").strip())

        try:
            soup = BeautifulSoup(content, 'html.parser')
            paragraphs = soup.find_all('p')

            # Track HTML tag counts
            for tag in soup.find_all():
                self.html_tag_counts[tag.name] += 1

            for p in paragraphs:
                text = p.get_text()
                tokenized_paragraph = re.findall(r'\b\w+\b', text)

                # Track token frequencies and unique tokens
                for token in tokenized_paragraph:
                    self.token_frequency_stats[token] += 1
                    self.unique_tokens.add(token)

                # Track control characters in individual tokens
                for token in tokenized_paragraph:
                    for char in token:
                        if (ord(char) < 32 and char not in '\n\t') or unicodedata.category(char) in ['Cc', 'Cf']:
                            self.error_character_stats[(f"U+{ord(char):04X}", unicodedata.name(char, "UNKNOWN"))] += 1

                # Count foreign and mixed-language sentences
                contains_foreign = any(any(ord(char) > 127 for char in token) for token in tokenized_paragraph)
                contains_latin = any(any('LATIN' in unicodedata.name(char, '') for char in token if ord(char) > 127) for token in tokenized_paragraph)

                if contains_foreign:
                    self.foreign_sentence_count += 1

                if contains_foreign and contains_latin:
                    self.mixed_language_sentence_count += 1

                # Determine predominant language in paragraphs
                non_latin_count = sum(1 for token in tokenized_paragraph for char in token if ord(char) > 127)
                latin_count = sum(1 for token in tokenized_paragraph for char in token if 'LATIN' in unicodedata.name(char, '') and ord(char) <= 127)

                if non_latin_count > latin_count:
                    self.paragraph_language_counts['foreign'] += 1
                else:
                    self.paragraph_language_counts['latin'] += 1

                # Track paragraph length
                self.paragraph_length_stats.append(len(tokenized_paragraph))

                # Track special character usage
                for char in text:
                    if not char.isalnum() and char not in (' ', '\n', '\t'):
                        self.special_character_usage[char] += 1

                # Remove all control characters except newline (U+000A) and tab (U+0009)
                cleaned_text = re.sub(r'[^\x09\x0A\x20-\x7E\x80-\uFFFF]', '', text)
                
                # Identify and log any remaining control characters (excluding newline and tab)
                control_chars = [
                    (char, f"U+{ord(char):04X}", unicodedata.name(char, "UNKNOWN"))
                    for char in text if ord(char) < 32 and char not in '\n\t'
                ]

                if control_chars:
                    char_details = "; ".join([f"{c} ({code}: {name})" for c, code, name in control_chars])
                    non_html_log_file.write(f"Control characters found in paragraph: {char_details}\n")
                    
                # Track sentence length statistics
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    self.sentence_length_stats.append(len(re.findall(r'\b\w+\b', sentence)))
                
                yield cleaned_text
        except Exception as e:
            self.parsing_errors_stats[str(e)] += 1
            non_html_log_file.write(f"Error processing as HTML: {str(e)}\n")
            yield None


    def docs(self, fileids=None, pattern=None, non_html_log_file=None):
        fileids = self.resolve(fileids)
        if pattern is not None:
            regex = re.compile(pattern, re.IGNORECASE)
        else:
            regex = re.compile(r'.*([\d\w\-.]+)\.(html|json)$', re.IGNORECASE)
            
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            if regex.search(path):
                encoding = self.detect_encoding(path)
                
                if path.lower().endswith('.json'):
                    with codecs.open(path, 'r', encoding=encoding) as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, str):
                                        for content in self.process_content(item, non_html_log_file):
                                            if content:
                                                yield self.replace_curly_quotes(content)
                            else:
                                print(f"Error: {path} does not contain a list of HTML strings.")
                        except json.JSONDecodeError:
                            print(f"Error: {path} is not a valid JSON file.")
                elif path.lower().endswith('.html'):
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        doc_content = f.read()
                        for content in self.process_content(doc_content, non_html_log_file):
                            if content:
                                yield self.replace_curly_quotes(content)
                else:
                    print(f"Unsupported file type: {path}. Only JSON and HTML files are allowed.")
                    continue


    def html(self, fileids=None):
        """
        Iterates over HTML documents, yielding content for each document.

        Parameters:
            fileids (str or None): Specific file identifier(s) or None.

        Yields:
            str: Parsed content from each HTML document.
        """
        for doc in self.docs(fileids):
            try:
                yield doc
            except Exception as e:
                print("Could not parse HTML: {}".format(e))
                continue

    def get_html_tag_statistics(self):
        """
        Returns the counts of different HTML tags found in the processed documents.

        Returns:
            dict: A dictionary with HTML tag names as keys and their counts as values.
        """
        return dict(self.html_tag_counts)

    def replace_curly_quotes(self, text):
        """
        Replaces curly quotes with straight quotes in the provided text.

        Parameters:
            text (str): The text to process.

        Returns:
            str: The text with curly quotes replaced by straight quotes.
        """
        quote_replacements = {
            u"\u2018": "'",  # Left single quotation mark
            u"\u2019": "'",  # Right single quotation mark
            u"\u201C": '"',  # Left double quotation mark
            u"\u201D": '"',  # Right double quotation mark
        }
        
        for curly_quote, straight_quote in quote_replacements.items():
            text = text.replace(curly_quote, straight_quote)
        
        return text

    def remove_non_printable_chars(self, text):
        """
        Removes non-printable characters from the provided text.

        Parameters:
            text (str): The text to process.

        Returns:
            str: The text with non-printable characters removed.
        """
        non_printable_pattern = re.compile(r'[\x00-\x1F\x7F-\x9F]+')
        cleaned_text = re.sub(non_printable_pattern, '', text)
        return cleaned_text.strip()


    def get_invalid_character_names(self, text):
        """
        Retrieves the names of non-printable characters in the text.

        Parameters:
            text (str): The text to analyze.

        Returns:
            set: A set of names of non-printable characters found in the text.
        """
        char_names = set()
        non_printable_pattern = re.compile(r'[\x00-\x1F\x7F-\x9F]')
        invalid_chars = non_printable_pattern.findall(text)
        for char in invalid_chars:
            try:
                name = unicodedata.name(char)
            except ValueError:
                name = "UNKNOWN CONTROL CHARACTER"
            char_names.add(name)
        return char_names

    def paras(self, parser_type='lxml', fileids=None):
            """
            Extracts paragraphs from HTML content based on specified tags.

            Parameters:
                parser_type (str): Parser type for BeautifulSoup (default is 'lxml').
                fileids (str or None): Specific file identifier(s) or None.

            Yields:
                str: Extracted paragraph text.
            """
            for html in self.html(fileids):
                # Check if html content looks like an HTML string
                if not isinstance(html, str) or "<" not in html:
                    print(f"Skipping non-HTML content: {html}")
                    continue
                
                soup = BeautifulSoup(html, parser_type)
                
                # Join tags into a CSS selector if `self.tags` is a list
                tag_selector = ",".join(self.tags) if isinstance(self.tags, list) else self.tags
                
                for element in soup.select(tag_selector):
                    text = element.text.strip()
                    yield text

    def sents(self, fileids=None):
            """
            Splits paragraphs into sentences.

            Parameters:
                fileids (str or None): Specific file identifier(s) or None.

            Yields:
                str: Extracted sentence text.
            """
            for paragraph in self.paras(fileids=fileids):
                for sentence in nltk.sent_tokenize(paragraph): 
                    yield sentence

    def words(self, fileids=None): 
            """
            Splits sentences into individual words.

            Parameters:
                fileids (str or None): Specific file identifier(s) or None.

            Yields:
                str: Extracted word token.
            """
            for sentence in self.sents(fileids=fileids):
                for token in nltk.wordpunct_tokenize(sentence):
                    yield token
                    
    def validate_paragraph(self, paragraph):
        """
        Validates a paragraph, checking for non-printable characters and whitespace-only content.

        Parameters:
            paragraph (str): The paragraph to validate.

        Returns:
            bool or str: True if valid, otherwise a string explaining the issues.
        """
        reasons = []
        if not paragraph.strip():
            reasons.append("Only whitespace")

        invalid_char_names = self.get_invalid_character_names(paragraph)
        
        if invalid_char_names:
            reason = f"Contains non-printable characters: {', '.join(invalid_char_names)}"
            reasons.append(reason)

        return True if not reasons else ', '.join(reasons)


    def get_token_frequency(self):
        """Returns token frequency statistics."""
        return {token: count for token, count in self.token_frequency_stats.items()}

    def get_error_statistics(self):
        """Return collected error statistics from the wrapped processing."""
        character_stats = {f"{code} ({name})": count for (code, name), count in self.error_character_stats.items()}
        other_error_stats = dict(self.parsing_errors_stats)

        return {
            "character_errors": character_stats,
            "parsing_errors": other_error_stats,
            "foreign_sentence_count": self.foreign_sentence_count,
            "mixed_language_sentence_count": self.mixed_language_sentence_count,
            "paragraph_language_counts": dict(self.paragraph_language_counts),
            "html_tag_counts": self.get_html_tag_statistics()
        }

    def process_all_and_collect_stats(self, content, non_html_log_file):
        """Process entire content and collect stats without changing the original `CorpusReader` methods."""
        list(self.process_content(self, content, non_html_log_file))  # Process content using the wrapper
        # Return error statistics after processing the entire content
        return self.get_error_statistics()
    
    def get_error_statistics(self):
        character_stats = {f"{code} ({name})": count for (code, name), count in self.error_character_stats.items()}
        other_error_stats = dict(self.parsing_errors_stats)
        
        try:
            sentence_length_mean = sum(self.sentence_length_stats) / len(self.sentence_length_stats) if self.sentence_length_stats else float('nan')
        except Exception:
            sentence_length_mean = float('nan')

        try:
            paragraph_length_mean = sum(self.paragraph_length_stats) / len(self.paragraph_length_stats) if self.paragraph_length_stats else float('nan')
        except Exception:
            paragraph_length_mean = float('nan')
        
        try:
            sentence_length_median = statistics.median(self.sentence_length_stats) if self.sentence_length_stats else float('nan')
        except statistics.StatisticsError:
            sentence_length_median = float('nan')

        try:
            paragraph_length_median = statistics.median(self.paragraph_length_stats) if self.paragraph_length_stats else float('nan')
        except statistics.StatisticsError:
            paragraph_length_median = float('nan')

        try:
            sentence_length_mode = statistics.mode(self.sentence_length_stats) if self.sentence_length_stats else float('nan')
        except statistics.StatisticsError:
            sentence_length_mode = float('nan')

        try:
            paragraph_length_mode = statistics.mode(self.paragraph_length_stats) if self.paragraph_length_stats else float('nan')
        except statistics.StatisticsError:
            paragraph_length_mode = float('nan')

        try:
            sentence_length_stdev = statistics.stdev(self.sentence_length_stats) if len(self.sentence_length_stats) > 1 else float('nan')
        except statistics.StatisticsError:
            sentence_length_stdev = float('nan')

        try:
            paragraph_length_stdev = statistics.stdev(self.paragraph_length_stats) if len(self.paragraph_length_stats) > 1 else float('nan')
        except statistics.StatisticsError:
            paragraph_length_stdev = float('nan')

        return {
            "character_errors": character_stats,
            "parsing_errors": other_error_stats,
            "foreign_sentence_count": self.foreign_sentence_count,
            "mixed_language_sentence_count": self.mixed_language_sentence_count,
            "paragraph_language_counts": dict(self.paragraph_language_counts),
            "unique_token_count": len(self.unique_tokens),
            "special_character_usage": dict(self.special_character_usage),
            "sentence_length_mean": sentence_length_mean,
            "paragraph_length_mean": paragraph_length_mean,
            "sentence_length_median": sentence_length_median,
            "paragraph_length_median": paragraph_length_median,
            "sentence_length_mode": sentence_length_mode,
            "paragraph_length_mode": paragraph_length_mode,
            "sentence_length_stdev": sentence_length_stdev,
            "paragraph_length_stdev": paragraph_length_stdev,
            "html_tag_counts": self.get_html_tag_statistics()
        }



    def generate(self, fileids=None, log_file_path=None, non_html_log_path=None):
        """
        Processes documents, logging invalid paragraphs and returning valid ones.

        Parameters:
            fileids (str or None): Specific file identifier(s) or None.
            log_file_path (str): Path for logging invalid paragraphs.
            non_html_log_path (str): Path for logging non-HTML content.

        Returns:
            tuple: A tuple containing:
                - documents (list): List of valid paragraphs.
                - error_dict (list): List of invalid paragraphs.
                - count (int): Number of valid paragraphs added after cleaning.
                - all_paragraph_count (int): Total number of valid paragraphs.
        """
        documents = []
        error_dict = []
        count = 0
        all_paragraph_count = 0 

        # Open two log files: one for invalid paragraphs and one for non-HTML content
        with open(log_file_path, 'a') as invalid_log_file, open(non_html_log_path, 'a') as non_html_log_file:
            # Using tqdm with an unspecified total initially
            progress_bar = tqdm(self.docs(fileids=fileids, non_html_log_file=non_html_log_file), desc="creating corpus", file=sys.stdout)
            
            for idx, html_content in enumerate(progress_bar):
                html_content = self.replace_curly_quotes(html_content)
                validation_result = self.validate_paragraph(html_content)

                if isinstance(validation_result, bool) and validation_result:  
                    # Valid paragraph
                    all_paragraph_count += 1
                    documents.append(html_content)
                else:
                    # Invalid paragraph; log to the invalid paragraphs file
                    if not isinstance(validation_result, bool):
                        invalid_log_file.write(f"Invalid Paragraph {count}: {validation_result}\n")
                        cleaned_html_content = self.remove_non_printable_chars(html_content)

                        if isinstance(self.validate_paragraph(cleaned_html_content), bool):
                            count += 1
                            documents.append(cleaned_html_content)
                        else:
                            error_dict.append(cleaned_html_content)

        return documents, self.get_error_statistics()



##########################################
# Filter out the specific warning message
##########################################
# RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp2")  # Suppress overflow warnings from Gensim (caused by ldamodel.py).
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")  # Suppress divide by zero warnings in Gensim topic coherence calculations.

# ComplexWarnings from numpy (used in pyLDAvis)
warnings.simplefilter('ignore', ComplexWarning)  # Suppress warnings related to complex numbers in numpy operations.

# Bokeh DeprecationWarnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)  # Suppress Bokeh deprecation warnings.

# Tornado StreamClosedError
class StreamClosedWarning(Warning):
    pass
warnings.filterwarnings("ignore", category=StreamClosedWarning)  # Suppress Tornado StreamClosedError warnings.

# Dask and Distributed Specific Warnings
import logging

# Suppress warnings from 'distributed.utils_perf'
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="distributed.utils_perf")

# Suppress DeprecationWarnings from Dask workers
warnings.filterwarnings("ignore", category=DeprecationWarning, module="distributed.worker")

# Set the logging level for distributed and Bokeh to WARNING
logging.getLogger('distributed').setLevel(logging.WARNING)
logging.getLogger("bokeh").setLevel(logging.WARNING)

# Suppress specific SettingWithCopyWarning from pyLDAvis internals
warnings.filterwarnings("ignore", category=Warning, module=r"pyLDAvis\._prepare")  # Suppress warnings caused by attempting to set a value on a copy of a DataFrame slice.

##########################################################################

# Function to split corpus into smaller chunks for parallel processing
def split_corpus(corpus_tuple, n_splits):
    chunk_size = math.ceil(len(corpus_tuple) / n_splits)
    return [corpus_tuple[i:i + chunk_size] for i in range(0, len(corpus_tuple), chunk_size)]

# Function to generate a stream of corpus paragraphs
def corpus_stream_generator(corpus):
    for item in corpus:
        yield item  # Process each item iteratively, no need to hold the entire list in memory.

# Streaming version of `process_corpus`
def process_corpus_streaming(corpus_stream_generator, lemmatize=False, include_stopwords=True):
    texts_out = []
    for paras in corpus_stream_generator:
        inner_text = []
        doc = nlp(paras)
        for token in doc:
            if len(token.text) > 1:
                if include_stopwords or (token.text.lower() not in stop_words and token.lemma_.lower() not in stop_words):
                    inner_text.append(token.lemma_ if lemmatize else token.text)
        if len(inner_text) > 0:
            texts_out.append(inner_text)

    # Move return statement outside the loop
    return texts_out, lemmatize, include_stopwords





if __name__ == "__main__":
        
    import dask
    import logging
    import pandas as pd
    from tqdm import tqdm
    import sys

    # Enable serialization optimizations 
    dask.config.set(scheduler='distributed', serialize=True)
    dask.config.set({'logging.distributed': 'error'})
    dask.config.set({"distributed.scheduler.worker-ttl": '30m'})
    dask.config.set({'distributed.worker.daemon': False})

    #These settings disable automatic spilling but allow for pausing work when 80% of memory is consumed and terminating workers at 99%.
    dask.config.set({'distributed.worker.memory.target': False,
                    'distributed.worker.memory.spill': False,
                    'distributed.worker.memory.pause': 0.8
                    ,'distributed.worker.memory.terminate': 0.99})


    DASK_DIR = r"C:\Temp\max_spill"
    CORES = 12
    MAXIMUM_CORES = 14
    THREADS_PER_CORE = 1
    RAM_MEMORY_LIMIT = "10GB"
    # Configuration for Dask Distributed
    start_time = pd.Timestamp.now()
    formatted_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"START TIME OF PROCESSING THE CORPUS: {formatted_start_time}")

    cluster = LocalCluster(
                n_workers=CORES,
                threads_per_worker=THREADS_PER_CORE,
                processes=True,
                memory_limit="10GB",
                local_directory=DASK_DIR,
                #dashboard_address=None,
                dashboard_address=":8787",
                protocol="tcp",
                death_timeout='1000s',  # Increase timeout before forced kill
        )


    # Create the distributed client
    client = Client(cluster, timeout='1000s')

    # set for adaptive scaling
    client.cluster.adapt(minimum=CORES, maximum=MAXIMUM_CORES)
        
    # Get information about workers from scheduler
    workers_info = client.scheduler_info()["workers"]

    # Iterate over workers and set their memory limits
    for worker_id, worker_info in workers_info.items():
        worker_info["memory_limit"] = "10GB"

    # Check if the Dask client is connected to a scheduler:
    if client.status == "running":
            print("Dask client is connected to a scheduler.")
            # Scatter the embedding vectors across Dask workers
    else:
            print("Dask client is not connected to a scheduler.")
            print("The system is shutting down.")
            client.close()
            cluster.close()
            sys.exit()

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
            print(f"{CORES} Dask workers are running.")
    else:
            client.close()
            cluster.close()
            sys.exit()



    #corpus_path = os.path.join("topic-modeling", "data", "docs-to-process", "PROJECT_FOLDER")
    corpus_path = f"C:/SpectraSync/raw_material/mmwr/{DOC_FOLDER}/cleaned"

    _corpus = UnifiedParser(corpus_path)
    # print filenames
    print("The file being processed:")
    pp.pprint(_corpus.fileids())


    # In[ ]:


    # Define generic paths for log files
    base_path = f"C:/SpectraSync/raw_material/mmwr/{DOC_FOLDER}/log"
    os.makedirs(base_path, exist_ok=True)

    log_file_path = os.path.join(base_path, f"log_error.log")
    non_html_log_path = os.path.join(base_path, f"non_html_content.log")

    # Run the generate function with generic paths
    corpus_tuple, error_statistics = _corpus.generate(
        log_file_path=log_file_path,
        non_html_log_path=non_html_log_path
    )


    pp.pprint(error_statistics)


    # Define generic paths for log files
    base_path = f"C:/SpectraSync/raw_material/mmwr/{DOC_FOLDER}/statistics/"
    os.makedirs(base_path, exist_ok=True)


    #####################
    # ERROR CSV
    #####################
    output_csv_path = f"C:/SpectraSync/raw_material/mmwr/{DOC_FOLDER}/statistics/{DOC_FOLDER}_stats.csv"

    # Convert nested dictionaries to JSON strings
    data_for_csv = {key: (json.dumps(value) if isinstance(value, dict) else value)
                    for key, value in error_statistics.items()}

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        def write_nested_dict(d, parent_key=""):
            for key, value in d.items():
                if isinstance(value, dict):  # Nested dictionary
                    write_nested_dict(value, parent_key=f"{parent_key}{key}.")
                else:
                    writer.writerow([f"{parent_key}{key}", value])
        
        write_nested_dict(error_statistics)

    print(f"Data written to {output_csv_path}")


    ###########################
    # Flatten nested dictionary
    ###########################
    # Write to CSV in RFC 4180 format
    output_csv_path = f"C:/SpectraSync/raw_material/mmwr/{DOC_FOLDER}/statistics/{DOC_FOLDER}_rfc4180_stats.csv"

    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):  # If nested, recursively flatten
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened = flatten_dict(error_statistics)

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=flattened.keys())
        
        # Write header
        writer.writeheader()
        
        # Write row (single row for the flattened data)
        writer.writerow(flattened)

    print(f"Data written in RFC 4180-compliant format to {output_csv_path}")


    #####################
    # WRITE TO JSON
    ####################
    # Output JSON file path
    output_json_path = f"C:/SpectraSync/raw_material/mmwr/{DOC_FOLDER}/statistics/{DOC_FOLDER}_stats.json"

    # Write to JSON file
    with open(output_json_path, mode="w", encoding="utf-8") as file:
        json.dump(error_statistics, file, indent=4)  # Use indent=4 for pretty-printing

    print(f"Data written to {output_json_path}")



    ########################
    # BEGIN PROCESSING
    ########################
    texts_out_lemmatized_with_stopwords = []
    texts_out_lemmatized_without_stopwords = []
    texts_out_not_lemmatized_with_stopwords = []
    texts_out_not_lemmatized_without_stopwords = []

    # Split the corpus into smaller chunks
    corpus_chunks = split_corpus(corpus_tuple, 14)

    # Split the corpus into smaller chunks
    corpus_chunks = split_corpus(corpus_tuple, 14)

    # Create delayed objects for each chunk and each configuration
    delayed_futures = []
    for chunk in tqdm(corpus_chunks, total=len(corpus_chunks), desc="Processing Delayed Objects", file=sys.stdout):
        delayed_futures.append(delayed(process_corpus_streaming)(chunk, True, True))
        delayed_futures.append(delayed(process_corpus_streaming)(chunk, True, False))
        delayed_futures.append(delayed(process_corpus_streaming)(chunk, False, True))
        delayed_futures.append(delayed(process_corpus_streaming)(chunk, False, False))

    # Compute delayed futures in manageable batches with tqdm progress bar
    batch_size = 20  # Define an appropriate batch size based on available memory

    # Using tqdm to display progress for the batch computations
    for i in tqdm(range(0, len(delayed_futures), batch_size), desc="Processing batches with Dask", file=sys.stdout):
        batch = delayed_futures[i:i + batch_size]
        results = dask.compute(*batch)
        for result in results:
            sentences, is_lemmatized, has_stopwords = result
            if is_lemmatized and has_stopwords:
                texts_out_lemmatized_with_stopwords.extend(sentences)
            elif is_lemmatized and not has_stopwords:
                texts_out_lemmatized_without_stopwords.extend(sentences)
            elif not is_lemmatized and has_stopwords:
                texts_out_not_lemmatized_with_stopwords.extend(sentences)
            elif not is_lemmatized and not has_stopwords:
                texts_out_not_lemmatized_without_stopwords.extend(sentences)

    
    print("writing texts_out_not_lemmatized_without_stopwords.json")
    # Define the file path for saving processed text in JSON format
    filename2 = f"C:/SpectraSync/processed_material/mmwr/{DOC_FOLDER}/texts_out_lemmatized_with_stopwords.json"
    # Open the specified file in write mode
    with open(filename2, 'w', encoding='utf-8') as jsonfile:
        # Write the processed text data to the JSON file with formatting and UTF-8 encoding
        json.dump(texts_out_lemmatized_with_stopwords, jsonfile, indent=4, ensure_ascii=False)

    print("writing texts_out_lemmatized_without_stopwords.json")
    # Define the file path for saving processed text in JSON format
    filename2 = f"C:/SpectraSync/processed_material/mmwr/{DOC_FOLDER}/texts_out_lemmatized_without_stopwords.json"
    # Open the specified file in write mode
    with open(filename2, 'w', encoding='utf-8') as jsonfile:
        # Write the processed text data to the JSON file with formatting and UTF-8 encoding
        json.dump(texts_out_lemmatized_without_stopwords, jsonfile, indent=4, ensure_ascii=False)

    print("writing texts_out_not_lemmatized_with_stopwords.json ")
    # Define the file path for saving processed text in JSON format
    filename2 = f"C:/SpectraSync/processed_material/mmwr/{DOC_FOLDER}/texts_out_not_lemmatized_with_stopwords.json"
    # Open the specified file in write mode
    with open(filename2, 'w', encoding='utf-8') as jsonfile:
        # Write the processed text data to the JSON file with formatting and UTF-8 encoding
        json.dump(texts_out_not_lemmatized_with_stopwords, jsonfile, indent=4, ensure_ascii=False)

    print("writing texts_out_not_lemmatized_without_stopwords.json")
    # Define the file path for saving processed text in JSON format
    filename2 = f"C:/SpectraSync/processed_material/mmwr/{DOC_FOLDER}/texts_out_not_lemmatized_without_stopwords.json"
    # Open the specified file in write mode
    with open(filename2, 'w', encoding='utf-8') as jsonfile:
        # Write the processed text data to the JSON file with formatting and UTF-8 encoding
        json.dump(texts_out_not_lemmatized_without_stopwords, jsonfile, indent=4, ensure_ascii=False)


    # Capture the end time and log it
    end_time = pd.Timestamp.now()
    formatted_end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"END TIME OF PROCESSING THE CORPUS: {formatted_end_time}")

    # Calculate and log the duration
    duration = end_time - start_time
    logging.info(f"DURATION OF PROCESSING: {duration}")

    # Shut down the client and cluster
    client.close()
    cluster.close()