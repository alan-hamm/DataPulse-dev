

#%%
# Standard library imports
import os
import sys
import json
import re
import threading
from time import time
from collections import defaultdict
import statistics
import unicodedata  # Handles Unicode character information, useful for identifying and removing non-printable characters.

# Third-party library imports
from bs4 import BeautifulSoup  # Parses HTML and XML documents
import chardet  # Detects character encoding of text files
import pandas as pd  # Data manipulation and analysis
import psutil  # System and process utilities
from tqdm import tqdm  # Progress bar for loops

# Dask imports (for parallel and distributed computing)
import dask  # Parallel computing library for handling large computations across multiple cores
from dask import delayed, compute
from dask.distributed import get_client, Client, LocalCluster, performance_report, wait
from distributed import Future  # Represents a future result from a Dask computation

# Miscellaneous imports
import pprint as pp  # Pretty printing for debugging or logging


#%%

class HTMLParserStatistics:
    """
    A class to collect and manage statistics during HTML parsing.

    This class tracks various statistics, such as token frequencies, 
    paragraph lengths, special character usage, HTML tag counts, and 
    parsing errors. It provides methods to retrieve these statistics 
    in structured formats.

    Attributes:
        error_character_stats (defaultdict): Counts of invalid or error characters encountered.
        parsing_errors_stats (defaultdict): Counts of parsing errors by error type.
        token_frequency_stats (defaultdict): Frequency counts of individual tokens.
        foreign_sentence_count (int): Number of sentences in foreign languages.
        mixed_language_sentence_count (int): Number of sentences containing multiple languages.
        paragraph_language_counts (defaultdict): Counts of paragraphs by detected language.
        sentence_length_stats (list): List of sentence lengths for statistical analysis.
        paragraph_length_stats (list): List of paragraph lengths for statistical analysis.
        unique_tokens (set): A set of unique tokens encountered.
        special_character_usage (defaultdict): Counts of special characters encountered.
        html_tag_counts (defaultdict): Counts of encountered HTML tags.

    Methods:
        get_html_tag_statistics():
            Returns a dictionary representation of HTML tag counts.

        get_error_statistics():
            Computes and returns a structured dictionary of all collected statistics,
            including averages, medians, modes, and standard deviations for sentence 
            and paragraph lengths.
    """
    def __init__(self):
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

    def get_html_tag_statistics(self):
        return dict(self.html_tag_counts)

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
            "html_tag_counts": self.get_html_tag_statistics(),
        }


def parse_with_simulated_progress(file_path):
    """
    Parses an HTML file with a progress bar.

    This function reads the contents of an HTML file and uses `BeautifulSoup` 
    to parse it into a structured format. During the parsing process, a progress 
    bar is displayed to indicate the size of the file being processed.

    Args:
        file_path (str): The path to the HTML file to be parsed.

    Returns:
        BeautifulSoup: A `BeautifulSoup` object representing the parsed HTML content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an issue reading the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    total_size = len(html_content)

    with tqdm(total=total_size, desc="Parsing HTML", unit="chars", unit_scale=True) as pbar:
        soup = BeautifulSoup(html_content, "html.parser")
        pbar.update(total_size)

    return soup

# Function to scrape and extract paragraphs with tags
def scrape_paragraphs_with_tags(file_path, html_data, regex_patterns):
    """
    Extracts and processes HTML paragraphs from a file with parallelization support.

    This method uses `BeautifulSoup` to parse the HTML content of a given file,
    processes its `<p>` tags to collect statistical data, cleans undesired nested
    tags, and returns the processed paragraphs along with statistics. The method
    is decorated with `@delayed` to allow integration with Dask for parallelized
    execution.

    Args:
        file_path (str): The path to the HTML file to be processed.
        regex_patterns (list): A list of compiled regex patterns to identify
            undesired attributes for removal.

    Returns:
        tuple: A tuple containing:
            - `HTMLParserStatistics`: An object with accumulated statistics
              about the HTML content.
            - `list`: A list of cleaned paragraphs as strings.

    Raises:
        ValueError: If no content remains in the cleaned paragraphs after processing.
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an issue reading the file.

    Notes:
        - The `@delayed` decorator allows this function to be scheduled for
          execution in a Dask workflow.
        - This function logs errors encountered during processing and
          continues with subsequent paragraphs.
    """

    stats = HTMLParserStatistics()
    cleaned_paragraphs = []
    try:
        #with open(file_path, "r", encoding="utf-8") as file:
        #    html_content = file.read()
        html_content = dask.compute(html_data)
        if not isinstance(html_content, tuple):
            raise ValueError("The scattered HTML was not unpacked.")
        
        #print("This is the first part of HTML JSON...")
        #print(html_content[0][:100])

        soup = BeautifulSoup(html_content[0], "html.parser")
        paragraphs = soup.find_all("p")

        print("Parsing BeautifulSoup Paragraph Object")
        for paragraph in tqdm(paragraphs, total=len(paragraphs), desc=f"Processing: {file_path}", file=sys.stdout):
            if paragraph is None:
                print(f"Skipping NoneType paragraph in {file_path}")
                continue

            stats.html_tag_counts["p"] += 1  # Count the paragraph tag

            try:
                text = paragraph.get_text()
                tokenized_paragraph = re.findall(r"\b\w+\b", text)
            except Exception as e:
                print(f"Failed to extract or tokenize paragraph text: {e}")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Track token frequencies and unique tokens
            try:
                for token in tokenized_paragraph:
                    stats.token_frequency_stats[token] += 1
                    stats.unique_tokens.add(token)
            except Exception as e:
                print("Failed to track token frequencies and unique counts")
                stats.parsing_errors_stats[str(e)] += 1
                continue               

            # Collect character stats
            try:
                for token in tokenized_paragraph:
                    for char in token:
                        if (ord(char) < 32 and char not in "\n\t") or unicodedata.category(char) in ["Cc", "Cf"]:
                            stats.error_character_stats[(f"U+{ord(char):04X}", unicodedata.name(char, "UNKNOWN"))] += 1
            except Exception as e:
                print("Failed to collect character stats")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Collect special character usage
            try:
                for char in text:
                    if not char.isalnum() and char not in " \n\t":
                        stats.special_character_usage[char] += 1
            except Exception as e:
                print("Failed to collect special character usage")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Collect paragraph length stats
            try:
                stats.paragraph_length_stats.append(len(tokenized_paragraph))
            except Exception as e:
                print("Failed to calculate paragraph length")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Count nested tags
            try:
                nested_tags = paragraph.find_all()
                for tag in nested_tags:
                    stats.html_tag_counts[tag.name] += 1
            except Exception as e:
                print("Failed to count nested tags")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Remove paragraph if necessary
            try:
                # Ensure paragraph.attrs is not None before accessing .keys()
                if paragraph.attrs:
                    for key in list(paragraph.attrs.keys()):
                        if any(pattern.match(key) for pattern in regex_patterns):
                            stats.html_tag_counts[key] += 1
                            paragraph.decompose()
                            break
            except Exception as e:
                print(f"Error processing attributes: {e}")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Skip to the next paragraph if this one was decomposed
            if not paragraph.parent:
                continue
                
            try:
                # Clean undesired nested tags within the paragraph
                nested_tags = paragraph.find_all()
                for tag in nested_tags:
                    stats.html_tag_counts[tag.name] += 1
                    if tag.name not in ['i']:  # Replace with desired tags to keep
                        tag.decompose()
            except Exception as e:
                print(f"Failed to clean undesired nested tags within the paragraph: {e}")
                stats.parsing_errors_stats[str(e)] += 1
                continue

            # Use a generator to incrementally extract text characters
            try:
                paragraph_text = paragraph.stripped_strings  # Generator for stripped text pieces
                valid = False
                length = 0

                for chunk in paragraph_text:
                    for char in chunk:
                        if char.isalpha():
                            valid = True
                        length += 1
                    if valid and length >= 3:
                        break  # Stop further processing once conditions are met

                if not (valid and length >= 3):
                    continue
            except Exception as e:
                print("Failed to check for valid length and presence of alpha-characters")
            

            try:
                cleaned_paragraphs.append(str(paragraph))  # Save processed paragraph content
            except Exception as e:
                print(f"Error appending cleaned paragraph: {e}")
                stats.parsing_errors_stats[str(e)] += 1
                continue

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        stats.parsing_errors_stats[str(e)] += 1

    print(f"{file_path}: cleaned paragraphs:: {len(cleaned_paragraphs)}")
    if len(cleaned_paragraphs) == 0:
         raise ValueError("The cleaned paragraphs contained no content.")
    
    return stats, cleaned_paragraphs, file_path



def save_statistics(stats, output_path):
    error_statistics = stats.get_error_statistics()

    with open(output_path, mode="w", encoding="utf-8") as file:
        json.dump(error_statistics, file, indent=4)

    print(f"Data written to {output_path}")


# Function to parse a directory for specific file types
def parse_directory_for_files(directory_path, file_extension):
    matching_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                matching_files.append(os.path.join(root, file))
    return matching_files

def main():

    # Call your scraping function
    regex_attr_to_remove = [
        re.compile(r'^accept.*'), re.compile(r'access.*'), re.compile(r'action.*'), 
        re.compile(r'align.*'), re.compile(r'alt.*'), re.compile(r'async.*'), re.compile(r'auto.*'),
        re.compile(r'bg.*'), re.compile(r'border.*'),
        re.compile(r'charset.*'), re.compile(r'cite.*'), re.compile(r'class.*'), re.compile(r'color.*'), re.compile(r'col.*'), 
        re.compile(r'content.*'), re.compile(r'control.*'), re.compile(r'coords.*'),
        re.compile(r'data.*'), re.compile(r'date.*'), re.compile(r'disabled.*'), re.compile(r'download.*'), re.compile(r'draggable.*'),
        re.compile(r'enctype.*'), re.compile(r'enter.*'),
        re.compile(r'for.*'),
        re.compile(r'height.*'), re.compile(r'hidden.*'), re.compile(r'high.*'), re.compile(r'href.*'), re.compile(r'https.*'),
        re.compile(r'id.*'), re.compile(r'inert.*'), re.compile(r'input.*'), re.compile(r'ismap.*'),
        re.compile(r'kind.*'),
        re.compile(r'label.*'), re.compile(r'lang.*'), re.compile(r'list.*'), re.compile(r'loop.*'), re.compile(r'low.*'),
        re.compile(r'max.*'), re.compile(r'media.*'), re.compile(r'method.*'), re.compile(r'min.*'), re.compile(r'multiple.*'), re.compile(r'muted.*'),
        re.compile(r'name.*'), re.compile(r'novalidate.*'),
        re.compile(r'on.*'), re.compile(r'open.*'), re.compile(r'optimum.*'),
        re.compile(r'pattern.*'), re.compile(r'place.*'), re.compile(r'pop.*'), re.compile(r'poster.*'), re.compile(r'pre.*'),
        re.compile(r'read.*'), re.compile(r'rel.*'), re.compile(r'required.*'), re.compile(r'reverse.*'), re.compile(r'row.*'),
        re.compile(r'sand.*'), re.compile(r'scope.*'), re.compile(r'select.*'), re.compile(r'shape.*'), re.compile(r'size.*'), re.compile(r'span.*'),
        re.compile(r'spell.*'), re.compile(r'src.*'), re.compile(r'start.*'), re.compile(r'step.*'), re.compile(r'style.*'),
        re.compile(r'tab.*'), re.compile(r'target.*'), re.compile(r'translate.*'), re.compile(r'type.*'), 
        re.compile(r'usemap.*'),
        re.compile(r'value.*'),
        re.compile(r'width.*'),
        re.compile(r'wrap.*')
    ]

    CLEANED = os.path.join("C:\\", "SpectraSync", "raw_material", "mmwr", "2020_2024", "cleaned")
    STATS_DIR = os.path.join("C:\\", "SpectraSync", "raw_material", "mmwr", "2020_2024", "statistics")

    TITLE_LIST = parse_directory_for_files("C:/SpectraSync/raw_material/mmwr/2020_2024/pre-processed", ".json")
    pp.pprint(TITLE_LIST)

    futures = []
    for file_path in TITLE_LIST:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        scattered = client.scatter(html_content)
        try:
            future = client.submit( scrape_paragraphs_with_tags, file_path, scattered, regex_attr_to_remove)
            futures.append(future)
        except Exception as e:
            print(f"Error scheduling task for {file_path}: {e}")
            continue

    print("Computing futures objects...")
    completed_futures, not_done = wait(futures)

    results = [done.result() for done in completed_futures]
    
    invalid_results = [result for result in results if not (isinstance(result, tuple) and len(result) == 3)]
    if invalid_results:
        print(f"Skipping {len(invalid_results)} invalid results: {invalid_results}")


    #for stats, paragraphs, file_path in zip(computed_stats, cleaned_paragraphs_list, TITLE_LIST):
    for result in results:
        try:
            stats, paragraphs, file_path = result  # Unpack the tuple directly from each resolved future result

            if not isinstance(paragraphs, list):
                raise ValueError(f"Paragraphs are not a list for file {file_path}. Value: {paragraphs}")

            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Save cleaned paragraphs
            output_file_name = os.path.join(CLEANED, f"filtered_{base_name}.html")
            with open(output_file_name, 'w', encoding='utf-8') as output_file:
                for paragraph in paragraphs:
                    if not isinstance(paragraph, str):
                        raise ValueError(f"Paragraph is not a string for file {file_path}. Value: {paragraph}")
                    output_file.write(paragraph + '\n\n')

            # Save statistics
            stats_file = os.path.join(STATS_DIR, f"{base_name}_statistics.json")
            save_statistics(stats, stats_file)
            client.cancel(result)

            print("Rebalancing tasks across workers...")
            client.rebalance()
        except Exception as e:
            print(f"There was an error in writing the paragraph to file: {e}")


if __name__ == "__main__":
    start_time = pd.Timestamp.now()
    cluster = LocalCluster(
            n_workers=8,
            threads_per_worker=2,
            processes=True,
            memory_limit="18GB",
            local_directory=r"C:\Temp\dask",
            dashboard_address=":8787",
            protocol="tcp",
            death_timeout='1000s',  # Increase timeout before forced kill
    )


    # Create the distributed client
    client = Client(cluster, timeout='1000s')

    # set for adaptive scaling
    client.cluster.adapt(minimum=9, maximum=11)
    
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
        print(f"12 Dask workers are running.")
    else:
        print("No Dask workers are running.")
        print("The system is shutting down.")
        client.close()
        cluster.close()
        sys.exit()

    try:
        with performance_report(filename=r"C:\SpectraSync\raw_material\statistics\2020_2024_dask_report.html"):
            main()
    except Exception as e:
        print(f"There was an error in the main method: {e}")
    finally:
        client.close()
        cluster.close()

