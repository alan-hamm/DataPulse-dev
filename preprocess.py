

#%%
from bs4 import BeautifulSoup
import os
import pprint as pp
import re
from collections import defaultdict
import json
from tqdm import tqdm
import dask
from dask import delayed, compute
from collections import defaultdict
from bs4 import BeautifulSoup
import re
import sys
from dask.distributed import Client, LocalCluster
from more_itertools import chunked


#%%

# Function to scrape and extract paragraphs with tags
def scrape_paragraphs_with_tags(file_path, regex_patterns):
    html_tag_counts = defaultdict(int)
    try:
        # Open the local HTML file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all paragraph tags in the HTML
        paragraphs = soup.find_all('p')

        # Allowed nested tags
        allowed_nested_tags = {'i'}

        # Extract paragraphs that have only allowed nested tags
        extracted_paragraphs = []
        
        for paragraph in tqdm(paragraphs, total=len(paragraphs), desc="Processing paragraphs", file=sys.stdout):
            # Count all attributes of the paragraph tag
            if paragraph is None:
                continue

            # Count all attributes of the paragraph tag
            if not paragraph.attrs:  # Check if paragraph.attrs is not None
                continue
            
            for key in list(paragraph.attrs.keys()):
                if any(pattern.match(key) for pattern in regex_patterns):
                    html_tag_counts[key] += 1
                    paragraph.decompose()  # Remove the paragraph entirely
                    break  # No need to check further attributes for this paragraph

            # Skip further processing if the paragraph was decomposed
            if not paragraph.parent:
                continue
            
            # Count all nested tags inside the paragraph
            nested_tags = paragraph.find_all()
            for tag in nested_tags:
                html_tag_counts[tag.name] += 1  # Count the tag name (e.g., <b>, <i>)

            # Use a generator to incrementally extract text characters
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
                if valid and length >= 3:
                    break  # Stop further processing once conditions are met

            if not (valid and length >= 3):
                continue
            
            # Check if the paragraph has only allowed nested tags
            if all(tag.name in allowed_nested_tags for tag in nested_tags):
                # Convert the paragraph element back to a string including the tags
                paragraph_with_tags = str(paragraph)
                extracted_paragraphs.append(paragraph_with_tags)

        return extracted_paragraphs, html_tag_counts
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return [], html_tag_counts
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], html_tag_counts



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

    CLEANED = "C:/SpectraSync/raw_material/mmwr/2010_2014/cleaned/"

    TITLE_LIST = parse_directory_for_files("C:/SpectraSync/raw_material/mmwr/2010_2014/pre-processed", ".json")
    pp.pprint(TITLE_LIST)
    for file_path in TITLE_LIST:
        extracted_paragraphs, html_tag_counts = scrape_paragraphs_with_tags(file_path, regex_attr_to_remove)

        # Write the extracted paragraphs to a new file
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Get the filename without the extension
        output_file_name = os.path.join(CLEANED,f"filtered_{base_name}.html")
        with open(output_file_name, 'w', encoding='utf-8') as output_file:
            for paragraph in extracted_paragraphs:
                output_file.write(paragraph + '\n\n')  # Add a blank line between paragraphs for readability
        
        filename = f"C:/SpectraSync/raw_material/mmwr/2010_2014/log/{base_name}_html_stats.json"
        # Open the specified file in write mode
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            # Write the processed text data to the JSON file with formatting and UTF-8 encoding
            json.dump(dict(html_tag_counts), jsonfile, indent=4, ensure_ascii=False)

    
if __name__ == "__main__":


    # Main execution with exception handling
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Processing complete.")

