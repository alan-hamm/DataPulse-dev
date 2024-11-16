
#%%
import csv
from collections import defaultdict

# Input CSV file
input_csv_path = r"C:\utma\data\GrandUnifiedProject\statistics\2024_stats (2).csv"

# Function to build a nested dictionary
def nested_dict():
    return defaultdict(nested_dict)

# Parse CSV into nested dictionary
data = nested_dict()

with open(input_csv_path, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) < 2:  # Skip invalid rows
            continue
        key, value = row[0], row[1]
        keys = key.split(".")  # Split hierarchical keys
        current = data
        for k in keys[:-1]:  # Traverse to the second-to-last key
            current = current[k]
        current[keys[-1]] = float(value) if value.replace(".", "", 1).isdigit() else value

# Convert defaultdict to regular dict
def convert_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_dict(v) for k, v in d.items()}
    return d

nested_data = convert_dict(data)

print(nested_data)
