# How It Works
# PDF Extraction:
# The script reads your PDF file starting from page index 1 (skipping the first page) and concatenates the text.

# Parsing Dates and Rows:

# It uses a regex to find date markers (e.g. “4th December 2024”).
# Then, for each date block, it extracts rows using a regex pattern that expects a half-hour period, a rate, consumption, and cost.
# Data Consolidation:
# All rows are stored in a list and then converted into a pandas DataFrame.

# Filtering:
# A helper function checks if the starting time of the period is within the off-peak window (before 05:30 or from 23:30 onward). The DataFrame is filtered to retain only those rows where the rate is 6.67p and the period is not off-peak.

# You can adjust the regex patterns if the formatting in your PDF differs slightly. This should consolidate your data and produce the desired filtered view.

import re
from datetime import datetime, timedelta
import pandas as pd
import PyPDF2
import os
import glob
import argparse

# Define a helper function to remove ordinal suffixes from day numbers in dates
def remove_ordinal_suffix(s):
    return re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', s)

# Define a regex pattern to capture dates in the format like "4th December 2024"
date_pattern = r"(\d{1,2}(?:st|nd|rd|th)\s+\w+\s+\d{4})"

# Define a regex to capture each half-hour row.
# This pattern assumes rows are formatted like:
# "00:00-00:30   6.67   7.36   49.042"
row_pattern = re.compile(r"(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process Octopus Energy bills from PDF files.')
parser.add_argument('-iog', '--iog', action='store_true', default=False,
                    help='Add IOG Cheap Rate column (true when time is between 23:30 and 05:00)')
parser.add_argument('--filter-rate', type=float, default=6.67,
                    help='Rate (in pence) to filter for bonus slots (default: 6.67)')
parser.add_argument('--quiet', '-q', action='store_true', default=False,
                    help='Suppress DataFrame preview output')
args = parser.parse_args()

# Time constants for performance (parsed once instead of on every function call)
OFFPEAK_START_TIME = datetime.strptime("23:30", "%H:%M").time()
OFFPEAK_END_TIME = datetime.strptime("05:30", "%H:%M").time()

# Helper functions
def create_timestamps(row):
    """Create ISO timestamps (Start and End) from date and period"""
    try:
        # Extract start and end times from period
        period_parts = row['Period'].replace('\n', ' ').split('-')
        if len(period_parts) != 2:
            raise ValueError(f"Invalid period format: {row['Period']}")

        start_time_str = period_parts[0].strip()
        end_time_str = period_parts[1].strip()

        # Create start datetime
        start_dt = datetime.combine(row['Date'], datetime.strptime(start_time_str, '%H:%M').time())

        # Create end datetime - handle midnight crossing
        start_hour = int(start_time_str.split(':')[0])
        end_hour = int(end_time_str.split(':')[0])

        if end_hour < start_hour:  # End time is on the next day (e.g., 23:30-00:00)
            end_date = row['Date'] + timedelta(days=1)
        else:
            end_date = row['Date']

        end_dt = datetime.combine(end_date, datetime.strptime(end_time_str, '%H:%M').time())

        return pd.Series([start_dt.strftime('%Y-%m-%dT%H:%M:00'), end_dt.strftime('%Y-%m-%dT%H:%M:00')])
    except Exception as e:
        print(f"Error creating timestamps for row: {row['Period']} on {row['Date']}: {e}")
        return pd.Series([None, None])

def is_offpeak(period_str):
    """Determine if the period starts during the off-peak window (23:30 - 05:30)"""
    start_str = period_str.replace('\n', ' ').split('-')[0].strip()
    start_time = datetime.strptime(start_str, "%H:%M").time()
    # Because the off-peak window spans midnight,
    # we consider times from 23:30 to midnight OR from midnight to 05:30 as off-peak.
    return (start_time >= OFFPEAK_START_TIME) or (start_time < OFFPEAK_END_TIME)

def process_pdf(pdf_path):
    """Process a single PDF file and return its data rows"""
    data_rows = []

    # Use context manager to ensure file is properly closed
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Use list and join for better performance instead of string concatenation
        text_parts = []
        for i in range(1, len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        all_text = "\n".join(text_parts)

    # Find all date markers
    date_matches = list(re.finditer(date_pattern, all_text))
    
    # Process each date block
    for idx, date_match in enumerate(date_matches):
        date_str = date_match.group(1)
        date_clean = remove_ordinal_suffix(date_str)

        # Try multiple date formats to handle variations in PDF formatting
        date_parsed = None
        date_formats = [
            "%d %B %Y",  # Full month name: "19 December 2024"
            "%d %b %Y",  # Abbreviated month name: "19 Dec 2024"
        ]

        for date_format in date_formats:
            try:
                date_parsed = datetime.strptime(date_clean, date_format).date()
                break  # Successfully parsed, exit the loop
            except ValueError:
                continue  # Try next format

        if date_parsed is None:
            print(f"Error parsing date '{date_clean}' in {pdf_path}: No matching format found")
            continue

        start_index = date_match.end()
        end_index = date_matches[idx + 1].start() if idx + 1 < len(date_matches) else len(all_text)
        block = all_text[start_index:end_index]
        
        for row in row_pattern.findall(block):
            period, rate, consumption, cost = row
            data_rows.append({
                "Date": date_parsed,
                "Period": period.strip().replace('\n', ' '),
                "Rate": float(rate),
                "Consumption": float(consumption),
                "Cost": float(cost)
            })
    
    return data_rows

# Get all PDF files in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_files = glob.glob(os.path.join(script_dir, "*.pdf"))

if not pdf_files:
    print("No PDF files found in directory:", script_dir)
    exit(1)

# Process all PDFs and collect their data
all_data_rows = []
for pdf_path in pdf_files:
    print(f"Processing: {pdf_path}")
    all_data_rows.extend(process_pdf(pdf_path))

# Check if any data was extracted
if not all_data_rows:
    print("\nNo data rows extracted from any PDF files. Please check:")
    print("- PDF format matches expected structure")
    print("- PDFs contain the energy usage tables")
    exit(1)

# After creating the DataFrame, set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Create a DataFrame with all the consolidated data
df = pd.DataFrame(all_data_rows)

# Add Start and End timestamp columns
df[['Start', 'End']] = df.apply(create_timestamps, axis=1)

# Add IOG Cheap Rate column if flag is enabled
if args.iog:
    df['IOG Cheap Rate'] = df['Period'].apply(is_offpeak)
    df = df[['Start', 'End', 'Date', 'Period', 'Rate', 'Consumption', 'Cost', 'IOG Cheap Rate']]
else:
    df = df[['Start', 'End', 'Date', 'Period', 'Rate', 'Consumption', 'Cost']]

# Sort by Start timestamp to ensure chronological order
df = df.sort_values('Start')

# Filter the DataFrame:
# Select rows where Rate matches the specified rate and the period is NOT off-peak.
df_filtered = df[(df['Rate'] == args.filter_rate) & (~df['Period'].apply(is_offpeak))]

# Export both DataFrames to CSV with absolute paths and debug output
# Create full paths for CSV files
all_data_path = os.path.join(script_dir, 'all_energy_data.csv')
filtered_data_path = os.path.join(script_dir, 'filtered_energy_data.csv')

# Export with error handling
try:
    df.to_csv(all_data_path, index=False)
    print(f"\nAll data exported to: {all_data_path}")
except Exception as e:
    print(f"Error saving all_energy_data.csv: {e}")

try:
    df_filtered.to_csv(filtered_data_path, index=False)
    print(f"Filtered data exported to: {filtered_data_path}")
except Exception as e:
    print(f"Error saving filtered_energy_data.csv: {e}")

# Show results with improved formatting (unless quiet mode is enabled)
if not args.quiet:
    print("\nConsolidated DataFrame (first few rows):")
    print(df.head().to_string(index=False))
    print(f"\nFiltered DataFrame (Rate = {args.filter_rate}p outside off-peak 23:30-05:30):")
    print(df_filtered.to_string(index=False))
