import os
import pandas as pd
import spacy
import tarfile
from bs4 import BeautifulSoup
from datetime import datetime
import re

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10000000

# Constants
CSV_FILE = "rating/screenshot_urls.csv"  # Path to your CSV file containing the URLs and TAR paths
OUTPUT_FILE = "rating/warc_extracted_content.csv"  # Output file to store the extracted data

# Function to preprocess and clean the text before extracting POS tags
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())  # Remove extra spaces and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to extract POS tags using spaCy
def extract_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc if token.is_alpha]  # Extract POS tags for alphabetic tokens
    return " ".join(pos_tags)  # Return as a space-separated string of POS tags

# Function to extract text from a specific TAR file path
def extract_text_from_tar(tar_path, query_path):
    extracted_text = ""
    try:
        with tarfile.open(tar_path, "r") as tar:
            # Extract the HTML content using the query path within the TAR file
            html_file = tar.extractfile(query_path)
            if html_file:
                html_content = html_file.read().decode("utf-8", errors="ignore")
                # Parse HTML content and extract plain text
                soup = BeautifulSoup(html_content, 'html.parser')
                extracted_text = soup.get_text()
    except Exception as e:
        print(f"Error extracting text from TAR file {tar_path}: {e}")
    return extracted_text

# Function to process each row in the CSV file and extract relevant data
def process_csv_and_extract_data(csv_file):
    # Load the CSV file containing URLs and TAR paths
    data = pd.read_csv(csv_file)

    # Prepare a list to store the results
    results = []

    # Iterate over each row in the CSV
    for index, row in data.iterrows():
        screenshot = row['Screenshot']
        url = row['URL']
        tar_path = row['TAR File Path']
        query_path = row['Query Path']

        print(f"Processing URL: {url}")

        # Extract text from the TAR file
        text = extract_text_from_tar(tar_path, query_path)

        # Preprocess the text and extract POS tags
        processed_text = preprocess_text(text)
        pos_tags = extract_pos_tags(processed_text)

        # Store the results for this row
        results.append({
            'Screenshot': screenshot,
            'URL': url,
            'Text': processed_text,
            'POS_Tags': pos_tags
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data extracted and saved to {OUTPUT_FILE}")

# Run the function to process the CSV and extract data
process_csv_and_extract_data(CSV_FILE)



# import pandas as pd

# # Load the two CSV files
# file1 = "rating/ground_truth_spam_with_url.csv"  # Replace with your first file path
# file2 = "rating/screenshot_urls.csv"  # Replace with your second file path

# # Read the CSV files into pandas DataFrames
# df1 = pd.read_csv(file1)
# df2 = pd.read_csv(file2)

# # Assuming the column name containing the URLs is 'url' in both CSV files
# # You can change this if the column name is different

# # Get the set of URLs from both files
# urls_file1 = set(df1['URL'].str.strip().str.lower())  # Normalize by stripping and lowering case
# urls_file2 = set(df2['URL'].str.strip().str.lower())

# # Find missing URLs from file1 that are not in file2
# missing_urls = urls_file1 - urls_file2

# # If you want to print the missing URLs:
# print(f"Missing URLs from {file1}:")
# for url in missing_urls:
#     print(url)

# # If you want to save the missing URLs to a CSV file:
# missing_urls_df = pd.DataFrame(list(missing_urls), columns=["Missing URLs"])
# missing_urls_df.to_csv("missing_urls.csv", index=False)

