import pandas as pd
import re

# Read CSV file into DataFrame
df = pd.read_csv('screenshot_urls.csv')  # Replace 'your_file.csv' with the actual file path

# Regex patterns
date_pattern = r'(\d{4}-\d{2}-\d{2})'  # Pattern to extract dates
query_pattern = r'query-(\d+)'  # Pattern to extract query number

# Function to extract date and query number from TAR and Query paths
def extract_date_and_query(row):
    # Extract date from TAR File Path
    date_match = re.search(date_pattern, row['TAR File Path'])
    date = date_match.group(1) if date_match else None

    # Extract query number from Query Path
    query_match = re.search(query_pattern, row['Query Path'])
    query = query_match.group(1) if query_match else None
    
    return pd.Series([date, query])

# Apply the function to extract date and query number
df[['Date', 'Query Number']] = df.apply(extract_date_and_query, axis=1)

# Check for repeated Date and Query combinations
repeated_combinations = df[df.duplicated(subset=['Date', 'Query Number'], keep=False)]

# Display repeated combinations
print("Repeated Date and Query Combinations:")
print(repeated_combinations)




# import os

# screenshots_output_dir = 'screenshots'  # Directory where screenshots are stored

# def count_extracted_files(output_dir):
#     """Count the number of extracted PNG screenshot files."""
#     if not os.path.exists(output_dir):
#         print(f"Directory '{output_dir}' does not exist.")
#         return 0

#     file_count = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
#     print(f"Total PNG screenshots in '{output_dir}': {file_count}")
#     return file_count

# if __name__ == '__main__':
#     count_extracted_files(screenshots_output_dir)

# import os

# screenshots_output_dir = 'screenshots'  # Directory where screenshots are stored

# def rename_screenshots(output_dir):
#     """Safely rename screenshots sequentially from screenshot_0.png to screenshot_99.png."""
#     if not os.path.exists(output_dir):
#         print(f"Directory '{output_dir}' does not exist.")
#         return

#     # Get all PNG files and sort them
#     files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])

#     # First pass: Rename all files to temporary names to avoid conflicts
#     temp_names = {}
#     for i, filename in enumerate(files):
#         temp_name = os.path.join(output_dir, f"temp_{i}.png")
#         old_path = os.path.join(output_dir, filename)
#         os.rename(old_path, temp_name)
#         temp_names[temp_name] = i  # Store mapping for final renaming

#     # Second pass: Rename temp files to final names
#     for temp_name, i in temp_names.items():
#         final_name = os.path.join(output_dir, f"screenshot_{i}.png")
#         os.rename(temp_name, final_name)
#         print(f"Renamed {temp_name} -> {final_name}")

# if __name__ == '__main__':
#     rename_screenshots(screenshots_output_dir)
