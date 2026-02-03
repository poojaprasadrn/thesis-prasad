import pandas as pd

# Load the dataset with URLs and POS tags (assuming your dataset is named 'extracted_pos_tags_for_urls.csv')
data = pd.read_csv("warc_extracted_content.csv")
data['url'] = data['url'].str.strip().str.lower().str.rstrip('/')
# Remove duplicates based on the 'url' column, keeping only the first occurrence of each URL
cleaned_data = data.drop_duplicates(subset="url", keep="first")

# Save the cleaned data to a new file
cleaned_data.to_csv("cleaned_extracted_pos_tags_for_urls.csv", index=False)

# Verify that the duplicates are removed by checking the shape of the cleaned data
print(f"done")
