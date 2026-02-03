# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("rating/screenshot_urls_final_2.csv")  # Replace with your actual file name

# # List of domains to check for
# exclude_domains = [
#     "amazon", "pinterest", "youtube", "quora", "ebay", "flipkart", "etsy", "linkedin", 
#     "outdoorgearlab", "bestcraftorganizer", "wahoox", "play.google.com", "ukcurtainpoles",
#     "onelittleproject", "forum.electricunicycle", "houzz", "wikipedia", "american-footballshop"
# ]

# # Check which URLs contain any of the exclude domains
# mask = df['URL'].str.contains('|'.join(exclude_domains), case=False, na=False)
# excluded_matches = df[mask]

# # Print results
# print(f"🔎 Found {len(excluded_matches)} URLs containing excluded domains.")
# print(excluded_matches[['URL']])  # Show the matching URLs



# import pandas as pd

# # Load your CSV file
# df = pd.read_csv("extracted_new_urls_with_text.csv")  # Replace with your file name

# # Check for duplicate URLs
# duplicates = df[df.duplicated(subset="URL", keep=False)]

# # Show duplicates (if any)
# if not duplicates.empty:
#     print("🔁 Duplicate URLs found:")
#     print(duplicates)
# else:
#     print("✅ No duplicate URLs found.")


import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("rating/screenshot_urls_final_2.csv")  # Replace with your actual filename
df2 = pd.read_csv("extracted_new_urls_with_text.csv")  # Replace with your actual filename

# Ensure both have a URL column (adjust column name if different)
urls1 = set(df1['URL'])
urls2 = set(df2['URL'])

# Find common and different URLs
common_urls = urls1.intersection(urls2)
only_in_file1 = urls1 - urls2
only_in_file2 = urls2 - urls1

# Display results
print(f"✅ Common URLs: {len(common_urls)}")
print(f"🔹 Only in file1: {len(only_in_file1)}")
print(f"🔹 Only in file2: {len(only_in_file2)}")

# Optional: Save results to CSV
pd.DataFrame({'URL': list(common_urls)}).to_csv("common_urls.csv", index=False)
pd.DataFrame({'URL': list(only_in_file1)}).to_csv("only_in_file1.csv", index=False)
pd.DataFrame({'URL': list(only_in_file2)}).to_csv("only_in_file2.csv", index=False)
