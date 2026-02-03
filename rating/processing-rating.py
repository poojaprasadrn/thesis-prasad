import pandas as pd
from collections import Counter

# Step 1: Load and combine CSV files
# file_paths = [
#     "ratings-abhi-set1.csv", "ratings-anu-set2.csv", "ratings-naks-set3.csv",
#     "ratings-pav-set2.csv", "ratings-prah-set3.csv", "ratings-shrads-set3.csv",
#     "ratings-sum-set2.csv", "ratings-supi-set1.csv", "ratings-vish-set1.csv"
# ]
file_paths = [
    "ratings-chatgpt.csv"
    # "ratings-set1-pooja.csv", "ratings-set2-pooja.csv","ratings-set3-pooja.csv"
]
df_list = [pd.read_csv(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.strip()

# Step 2: Clean ratings
df.rename(columns={"User1": "Rating"}, inplace=True)
df["Rating"] = df["Rating"].astype(str).str.strip()
df["Screenshot"] = df["Screenshot"].str.strip()

# Step 3: Load screenshot URLs and clean
screenshot_urls_df = pd.read_csv('screenshot_urls_final_2.csv')
screenshot_urls_df.columns = screenshot_urls_df.columns.str.strip()
screenshot_urls_df.rename(columns={'Screenshot Path': 'Screenshot'}, inplace=True)
screenshot_urls_df['Screenshot'] = screenshot_urls_df['Screenshot'].str.strip()
print(screenshot_urls_df.columns)

# === Add: Normalize screenshot filenames to only the filename (no path), lowercased ===
def extract_filename(path):
    return str(path).split('/')[-1].strip().lower()

df['Screenshot_norm'] = df['Screenshot'].apply(extract_filename)
screenshot_urls_df['Screenshot_norm'] = screenshot_urls_df['Screenshot'].apply(extract_filename)
screenshot_urls_df["URL"] = screenshot_urls_df["URL"].astype(str).str.strip().str.lower()

# Merge URLs
df_merged = pd.merge(df, screenshot_urls_df, on="Screenshot_norm", how="left")
print(df_merged[['Screenshot_norm', 'URL']].head())

# Optional: See if any didn't match
missing_urls = df_merged['URL'].isnull().sum()
if missing_urls > 0:
    print(f"WARNING: {missing_urls} screenshots could not be matched to a URL.")

# Step 4: Majority voting function
def majority_vote(ratings):
    ratings = [r for r in ratings if pd.notna(r)]
    count = Counter(ratings)
    most_common = count.most_common()
    if len(most_common) == 0:
        return ""
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]  # majority
    return ", ".join(sorted(set(ratings)))  # no majority

# Step 5: Aggregate with majority vote
df_final = df_merged.groupby("Screenshot_norm").agg({
    "Rating": lambda r: majority_vote(r),
    "URL": "first"
}).reset_index().rename(columns={"Screenshot_norm": "Screenshot"})


# Step 6: Save
df_final.to_csv("ground_truth_spam_with_url_final_v3.csv", index=False)

print("✅ Majority-voted dataset saved as ground_truth_spam_with_url_final_v3.csv")




# import pandas as pd

# # Step 1: Load and Combine CSV Files
# file_paths = ["ratings1.csv", "ratings2.csv", "ratings.csv"]  # Update with actual file names
# df_list = [pd.read_csv(file) for file in file_paths]

# # Ensure all files have the same column names (remove extra spaces)
# df = pd.concat(df_list, ignore_index=True)
# df.columns = df.columns.str.strip()  # Remove spaces from column names

# # Step 2: Define Spam vs. Non-Spam Categories
# spam_categories = {"class-4", "count-banner", "count-broken", "count-error"}
# non_spam_categories = {"class-1", "class-2", "class-3", "count-webshop", "count-guide", "count-other"}

# # Step 3: Convert Ratings to Numeric Binary Labels
# rating_to_numeric = {label: 1 for label in spam_categories}  # Spam → 1
# rating_to_numeric.update({label: 0 for label in non_spam_categories})  # Non-Spam → 0

# # Step 4: Rename "User1" to "Rating" and Strip Spaces from Values
# df.rename(columns={"User1": "Rating"}, inplace=True)  # Rename column
# df["Rating"] = df["Rating"].astype(str).str.strip()  # Ensure no extra spaces

# # Step 5: Apply Mapping to Convert Ratings to Binary Labels
# df["Label"] = df["Rating"].map(rating_to_numeric)

# # Step 6: Aggregation Function for Ratings (Includes Vote Count for 90-99)
# def aggregate_ratings(series, screenshot):
#     """Aggregates ratings for screenshots 90-99 with counts."""
#     rating_counts = series.dropna().value_counts().to_dict()  # Count occurrences of each rating
    
#     # Extract screenshot number safely
#     try:
#         screenshot_num = int(''.join(filter(str.isdigit, screenshot)))  # Extract numeric part
#     except ValueError:
#         return ', '.join(set(series.dropna().astype(str)))  # If filename isn't standard, return normal ratings
    
#     if 90 <= screenshot_num <= 99:  # Only for screenshots 90-99
#         return ', '.join([f"{rating}({count})" for rating, count in rating_counts.items()])
#     else:
#         return ', '.join(set(series.dropna().astype(str)))  # Keep unique ratings normally

# # Step 7: Aggregate by Screenshot (Keeping All Ratings and Averaging Label)
# df_grouped = df.groupby("Screenshot").agg({
#     "Rating": lambda x: aggregate_ratings(x, x.name),  # Use aggregation function
#     "Label": "mean"  # Average binary values for repeated screenshots
# }).reset_index()

# # Step 8: Convert Averaged Values to "Spam" or "Non-Spam" (Threshold: 0.5)
# df_grouped["Label"] = df_grouped["Label"].apply(lambda x: "Spam" if x >= 0.5 else "Non-Spam")

# # Step 9: Save the Processed Ground Truth Data
# df_grouped.to_csv("ground_truth_spam.csv", index=False)

# print("✅ Processed dataset saved as ground_truth_spam.csv")
