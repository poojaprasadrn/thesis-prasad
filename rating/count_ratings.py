import pandas as pd

# Load the CSV
df = pd.read_csv("ground_truth_spam_with_url_final.csv")

# Helper to clean and split rating strings
def get_ratings_list(rating):
    return [r.strip() for r in str(rating).split(',') if r.strip()]

# Count occurrences
rating_counts = {}
for rating_str in df['Rating']:
    ratings = get_ratings_list(rating_str)
    ratings_key = ', '.join(sorted(ratings))
    rating_counts[ratings_key] = rating_counts.get(ratings_key, 0) + 1

# Separate into Individual vs Combination
output_rows = []
for rating_key, count in rating_counts.items():
    label_type = "Individual" if len(rating_key.split(',')) == 1 else "Combination"
    output_rows.append({"Type": label_type, "Rating_or_Combination": rating_key, "Count": count})

# Save to CSV
output_df = pd.DataFrame(output_rows)
output_df.to_csv("all_rating_counts.csv", index=False)


print("✅ Results saved to 'all_rating_counts.csv'")
