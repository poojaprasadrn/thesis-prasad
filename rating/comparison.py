import pandas as pd

# === CATEGORY NORMALIZATION MAP ===
category_map = {
    # Webshop
    "webshop": "Webshop", "count-webshop": "Webshop",
    # Other/Informational
    "other website": "Other", "count-other": "Other", "other": "Other",
    # Buyer's Guide
    "buyer guide": "Buyer's Guide", "buyer's guide": "Buyer's Guide", "buyers guide": "Buyer's Guide", "buyer’s guide": "Buyer's Guide", "count-guide": "Buyer's Guide",
    # Banner/Ad overlay
    "ad or banner overlay": "Banner", "count-banner": "Banner",
    # Broken Design
    "count-broken": "Broken Design",
    # Reviews
    "great review": "Great Review", "class-1": "Great Review",
    "good review": "Good Review", "class-3": "Good Review",
    "decent review": "Decent Review", "class-2": "Decent Review",
    "poor review": "Poor Review", "class-4": "Poor Review",
    "spam review": "Spam Review", "spam review": "Spam Review",
}

def normalize(val):
    val = str(val).split(',')[0].strip().lower()
    return category_map.get(val, val)

# === READ FILES ===
#chatgpt = pd.read_csv("ratings-chatgpt.csv")        # Screenshot, Rating
user1   = pd.read_csv("ground_truth_spam_with_url_final_v2.csv") # Screenshot, Rating, URL
user2   = pd.read_csv("ground_truth_spam_with_url_final.csv")    # Screenshot, Rating, URL

# === RENAME COLUMNS ===
#chatgpt = chatgpt.rename(columns={"Rating": "ChatGPT"})
user1   = user1.rename(columns={"Rating": "User1"})
user2   = user2.rename(columns={"Rating": "User2"})

# === MERGE ALL ON 'Screenshot' ===
merged = user2.merge(user1[["Screenshot", "User1"]], on="Screenshot", how="outer")
#merged = merged.merge(user2[["Screenshot", "User2"]], on="Screenshot", how="outer")

# === NORMALIZE RATINGS ===
#merged["ChatGPT_norm"] = merged["ChatGPT"].apply(normalize)
merged["User1_norm"]   = merged["User1"].apply(normalize)
merged["User2_norm"]   = merged["User2"].apply(normalize)

# === AGREEMENT LOGIC ===
def get_agreement(row):
    values = [row["User2_norm"], row["User1_norm"]]
    uniques = set(values)
    if len(uniques) == 1:
        return "All Same"
    # elif len(uniques) == 2:
    #     # If any two match
    #     if values[0] == values[1] or values[0] == values[2] or values[1] == values[2]:
    #         return "Two Same"
    #     else:
    #         return "All Different"
    else:
        return "All Different"

merged["Agreement"] = merged.apply(get_agreement, axis=1)

# === FINAL OUTPUT ===
output = merged[["Screenshot", "User2_norm", "User1_norm", "Agreement"]]
output.to_csv("agreement_analysis_v3.csv", index=False)

# === SUMMARY TABLE ===
summary = output["Agreement"].value_counts().reset_index()
summary.columns = ["Agreement Type", "Count"]
summary.to_csv("agreement_summary_v3.csv", index=False)


