import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Load CSV file
def load_csv(file_path, rename_first_col=None):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()  # Convert columns to lowercase and remove spaces
    if rename_first_col:
        df.rename(columns={df.columns[0]: rename_first_col}, inplace=True)  # Ensure first column is named correctly
    print("Available columns in loaded dataset:", df.columns.tolist())  # Debugging: Print available columns
    return df

# Extract features using TF-IDF vectorization
def extract_text_features(df, text_column="text"):
    if text_column in df.columns:
        vectorizer = TfidfVectorizer(max_features=500)
        text_features = vectorizer.fit_transform(df[text_column].fillna(""))
        return text_features.toarray()
    return np.zeros((len(df), 500))  # Return empty feature set if text column is missing

# Prepare data using extracted features
def prepare_data(df, target_column, text_column="text"):
    if target_column not in df.columns:
        raise KeyError(f"Error: Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    df = df.dropna(subset=[target_column])  # Ensure we have known target values
    text_features = extract_text_features(df, text_column)
    y = df[target_column].values  # Target is Lighthouse Score
    
    print("X shape before processing:", text_features.shape)
    print("X sample data:\n", text_features[:5])
    
    return text_features, y

# Train regression model using Random Forest
def train_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split into train and test
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use Random Forest for better predictions
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance: MAE = {mae:.4f}, R2 Score = {r2:.4f}")
    
    return model

# Predict missing scores using actual Lighthouse scores from combined dataset
def predict_missing_scores(gt_df, urls_df, model, target_column, text_column="text"):
    print("Columns in urls_df before prediction:", urls_df.columns.tolist())  # Debugging
    
    # Merge URLs with available Lighthouse Scores from ground truth
    urls_df = urls_df.merge(gt_df[["url", target_column]], on="url", how="left")
    
    # Predict only for missing scores
    missing_mask = urls_df[target_column].isna()
    if missing_mask.any():
        X_missing = extract_text_features(urls_df.loc[missing_mask], text_column)
        urls_df.loc[missing_mask, target_column] = model.predict(X_missing)
    
    urls_df[target_column] = urls_df[target_column].round().astype(int)  # Round to whole number
    
    return urls_df[["url", target_column]]  # Keep only URL and predicted score

# Main execution
gt_file_path = "/mnt/ceph/storage/data-tmp/current/yili5634/lighthouse_scores/combined_lighthouse_scores.csv"
urls_file_path = "/home/yili5634/Desktop/thesis-pooja/warc_extracted_5_urls.csv"
target_column = "lighthouse score".lower().strip()  # Normalize target column name
text_column = "text"  # Use webpage text for feature extraction
url_column = "url"  # Ensure this matches the column name in CSV

# Load ground truth and URLs
gt_df = load_csv(gt_file_path)
urls_df = load_csv(urls_file_path, rename_first_col=url_column)  # Explicitly rename first column to 'url'

# Prepare data for training
X, y = prepare_data(gt_df, target_column, text_column)
model = train_regression(X, y)

# Predict scores for the URLs dataset using actual Lighthouse scores from ground truth
urls_df = predict_missing_scores(gt_df, urls_df, model, target_column, text_column)

# Save updated CSV
urls_df.to_csv("predicted_lh_scores.csv", index=False)
print("Predicted scores saved to predicted_lh_scores.csv")
