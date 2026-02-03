import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

from sklearn.pipeline import Pipeline

# Constants
TRAINING_DATA = "training_data.csv"
BEST_MODEL_FILE = "best_model.pkl"
BEST_VECTORIZER_FILE = "best_vectorizer.pkl"

# Load Training Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data['text']
    y = data['label']
    return X, y

# Perform Hyperparameter Tuning
def tune_hyperparameters(X_train, y_train):
    # Define the pipeline
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Define parameter grid
    param_grid = {
        "vectorizer__max_features": [5000, 10000, 15000,20000, 30000,40000,50000],
        "vectorizer__ngram_range": [(1, 1), (1, 2),(1,3)],
        "classifier__C": [1, 10, 20],
        "classifier__penalty": ["l2"]  # Removed invalid 'none'
    }

    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Output results
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    return grid_search.best_estimator_

# Main Workflow
if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data(TRAINING_DATA)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    print("Tuning hyperparameters...")
    best_model = tune_hyperparameters(X_train, y_train)

    # Save the best model and vectorizer
    with open(BEST_MODEL_FILE, "wb") as mf:
        pickle.dump(best_model.named_steps["classifier"], mf)
    with open(BEST_VECTORIZER_FILE, "wb") as vf:
        pickle.dump(best_model.named_steps["vectorizer"], vf)

    # Evaluate the best model on the validation set
    print("Evaluating best model on validation set...")
    y_pred = best_model.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
