import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from elasticsearch import Elasticsearch, helpers

# === CONFIG ===

HUMAN_JSONL = "pan24-generative-authorship-news-train/human.jsonl"
AI_JSONL = "pan24-generative-authorship-news-train/machines/gpt-4-turbo-preview.jsonl"

PREEXTRACTED_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/startpage_10_per_date_v1.csv"
BEST_MODEL_FILE = "best_model.pkl"
BEST_VECTORIZER_FILE = "best_vectorizer.pkl"
CLASSIFIED_RESULTS_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v3.csv"

ES_HOST = "https://elasticsearch.srv.webis.de/"
ES_USERNAME = "yili5634"
ES_PASSWORD = "KqAk50zmz80BSjtn"
ES_INDEX = "wstud_yili5634_classification_results"

FORCE_RETRAIN = False
UPLOAD_TO_ES = True

# === 1) Train ===
# def evaluate_models():
#     df_human = pd.read_json(HUMAN_JSONL, lines=True)
#     df_human['label'] = 1
#     df_ai = pd.read_json(AI_JSONL, lines=True)
#     df_ai['label'] = 0

#     df = pd.concat([df_human, df_ai]).reset_index(drop=True)
#     X = df['text']
#     y = df['label']

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     vectorizers = {
#         "TF": CountVectorizer(max_features=10000, ngram_range=(1, 2)),
#         "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
#     }
#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#         "Multinomial Naive Bayes": MultinomialNB(),
#         "SVM": SVC(probability=True, kernel='linear', random_state=42)
#     }

#     best_score = 0
#     best_model = None
#     best_vectorizer = None

#     for vec_name, vectorizer in vectorizers.items():
#         X_train_vec = vectorizer.fit_transform(X_train)
#         X_val_vec = vectorizer.transform(X_val)

#         for model_name, model in models.items():
#             print(f"🔬 {model_name} + {vec_name}")
#             model.fit(X_train_vec, y_train)
#             y_pred = model.predict(X_val_vec)
#             score = accuracy_score(y_val, y_pred)
#             print(f"Accuracy: {score:.4f}")
#             print(classification_report(y_val, y_pred))

#             if score > best_score:
#                 best_score = score
#                 best_model = model
#                 best_vectorizer = vectorizer

#     with open(BEST_MODEL_FILE, "wb") as mf:
#         pickle.dump(best_model, mf)
#     with open(BEST_VECTORIZER_FILE, "wb") as vf:
#         pickle.dump(best_vectorizer, vf)

#     print(f"✅ Best: {best_model} with {best_vectorizer} (Accuracy: {best_score:.4f})")

def evaluate_models():
    # --- load GT data ---
    df_human = pd.read_json(HUMAN_JSONL, lines=True)
    df_human['label'] = 1
    df_ai = pd.read_json(AI_JSONL, lines=True)
    df_ai['label'] = 0

    df = pd.concat([df_human, df_ai]).reset_index(drop=True)
    X_all = df['text'].astype(str)
    y_all = df['label'].astype(int)

    # --- 80/10/10 split (stratified) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # --- candidates ---
    vectorizers = {
        "TF": CountVectorizer(max_features=10000, ngram_range=(1, 2)),
        "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    }
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, kernel='linear', random_state=42)
    }

    # --- select best by validation F1-macro ---
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    best_score = -1.0
    best_model = None
    best_vectorizer = None
    best_name = None

    for vec_name, vectorizer in vectorizers.items():
        # fit ONLY on train
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec   = vectorizer.transform(X_val)

        for model_name, model in models.items():
            name = f"{model_name} + {vec_name}"
            print(f"\n🔬 {name}")

            model.fit(X_train_vec, y_train)
            y_val_pred = model.predict(X_val_vec)

            acc = accuracy_score(y_val, y_val_pred)
            f1m = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
            print(f"[VAL] Acc={acc:.4f} | F1-macro={f1m:.4f}")
            print(classification_report(y_val, y_val_pred, target_names=["AI", "Human"], zero_division=0))

            if f1m > best_score:
                best_score = f1m
                best_model = model
                best_vectorizer = vectorizer
                best_name = name

    # --- save the best ---
    with open(BEST_MODEL_FILE, "wb") as mf:
        pickle.dump(best_model, mf)
    with open(BEST_VECTORIZER_FILE, "wb") as vf:
        pickle.dump(best_vectorizer, vf)
    print(f"\n✅ Best by VAL F1-macro: {best_name} (F1-macro={best_score:.4f})")
    print(f"   Saved → {BEST_MODEL_FILE}, {BEST_VECTORIZER_FILE}")

    # --- final evaluation on TEST (held-out 10%) ---
    X_test_vec = best_vectorizer.transform(X_test)
    y_test_pred = best_model.predict(X_test_vec)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1m = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

    print("\n🧪 HELD-OUT TEST RESULTS (10%)")
    print(f"[TEST] Acc={test_acc:.4f} | F1-macro={test_f1m:.4f}")
    print("Confusion matrix [rows: true, cols: pred] (0=AI, 1=Human):")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, target_names=["AI", "Human"], zero_division=0))


# === 2) Predict line by line + probability + real metrics if GT ===
def classify_text_line_by_line(input_csv, vectorizer_file, model_file, output_file=CLASSIFIED_RESULTS_CSV):
    with open(vectorizer_file, "rb") as vf:
        vectorizer = pickle.load(vf)
    with open(model_file, "rb") as mf:
        model = pickle.load(mf)

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("url,path,timestamp,ai_label,ai_probability\n")


    total = 0
    ai_count = 0
    human_count = 0
    ai_prob_sum_ai = 0.0
    ai_prob_sum_human = 0.0

    y_true = []
    y_pred = []


    # Track line index
    current_index = 0

    for row in pd.read_csv(input_csv, chunksize=1):
    
        text = row['Text'].values[0]
        text = str(text).strip()  # force to string, remove whitespace

        if not text or text.lower() == "nan" or text.isnumeric():
            current_index += 1
            # skip if text is empty, NaN, or just a number
            continue
        X = vectorizer.transform([text])
        prob_human = model.predict_proba(X)[0, 1]
        prob_ai = 1 - prob_human
        label = "AI Detected" if prob_ai > 0.5 else "Human"

        if label == "AI Detected":
            ai_count += 1
            ai_prob_sum_ai += prob_ai
        else:
            human_count += 1
            ai_prob_sum_human += prob_ai

        # If GT is present, use it
        if 'true_label' in row.columns:
            true_label = row['true_label'].values[0]
            y_true.append(true_label)
            y_pred.append(1 if label == "AI Detected" else 0)

        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"{row['URL'].values[0]},{row['File Path'].values[0]},{row['Timestamp'].values[0]},{label},{prob_ai:.5f}\n")

        total += 1
        if total % 1000 == 0:
            print(f"✅ {total} done. [AI: {ai_count}, Human: {human_count}]")

    print("\n🎉 Done predicting!")
    print(f"🔢 Total URLs: {total}")
    print(f"🤖 AI Detected: {ai_count}")
    print(f"🧑 Human: {human_count}")
    print(f"📊 Avg AI prob (AI class): {ai_prob_sum_ai/ai_count:.4f}" if ai_count > 0 else "No AI rows")
    print(f"📊 Avg AI prob (Human class): {ai_prob_sum_human/human_count:.4f}" if human_count > 0 else "No Human rows")

    # If GT present, print real metrics too
    if y_true:
        print("\n✅ EVALUATION ON TRUE LABELS:")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=["Human", "AI Detected"]))

# === 3) Upload ===
def upload_to_elasticsearch(es_host, es_username, es_password, classification_file, index_name):
    es = Elasticsearch(
        hosts=[es_host],
        basic_auth=(es_username, es_password),
        verify_certs=True
    )

    df = pd.read_csv(classification_file)
    actions = [
        {
            "_index": index_name,
            "_source": result
        }
        for result in df.to_dict(orient='records')
    ]

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        print(f"Index '{index_name}' created.")

    helpers.bulk(es, actions)
    print(f"✅ Uploaded results to Elasticsearch index '{index_name}'.")

# === MAIN ===
if __name__ == "__main__":
    if FORCE_RETRAIN or not (os.path.exists(BEST_MODEL_FILE) and os.path.exists(BEST_VECTORIZER_FILE)):
        print("\n🎓 Training AI vs Human detector using real GT...")
        evaluate_models()
    else:
        print("\n⚡️ Using existing trained model and vectorizer.")

    print("\n🔬 Predicting each URL with full probability + metrics if GT present...")
    classify_text_line_by_line(PREEXTRACTED_CSV, BEST_VECTORIZER_FILE, BEST_MODEL_FILE)

    # if UPLOAD_TO_ES:
    #     #print("\n🔗 Uploading to Elasticsearch...")
    #     #upload_to_elasticsearch(ES_HOST, ES_USERNAME, ES_PASSWORD, CLASSIFIED_RESULTS_CSV, ES_INDEX)
    # else:
    #     print("\n📌 Skipped upload (UPLOAD_TO_ES=False)")
#----------------------------------------



# eval_saved.py
# import os, pickle, pandas as pd
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# # --- EDIT THESE PATHS IF NEEDED ---
# HUMAN_JSONL = "pan24-generative-authorship-news-train/human.jsonl"
# AI_JSONL    = "pan24-generative-authorship-news-train/machines/gpt-4-turbo-preview.jsonl"
# BEST_MODEL_FILE = "best_model.pkl"
# BEST_VECTORIZER_FILE = "best_vectorizer.pkl"

# # Optional: set to an int (e.g., 5000) to eval on a faster random subset.
# N_SAMPLE = None  # or e.g. 5000

# def load_pan24():
#     df_h = pd.read_json(HUMAN_JSONL, lines=True); df_h["label"] = 1
#     df_a = pd.read_json(AI_JSONL, lines=True);    df_a["label"] = 0
#     df = pd.concat([df_h, df_a], ignore_index=True)
#     df["text"] = df["text"].astype(str)
#     df["label"] = df["label"].astype(int)
#     return df

# def main():
#     if not (os.path.exists(BEST_MODEL_FILE) and os.path.exists(BEST_VECTORIZER_FILE)):
#         raise SystemExit("Saved model/vectorizer not found. Put best_model.pkl & best_vectorizer.pkl next to this script.")

#     with open(BEST_VECTORIZER_FILE, "rb") as vf:
#         vectorizer = pickle.load(vf)
#     with open(BEST_MODEL_FILE, "rb") as mf:
#         model = pickle.load(mf)

#     df = load_pan24()
#     if N_SAMPLE:
#         df = df.sample(n=min(N_SAMPLE, len(df)), random_state=42)

#     X = vectorizer.transform(df["text"])
#     y = df["label"].values
#     y_pred = model.predict(X)

#     acc = accuracy_score(y, y_pred)
#     p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(y, y_pred, average="binary", pos_label=1, zero_division=0)
#     p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)

#     print("\n===== EVALUATION (saved model, no retrain) =====")
#     print(f"Samples: {len(y)}")
#     print(f"Accuracy       : {acc:.4f}")
#     print(f"Binary (pos=1) : Precision={p_bin:.4f}  Recall={r_bin:.4f}  F1={f1_bin:.4f}")
#     print(f"Macro          : Precision={p_mac:.4f}  Recall={r_mac:.4f}  F1={f1_mac:.4f}")
#     print("\nConfusion matrix (rows=true, cols=pred) [0=AI, 1=Human]:")
#     print(confusion_matrix(y, y_pred))
#     print("\nPer-class report:")
#     print(classification_report(y, y_pred, target_names=["AI", "Human"], zero_division=0))

# if __name__ == "__main__":
#     main()
