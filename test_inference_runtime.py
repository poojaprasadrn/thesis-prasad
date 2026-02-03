import time
import torch
import pandas as pd
import numpy as np
from joblib import load
from transformers import DistilBertTokenizerFast
import textstat
from textblob import TextBlob
import spacy
from scipy.sparse import csr_matrix, hstack

print("\n🚀 Starting lightweight inference runtime test...\n")

# ===========================
# CONFIGURATION
# ===========================
DATA_PATH = "/mnt/ceph/storage/data-tmp/current/yili5634/master_dataset.csv"
N_SAMPLES = 100
BERT_BIN_PATH = "/mnt/ceph/storage/data-tmp/current/yili5634/bert_bin_model"
BERT_MULTI_PATH = "/mnt/ceph/storage/data-tmp/current/yili5634/bert_multi_model"

# ===========================
# LOAD MODELS + TOKENIZERS
# ===========================
from torch import nn
from transformers import DistilBertModel

class BERTWithFeatures(nn.Module):
    def __init__(self, feature_dim, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + feature_dim, num_labels)

    def forward(self, input_ids, attention_mask, extra_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat((bert_output, extra_features), dim=1)
        logits = self.classifier(self.dropout(combined))
        return logits

print("📦 Loading preprocessing artifacts...")
tfidf_bin, selector_bin, scaler_bin = load(f"{BERT_BIN_PATH}/preprocessors.joblib")
tfidf_multi, selector_multi, scaler_multi = load(f"{BERT_MULTI_PATH}/preprocessors.joblib")

with open(f"{BERT_BIN_PATH}/feature_dim.txt") as f:
    feature_dim_bin = int(f.read())
with open(f"{BERT_MULTI_PATH}/feature_dim.txt") as f:
    feature_dim_multi = int(f.read())

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 Device detected: {device.upper()}")

print("🧩 Loading binary model...")
model_bin = torch.load(f"{BERT_BIN_PATH}/full_model_bin.pt", map_location=device)
print("🧩 Loading multiclass model...")
model_multi = torch.load(f"{BERT_MULTI_PATH}/full_model_multi.pt", map_location=device)
model_bin.eval()
model_multi.eval()

print("✅ Models and tokenizers loaded successfully.\n")

# ===========================
# LOAD SAMPLE DATA
# ===========================
print("📄 Loading dataset...")
df = pd.read_csv(DATA_PATH, nrows=5000)
# auto-detect text column
text_col = "Text" if "Text" in df.columns else "text"
df = df.dropna(subset=[text_col]).sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
print(f"✅ Loaded {len(df)} sample texts from column '{text_col}'.\n")

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ===========================
# FEATURE EXTRACTION FUNCTION
# ===========================
def extract_features(text, tfidf, selector, scaler):
    text_lower = text.lower()
    tfidf_vec = tfidf.transform([text_lower])

    doc = nlp(text_lower)
    pos_tags = ['VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM']
    counts = np.array([[sum(1 for token in doc if token.pos_ == tag) for tag in pos_tags]])
    norm_pos = counts / max(len(text_lower.split()), 1)

    try:
        flesch = textstat.flesch_reading_ease(text_lower)
        grade = textstat.flesch_kincaid_grade(text_lower)
    except:
        flesch, grade = 0.0, 0.0

    basic = np.array([[len(text_lower.split()),
                       sum(c in '!?.,;:' for c in text_lower) / max(len(text_lower.split()), 1),
                       TextBlob(text_lower).sentiment.polarity,
                       TextBlob(text_lower).sentiment.subjectivity,
                       flesch, grade]])

    all_feats = np.hstack([tfidf_vec.toarray(), counts, norm_pos, basic])
    scaled_feats = scaler.transform(selector.transform(all_feats))
    return torch.tensor(scaled_feats.squeeze(), dtype=torch.float32).unsqueeze(0).to(device)

# ===========================
# INFERENCE LOOP
# ===========================
bin_times, multi_times, total_times = [], [], []

print("⏱️ Running inference timing test for both stages...\n")

for i, row in df.iterrows():
    text = row[text_col]
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)

    # ----- Binary stage -----
    start_bin = time.time()
    feats_bin = extract_features(text, tfidf_bin, selector_bin, scaler_bin)
    with torch.no_grad():
        logits_bin = model_bin(inputs["input_ids"], inputs["attention_mask"], feats_bin)
    end_bin = time.time()
    elapsed_bin = (end_bin - start_bin) * 1000
    bin_times.append(elapsed_bin)

    pred_bin = torch.argmax(logits_bin, dim=1).item()

    # ----- Multiclass stage -----
    if pred_bin == 1:  # Only process if "Review"
        start_multi = time.time()
        feats_multi = extract_features(text, tfidf_multi, selector_multi, scaler_multi)
        with torch.no_grad():
            logits_multi = model_multi(inputs["input_ids"], inputs["attention_mask"], feats_multi)
        end_multi = time.time()
        elapsed_multi = (end_multi - start_multi) * 1000
        multi_times.append(elapsed_multi)
        total_time = elapsed_bin + elapsed_multi
    else:
        total_time = elapsed_bin  # skip second stage for non-reviews

    total_times.append(total_time)
    print(f"{i+1:02d}. Binary: {elapsed_bin:.2f} ms | Multi: {elapsed_multi if pred_bin==1 else 0:.2f} ms | Total: {total_time:.2f} ms")

# ===========================
# SUMMARY
# ===========================
print("\n================ SUMMARY ================\n")
print(f"🧩 Average binary stage time: {np.mean(bin_times):.2f} ms")
if multi_times:
    print(f"🧩 Average multiclass stage time: {np.mean(multi_times):.2f} ms (on {len(multi_times)} review docs)")
else:
    print("🧩 No reviews detected in sample for multiclass timing.")

print(f"🕒 Overall average total time: {np.mean(total_times):.2f} ms per document ({device.upper()})")
print(f"    Min: {np.min(total_times):.2f} ms | Max: {np.max(total_times):.2f} ms")
print("\n✅ Timing test completed.\n")
