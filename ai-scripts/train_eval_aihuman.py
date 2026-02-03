# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # python /home/yili5634/Desktop/thesis-pooja/ai-scripts/train_eval_aihuman.py \
# #   --human_jsonl /home/yili5634/Desktop/thesis-pooja/pan24-generative-authorship-news-train/human.jsonl \
# #   --ai_jsonl    /home/yili5634/Desktop/thesis-pooja/pan24-generative-authorship-news-train/machines/gpt-4-turbo-preview.jsonl \
# #   --out_dir     /home/yili5634/Desktop/thesis-pooja/ai-scripts
# #   --seed 42 \
# #   --threshold 0.5

# import os, json, argparse, pickle, hashlib
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_recall_fscore_support,
#     classification_report, confusion_matrix
# )
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC


# def sha256_of_bytes(b: bytes) -> str:
#     h = hashlib.sha256()
#     h.update(b)
#     return h.hexdigest()

# def np_hash(arr: np.ndarray) -> str:
#     if arr is None:
#         return "NONE"
#     return sha256_of_bytes(np.ascontiguousarray(arr).view(np.uint8))

# def signature_for_vectorizer(vec) -> dict:
#     # stable summary of what matters for features
#     vocab = getattr(vec, "vocabulary_", None)
#     if vocab:
#         # sort by token then hash joined "token:idx"
#         items = [f"{k}:{v}" for k, v in sorted(vocab.items())]
#         vocab_hash = sha256_of_bytes("\n".join(items).encode("utf-8"))
#         vocab_size = len(vocab)
#     else:
#         vocab_hash = "NONE"
#         vocab_size = 0

#     sig = {
#         "type": vec.__class__.__name__,
#         "ngram_range": getattr(vec, "ngram_range", None),
#         "analyzer": getattr(vec, "analyzer", None),
#         "lowercase": getattr(vec, "lowercase", None),
#         "max_features": getattr(vec, "max_features", None),
#         "stop_words": "SET" if getattr(vec, "stop_words", None) is not None else None,
#         "vocabulary_size": vocab_size,
#         "vocabulary_hash": vocab_hash,
#     }
#     return sig

# def signature_for_model(model) -> dict:
#     # try to capture learned params for common sklearn classifiers
#     name = model.__class__.__name__
#     sig = {"type": name}

#     if hasattr(model, "classes_"):
#         sig["classes_"] = sha256_of_bytes(model.classes_.tobytes())

#     # LogisticRegression, LinearSVC/SVC(linear) expose coef_/intercept_
#     for attr in ("coef_", "intercept_", "feature_log_prob_", "class_count_",
#                  "class_log_prior_", "theta_", "sigma_", "support_vectors_", "dual_coef_"):
#         if hasattr(model, attr):
#             val = getattr(model, attr)
#             if isinstance(val, np.ndarray):
#                 sig[attr] = np_hash(val)
#             elif isinstance(val, list) and val and isinstance(val[0], np.ndarray):
#                 cat = sha256_of_bytes(b"".join([np.ascontiguousarray(a).tobytes() for a in val]))
#                 sig[attr] = cat
#             else:
#                 # fallback: bytes of pickle of that attr
#                 try:
#                     sig[attr] = sha256_of_bytes(pickle.dumps(val, protocol=4))
#                 except Exception:
#                     sig[attr] = f"UNHASHABLE:{type(val).__name__}"

#     # Also hash model hyperparameters (sorted)
#     try:
#         params = model.get_params(deep=True)
#         # turn into deterministic string
#         pairs = [f"{k}={params[k]}" for k in sorted(params.keys())]
#         sig["hyperparams_hash"] = sha256_of_bytes("\n".join(pairs).encode("utf-8"))
#     except Exception:
#         sig["hyperparams_hash"] = "NA"

#     return sig

# def evaluate_split(y_true, y_pred, label_names=("AI", "Human")) -> dict:
#     acc = accuracy_score(y_true, y_pred)
#     p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
#     f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
#     return {
#         "accuracy": float(acc),
#         "precision_per_class": dict(zip(label_names, map(float, p))),
#         "recall_per_class":    dict(zip(label_names, map(float, r))),
#         "f1_per_class":        dict(zip(label_names, map(float, f1))),
#         "f1_macro": float(f1_macro),
#         "support_per_class":   dict(zip(label_names, map(int, support))),
#         "confusion_matrix":    confusion_matrix(y_true, y_pred).tolist(),
#         "report_text": classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
#     }

# def main():
#     ap = argparse.ArgumentParser(description="Train/eval AI-vs-Human (80/10/10), save artifacts + signatures")
#     ap.add_argument("--human_jsonl", required=True)
#     ap.add_argument("--ai_jsonl",     required=True)
#     ap.add_argument("--out_dir",      required=True)
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--threshold", type=float, default=0.5, help="For probability-based models if needed (kept for metadata)")
#     args = ap.parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Load labeled data
#     df_h = pd.read_json(args.human_jsonl, lines=True)
#     df_h["label"] = 1
#     df_a = pd.read_json(args.ai_jsonl,    lines=True)
#     df_a["label"] = 0
#     df = pd.concat([df_h, df_a], ignore_index=True)
#     df = df.dropna(subset=["text"])
#     X_all = df["text"].astype(str)
#     y_all = df["label"].astype(int)

#     # 80/10/10 split
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X_all, y_all, test_size=0.20, stratify=y_all, random_state=args.seed
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=args.seed
#     )

#     # candidates
#     vectorizers = {
#         "TF": CountVectorizer(max_features=10000, ngram_range=(1, 2)),
#         "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
#     }
#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=1000, random_state=args.seed),
#         "Multinomial Naive Bayes": MultinomialNB(),
#         "SVM (linear)": SVC(probability=True, kernel="linear", random_state=args.seed),
#     }

#     # Select by VAL F1-macro
#     best = {"score": -1, "name": None, "model": None, "vectorizer": None}
#     for vname, vec in vectorizers.items():
#         Xtr = vec.fit_transform(X_train)
#         Xva = vec.transform(X_val)
#         for mname, mdl in models.items():
#             mdl.fit(Xtr, y_train)
#             yv = mdl.predict(Xva)
#             f1m = f1_score(y_val, yv, average="macro", zero_division=0)
#             print(f"[VAL] {mname} + {vname}: F1-macro={f1m:.4f}")
#             if f1m > best["score"]:
#                 best.update(score=f1m, name=f"{mname} + {vname}", model=mdl, vectorizer=vec)

#     # Save artifacts
#     with open(out_dir / "model.pkl", "wb") as f:
#         pickle.dump(best["model"], f)
#     with open(out_dir / "vectorizer.pkl", "wb") as f:
#         pickle.dump(best["vectorizer"], f)

#     # Test eval with best
#     Xte = best["vectorizer"].transform(X_test)
#     yte = best["model"].predict(Xte)
#     test_metrics = evaluate_split(y_test, yte)

#     # Write reports/metadata
#     meta = {
#         "chosen": best["name"],
#         "val_f1_macro": best["score"],
#         "seed": args.seed,
#         "threshold": args.threshold,
#         "sizes": {
#             "train": int(len(y_train)),
#             "val":   int(len(y_val)),
#             "test":  int(len(y_test)),
#             "total": int(len(y_all)),
#         },
#         "class_balance_total": {
#             "AI(0)": int((y_all==0).sum()),
#             "Human(1)": int((y_all==1).sum()),
#         }
#     }
#     (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

#     (out_dir / "test_report.txt").write_text(
#         "=== HELD-OUT TEST (10%) ===\n"
#         f"Model: {best['name']}\n\n"
#         f"Accuracy: {test_metrics['accuracy']:.4f}\n"
#         f"F1-macro: {test_metrics['f1_macro']:.4f}\n\n"
#         "Confusion matrix (rows=true, cols=pred) [0=AI,1=Human]:\n"
#         f"{np.array(test_metrics['confusion_matrix'])}\n\n"
#         f"{test_metrics['report_text']}\n"
#     )

#     # Signatures for comparison later
#     sig = {
#         "vectorizer": signature_for_vectorizer(best["vectorizer"]),
#         "model":      signature_for_model(best["model"]),
#         "meta":       meta,
#     }
#     (out_dir / "signature.json").write_text(json.dumps(sig, indent=2))

#     print("\nSaved:")
#     print(" -", out_dir / "model.pkl")
#     print(" -", out_dir / "vectorizer.pkl")
#     print(" -", out_dir / "meta.json")
#     print(" -", out_dir / "signature.json")
#     print(" -", out_dir / "test_report.txt")
#     print("\nDone.")
    

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
#!/usr/bin/env python3



# eval_splits.py
# import argparse, numpy as np, pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score, classification_report

# def load_pan(human_jsonl, ai_jsonl):
#     df_h = pd.read_json(human_jsonl, lines=True)
#     df_h["label"] = 1
#     df_a = pd.read_json(ai_jsonl, lines=True)
#     df_a["label"] = 0
#     df = pd.concat([df_h, df_a], ignore_index=True)
#     df["text"] = df["text"].astype(str)
#     df["label"] = df["label"].astype(int)
#     return df

# def run_once(X, y, scheme, seed):
#     if scheme == "80-10-10":
#         X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
#         X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
#         # fit TF-IDF only on train
#         vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
#         Xtr = vec.fit_transform(X_tr)
#         Xva = vec.transform(X_va)
#         Xte = vec.transform(X_te)
#         # simple tuning example: C in {0.5,1,2}
#         best_f1, best_C = -1, None
#         for C in [0.5, 1.0, 2.0]:
#             clf = SVC(kernel="linear", probability=True, random_state=seed, C=C)
#             clf.fit(Xtr, y_tr)
#             pred_va = clf.predict(Xva)
#             f1 = f1_score(y_va, pred_va, average="macro")
#             if f1 > best_f1: best_f1, best_C = f1, C
#         # refit on train with best C, evaluate test
#         clf = SVC(kernel="linear", probability=True, random_state=seed, C=best_C)
#         clf.fit(Xtr, y_tr)
#         pred_te = clf.predict(Xte)
#         return {
#             "scheme": scheme, "seed": seed, "val_used": True, "C": best_C,
#             "acc": accuracy_score(y_te, pred_te),
#             "f1_macro": f1_score(y_te, pred_te, average="macro")
#         }

#     elif scheme in ("80-20", "70-15-15", "60-20-20"):
#         if scheme == "80-20":
#             # no separate val; 20% is test, no tuning (or very light inner tune across seeds)
#             X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
#             vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
#             Xtr = vec.fit_transform(X_tr); Xte = vec.transform(X_te)
#             clf = SVC(kernel="linear", probability=True, random_state=seed, C=1.0)
#             clf.fit(Xtr, y_tr)
#             pred_te = clf.predict(Xte)
#             return {
#                 "scheme": scheme, "seed": seed, "val_used": False, "C": 1.0,
#                 "acc": accuracy_score(y_te, pred_te),
#                 "f1_macro": f1_score(y_te, pred_te, average="macro")
#             }
#         else:
#             test_size = {"70-15-15": 0.30, "60-20-20": 0.40}[scheme]
#             X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
#             # split tmp into val/test with 50/50 to keep the second number for val and test equal
#             X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
#             vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
#             Xtr = vec.fit_transform(X_tr); Xva = vec.transform(X_va); Xte = vec.transform(X_te)
#             best_f1, best_C = -1, None
#             for C in [0.5, 1.0, 2.0]:
#                 clf = SVC(kernel="linear", probability=True, random_state=seed, C=C)
#                 clf.fit(Xtr, y_tr)
#                 f1 = f1_score(y_va, clf.predict(Xva), average="macro")
#                 if f1 > best_f1: best_f1, best_C = f1, C
#             clf = SVC(kernel="linear", probability=True, random_state=seed, C=best_C)
#             clf.fit(Xtr, y_tr)
#             pred_te = clf.predict(Xte)
#             return {
#                 "scheme": scheme, "seed": seed, "val_used": True, "C": best_C,
#                 "acc": accuracy_score(y_te, pred_te),
#                 "f1_macro": f1_score(y_te, pred_te, average="macro")
#             }

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--human_jsonl", required=True)
#     ap.add_argument("--ai_jsonl", required=True)
#     ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
#     args = ap.parse_args()

#     df = load_pan(args.human_jsonl, args.ai_jsonl)
#     X, y = df["text"].values, df["label"].values

#     schemes = ["80-10-10", "80-20", "70-15-15", "60-20-20"]
#     rows = []
#     for scheme in schemes:
#         for s in args.seeds:
#             rows.append(run_once(X, y, scheme, s))
#     res = pd.DataFrame(rows)
#     print("\n=== Split comparison (mean ± std over seeds) ===")
#     print(res.groupby("scheme")[["acc","f1_macro"]].agg(["mean","std"]).round(4))
#     print("\nDetails:")
#     print(res.sort_values(["scheme","seed"]).to_string(index=False))

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)

def load_gt(human_jsonl, ai_jsonl):
    df_h = pd.read_json(human_jsonl, lines=True)
    df_h["label"] = 1  # Human = 1
    df_a = pd.read_json(ai_jsonl, lines=True)
    df_a["label"] = 0  # AI = 0
    df = pd.concat([df_h, df_a], ignore_index=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_jsonl", required=True)
    ap.add_argument("--ai_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_gt(args.human_jsonl, args.ai_jsonl)
    X_all = df["text"].values
    y_all = df["label"].values

    # ----- 80/10/10 stratified -----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=args.seed
    )

    print("Split sizes:")
    print(f"  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    print("Class balance (AI=0, Human=1):")
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        uniq, cnt = np.unique(yy, return_counts=True)
        print(f"  {name}: " + ", ".join(f"{u}→{c}" for u, c in zip(uniq, cnt)))

    # ----- Vectorizer & Model (fixed: TFIDF 1–2, 10k + linear SVC) -----
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    # (val is not used for selection here; it’s kept separate for transparency)
    X_test_vec  = vectorizer.transform(X_test)

    model = SVC(kernel="linear", probability=True, random_state=args.seed)
    model.fit(X_train_vec, y_train)

    # ----- Evaluate on TEST (held-out 10%) -----
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec_mi, rec_mi, f1_mi, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0
    )
    prec_ma, rec_ma, f1_ma, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])  # rows=true [AI,Human]

    print("\n===== TEST SET METRICS (80/10/10) =====")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Micro-avg      : P={prec_mi:.4f}  R={rec_mi:.4f}  F1={f1_mi:.4f}")
    print(f"Macro-avg      : P={prec_ma:.4f}  R={rec_ma:.4f}  F1={f1_ma:.4f}")

    print("\nPer-class report (0=AI, 1=Human):")
    print(classification_report(y_test, y_pred, target_names=["AI","Human"], zero_division=0))

    print("Confusion matrix (rows=true, cols=pred) [0=AI, 1=Human]:")
    print(cm)

if __name__ == "__main__":
    main()
