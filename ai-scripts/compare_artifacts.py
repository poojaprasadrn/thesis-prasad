#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python compare_artifacts.py \
#   --old_model      /home/yili5634/Desktop/thesis-pooja/best_model.pkl \
#   --old_vectorizer /home/yili5634/Desktop/thesis-pooja/best_vectorizer.pkl \
#   --new_model      /home/yili5634/Desktop/thesis-pooja/ai-scripts/model.pkl \
#   --new_vectorizer /home/yili5634/Desktop/thesis-pooja/ai-scripts/vectorizer.pkl \
#   --old_threshold  0.5 \
#   --new_threshold  0.5

import argparse, json, pickle, hashlib, sys
import numpy as np

def sha256_of_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def np_hash(arr):
    if arr is None: return "NONE"
    import numpy as np
    return sha256_of_bytes(np.ascontiguousarray(arr).view(np.uint8))

def signature_for_vectorizer(vec) -> dict:
    vocab = getattr(vec, "vocabulary_", None)
    if vocab:
        items = [f"{k}:{v}" for k, v in sorted(vocab.items())]
        vocab_hash = sha256_of_bytes("\n".join(items).encode("utf-8"))
        vocab_size = len(vocab)
    else:
        vocab_hash = "NONE"; vocab_size = 0
    return {
        "type": vec.__class__.__name__,
        "ngram_range": getattr(vec, "ngram_range", None),
        "analyzer": getattr(vec, "analyzer", None),
        "lowercase": getattr(vec, "lowercase", None),
        "max_features": getattr(vec, "max_features", None),
        "stop_words": "SET" if getattr(vec, "stop_words", None) is not None else None,
        "vocabulary_size": vocab_size,
        "vocabulary_hash": vocab_hash,
    }

def signature_for_model(model) -> dict:
    name = model.__class__.__name__
    sig = {"type": name}
    if hasattr(model, "classes_"):
        sig["classes_"] = sha256_of_bytes(model.classes_.tobytes())

    for attr in ("coef_", "intercept_", "feature_log_prob_", "class_count_",
                 "class_log_prior_", "theta_", "sigma_", "support_vectors_", "dual_coef_"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, np.ndarray):
                sig[attr] = np_hash(val)
            elif isinstance(val, list) and val and isinstance(val[0], np.ndarray):
                cat = sha256_of_bytes(b"".join([np.ascontiguousarray(a).tobytes() for a in val]))
                sig[attr] = cat
            else:
                try:
                    sig[attr] = sha256_of_bytes(pickle.dumps(val, protocol=4))
                except Exception:
                    sig[attr] = f"UNHASHABLE:{type(val).__name__}"

    try:
        params = model.get_params(deep=True)
        pairs = [f"{k}={params[k]}" for k in sorted(params.keys())]
        sig["hyperparams_hash"] = sha256_of_bytes("\n".join(pairs).encode("utf-8"))
    except Exception:
        sig["hyperparams_hash"] = "NA"
    return sig

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def diff_dict(a, b, path=""):
    diffs = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        va = a.get(k, "<MISSING>")
        vb = b.get(k, "<MISSING>")
        if va != vb:
            diffs.append(f"{path}.{k}: {va}  !=  {vb}")
    return diffs

def main():
    ap = argparse.ArgumentParser(description="Compare OLD vs NEW model/vectorizer artifacts")
    ap.add_argument("--old_model", required=True)
    ap.add_argument("--old_vectorizer", required=True)
    ap.add_argument("--new_model", required=True)
    ap.add_argument("--new_vectorizer", required=True)
    ap.add_argument("--old_threshold", type=float, default=0.5)
    ap.add_argument("--new_threshold", type=float, default=0.5)
    args = ap.parse_args()

    old_m = load_pickle(args.old_model)
    old_v = load_pickle(args.old_vectorizer)
    new_m = load_pickle(args.new_model)
    new_v = load_pickle(args.new_vectorizer)

    sig_old = {"vectorizer": signature_for_vectorizer(old_v),
               "model":      signature_for_model(old_m),
               "threshold":  args.old_threshold}
    sig_new = {"vectorizer": signature_for_vectorizer(new_v),
               "model":      signature_for_model(new_m),
               "threshold":  args.new_threshold}

    vdiff = diff_dict(sig_old["vectorizer"], sig_new["vectorizer"], path="vectorizer")
    mdiff = diff_dict(sig_old["model"], sig_new["model"], path="model")
    tdiff = [] if sig_old["threshold"] == sig_new["threshold"] else [f"threshold: {sig_old['threshold']} != {sig_new['threshold']}"]

    if not (vdiff or mdiff or tdiff):
        print("✅ SAME: Artifacts and threshold match. You do NOT need to re-predict the 7M.")
        sys.exit(0)
    else:
        print("⚠️  DIFFERENT: Found differences below. You SHOULD re-predict the 7M with the NEW artifacts to keep results consistent.\n")
        if vdiff:
            print("Vectorizer differences:")
            for d in vdiff: print("  -", d)
        if mdiff:
            print("\nModel differences:")
            for d in mdiff: print("  -", d)
        if tdiff:
            print("\nThreshold difference:")
            for d in tdiff: print("  -", d)
        sys.exit(0)

if __name__ == "__main__":
    import numpy as np  # needed for isinstance checks
    main()
