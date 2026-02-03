import os, json, torch
from joblib import load
from transformers import AutoTokenizer

BIN="/mnt/ceph/storage/data-tmp/current/yili5634/bert_bin_model"
MULTI="/mnt/ceph/storage/data-tmp/current/yili5634/bert_multi_model"

def show_dir(tag, d):
    print(f"\n[{tag}] {d}")
    for f in ["full_model_bin.pt","full_model_multi.pt","preprocessors.joblib","config.json","tokenizer.json","vocab.txt","feature_dim.txt"]:
        p=os.path.join(d,f)
        if os.path.exists(p):
            sz=os.path.getsize(p)
            print(f"  ✓ {f} ({sz:,} bytes)")
        else:
            print(f"  ✗ {f} (not found)")

def tok_name(d):
    cfg=os.path.join(d,"config.json")
    if os.path.exists(cfg):
        try:
            print("  tokenizer:", AutoTokenizer.from_pretrained(d).__class__.__name__)
        except Exception as e:
            print("  tokenizer: <error>", e)

def model_head(d, fname):
    p=os.path.join(d,fname)
    if os.path.exists(p):
        try:
            m=torch.load(p, map_location="cpu")
            head=None
            if hasattr(m,"classifier"): head=m.classifier
            elif hasattr(m,"bert") and hasattr(m,"classifier"): head=m.classifier
            print("  model head:", type(head).__name__, "->", tuple(head.weight.shape) if hasattr(head,"weight") else "<n/a>")
        except Exception as e:
            print("  load model error:", e)

def pp_shape(d):
    from joblib import load
    pj=os.path.join(d,"preprocessors.joblib")
    if os.path.exists(pj):
        tfidf, selector, scaler = load(pj)
        print("  tfidf vocab:", len(getattr(tfidf, "vocabulary_", {})))
        print("  selector k:", getattr(selector, "k", None))

show_dir("BINARY", BIN)
tok_name(BIN); model_head(BIN, "full_model_bin.pt"); pp_shape(BIN)

show_dir("MULTI", MULTI)
tok_name(MULTI); model_head(MULTI, "full_model_multi.pt"); pp_shape(MULTI)
print("\n✅ If these are the same dirs you pass as --bin_dir / --multi_dir, you’re using these artifacts.")
