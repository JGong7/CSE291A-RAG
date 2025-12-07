import json
import faiss
import numpy as np
import torch
import random
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llm_query_processor import process_queries_with_llm

# ======== Config ========
DATA_1_PATH = "RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "manual_queries.json"
OUTPUT_PATH = "retrieval_results/LLM_structured_queries_results.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_PATH = "cache/model"
EMBEDDINGS_CACHE_PATH = "cache/embeddings_cache.npy"
FAISS_INDEX_CACHE_PATH = "cache/faiss_index_cache.bin"
USE_LLM_STRUCTURING = False # Set to False to skip LLM query structuring
TOP_K = 5
SEED = 42

# ======== Reproducibility ========
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


# ======== Step 1: Load data ========
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f" Warning: {path} not found, skipping.")
        return []

data1 = load_json(DATA_1_PATH)
data2 = load_json(DATA_2_PATH)
recipes = data1 + data2
print(f"Loaded {len(recipes)} total recipes ({len(data1)} from RecipeNLG, {len(data2)} from Spoonacular).")

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)

# Load or cache model
if os.path.exists(MODEL_CACHE_PATH) and os.path.isdir(MODEL_CACHE_PATH):
    print(f"Loading model from cache: {MODEL_CACHE_PATH}")
    model = SentenceTransformer(MODEL_CACHE_PATH)
else:
    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Saving model to cache: {MODEL_CACHE_PATH}")
    model.save(MODEL_CACHE_PATH)


# ======== Step 2: Build embeddings ========
def build_text(r):
    title = r.get("title", "")
    ingredients = " ".join(r.get("ingredients", []))
    instructions = r.get("instructions", "")
    return f"{title} {ingredients} {instructions}".strip()

# Check if embeddings cache exists
if os.path.exists(EMBEDDINGS_CACHE_PATH):
    print(f"Loading embeddings from cache: {EMBEDDINGS_CACHE_PATH}")
    embeddings = np.load(EMBEDDINGS_CACHE_PATH)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
else:
    print("Encoding all recipes...")
    texts = [build_text(r) for r in recipes]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    print(f"Saving embeddings to cache: {EMBEDDINGS_CACHE_PATH}")
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)

dim = embeddings.shape[1]
print(f"Embedding dimension: {dim}")


# ======== Step 3: Build FAISS index ========
# Check if FAISS index cache exists
if os.path.exists(FAISS_INDEX_CACHE_PATH):
    print(f"Loading FAISS index from cache: {FAISS_INDEX_CACHE_PATH}")
    index = faiss.read_index(FAISS_INDEX_CACHE_PATH)
    print(f"FAISS index loaded with {index.ntotal} recipes.")
else:
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"Saving FAISS index to cache: {FAISS_INDEX_CACHE_PATH}")
    faiss.write_index(index, FAISS_INDEX_CACHE_PATH)
    print(f"FAISS index built with {index.ntotal} recipes.")


# ======== Step 4: Query retrieval ========

if USE_LLM_STRUCTURING:
    print("Structuring queries with LLM...")
    queries = process_queries_with_llm(queries, model="gpt-4o-mini", use_llm=True)
    print(queries[0])
    print(f"Processed {len(queries)} queries with LLM.")

results = []
for q in tqdm(queries, desc="Retrieving"):
    q_emb = model.encode(q["query"], convert_to_numpy=True)
    q_emb = np.expand_dims(q_emb, axis=0)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, TOP_K)

    matched = [
        {
            "rank": int(rank + 1),
            "score": float(D[0][rank]),
            "recipe": recipes[int(I[0][rank])]
        }
        for rank in range(TOP_K)
    ]
    result_entry = {
        "query_id": q["id"],
        "query": q["query"],
        "results": matched
    }
    # Include original query if it was structured by LLM
    if "original_query" in q:
        result_entry["original_query"] = q["original_query"]
    results.append(result_entry)

# ======== Step 5: Save ========
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Retrieval results saved to {OUTPUT_PATH}")
