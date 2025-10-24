import json
import faiss
import numpy as np
import torch
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ======== Config ========
DATA_1_PATH = "RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "manual_queries.json"
OUTPUT_PATH = "retrieval_results/faiss_fusion_results.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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
print("Loading datasets...")

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

model = SentenceTransformer(EMBED_MODEL)

# ======== Step 2: Build embeddings ========
print("Encoding all recipes...")

def build_text(r):
    title = r.get("title", "")
    ingredients = " ".join(r.get("ingredients", []))
    instructions = r.get("instructions", "")
    return f"{title} {ingredients} {instructions}".strip()

texts = [build_text(r) for r in recipes]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)

dim = embeddings.shape[1]
print(f"Embedding dimension: {dim}")

# ======== Step 3: Build FAISS index ========
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} recipes.")

# ======== Step 4: Query retrieval ========
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
    results.append({
        "query_id": q["id"],
        "query": q["query"],
        "results": matched
    })

# ======== Step 5: Save ========
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Retrieval results saved to {OUTPUT_PATH}")
