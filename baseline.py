import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

# ========== CONFIG ==========
DATASETS = {
    "RecipeNLG": "RecipeNLG_dataset/recipes_nlg_clean.json",
    "Spoonacular": "Spoonacular_API/spoonacular_dataset.json"
}
QUERIES_PATH = "manual_queries.json"
OUTPUT_DIR = "retrieval_results"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== STEP 1: Load Queries ==========
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)

model = SentenceTransformer(EMBED_MODEL)

# ========== STEP 2: Process Each Dataset Separately ==========
for dataset_name, data_path in DATASETS.items():
    print(f"\n🔹 Processing dataset: {dataset_name}")

    # ----- Load recipes -----
    with open(data_path, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"Loaded {len(recipes)} recipes from {data_path}")

    # ----- Normalize fields (some have 'directions', others 'instructions') -----
    def get_text(recipe):
        title = recipe.get("title", "")
        ingredients = " ".join(recipe.get("ingredients", []))
        instructions = " ".join(recipe.get("directions", [])) or recipe.get("instructions", "")
        return f"{title} {ingredients} {instructions}"

    # ----- Build embeddings -----
    print("Encoding recipes into embeddings...")
    texts = [get_text(r) for r in recipes]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)

    if len(embeddings.shape) != 2:
        raise ValueError(f"Embedding failed — shape is {embeddings.shape}, expected (N, D).")

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # ----- Build FAISS index -----
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    # ----- Run retrieval -----
    print("Running retrieval...")
    results = []
    for q in tqdm(queries, desc=f"Retrieving ({dataset_name})"):
        q_emb = model.encode(q["query"], convert_to_numpy=True)
        q_emb = np.expand_dims(q_emb, axis=0)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, TOP_K)

        matched = [
            {"rank": int(rank), "score": float(D[0][rank]), "recipe": recipes[int(I[0][rank])]}
            for rank in range(TOP_K)
        ]
        results.append({
            "dataset": dataset_name,
            "query_id": q["id"],
            "query": q["query"],
            "results": matched
        })

    # ----- Save results -----
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_faiss_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_path}")
