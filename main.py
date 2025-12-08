import json
import faiss
import numpy as np
import torch
import random
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llm_query_processor import process_queries_with_llm
from llm_reranker import llm_rerank_results
from hybrid_retrieval import HybridRetriever  # Import hybrid retriever
import time    #Add timer
# ======== Config ========
DATA_1_PATH = "RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "manual_queries.json"
OUTPUT_PATH = "retrieval_results/hybrid_retrieval_result.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_PATH = "cache/model"
EMBEDDINGS_CACHE_PATH = "cache/embeddings_cache.npy"
FAISS_INDEX_CACHE_PATH = "cache/faiss_index_cache.bin"
METADATA_CACHE_PATH = "cache/metadata_cache.json"  # Cache for recipe metadata
USE_LLM_STRUCTURING = False # Set to False to skip LLM query structuring

SEED = 42

# ===== Hybrid Retrieval Config =====
USE_HYBRID_RETRIEVAL = True     # Use hybrid retrieval (metadata filtering + vector search)
USE_METADATA_FILTER = True      # Enable metadata filtering
USE_QUANTITY_FILTER = True      # Enable quantity filtering

# ===== Two-stage Retrieval Config =====
USE_LLM_RERANK = False       # Turn re-ranking on/off
FIRST_STAGE_TOP_K = 10      # FAISS retrieval
SECOND_STAGE_TOP_K = 5      # LLM rerank output
LLM_RERANK_MODEL = "gpt-4o-mini"   # Model for reranking

TOP_K = FIRST_STAGE_TOP_K if USE_LLM_RERANK else SECOND_STAGE_TOP_K  # Number of items to retrieve from FAISS
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

start_time = time.time() # Start timer

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

# ======== Step 2.5: Report Using Model, Config ========
print(f"Using embedding model: {EMBED_MODEL}")
print(f"Using LLM structuring: {USE_LLM_STRUCTURING}")
print(f"Current Top K for retrieval: {TOP_K}")
print(f"Using hybrid retrieval: {USE_HYBRID_RETRIEVAL}")
print(f"- Metadata filtering: {USE_METADATA_FILTER}" if USE_HYBRID_RETRIEVAL else "")
print(f"- Quantity filtering: {USE_QUANTITY_FILTER}" if USE_HYBRID_RETRIEVAL else "")
print(f"Using two-stage retrieval: {USE_LLM_RERANK}")
print(f"- First Stage Top K: {FIRST_STAGE_TOP_K}" if USE_LLM_RERANK else "")
print(f"- Second Stage Top K: {SECOND_STAGE_TOP_K}" if USE_LLM_RERANK else "")
print(f"- LLM Rerank Model: {LLM_RERANK_MODEL}" if USE_LLM_RERANK else "")

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

# ===== Initialize Hybrid Retriever (if enabled) =====
if USE_HYBRID_RETRIEVAL:
    print("Initializing Hybrid Retriever...")
    
    # Load or extract metadata
    metadata_list = None
    if os.path.exists(METADATA_CACHE_PATH):
        print(f"Loading metadata from cache: {METADATA_CACHE_PATH}")
        with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
            metadata_cache = json.load(f)
            # Convert sets back from lists (JSON doesn't support sets)
            metadata_list = []
            for m in metadata_cache:
                m['dietary_tags'] = set(m['dietary_tags'])
                m['dish_types'] = set(m['dish_types'])
                m['cooking_methods'] = set(m['cooking_methods'])
                m['contains_ingredients'] = set(m['contains_ingredients'])
                m['excludes_ingredients'] = set(m['excludes_ingredients'])
                metadata_list.append(m)
        print(f"Loaded metadata for {len(metadata_list)} recipes")
    
    # Pass pre-computed embeddings and metadata to avoid re-processing
    hybrid_retriever = HybridRetriever(recipes, model, embeddings=embeddings, metadata_list=metadata_list)
    
    # Save metadata cache if it was newly extracted
    if metadata_list is None and hasattr(hybrid_retriever, 'metadata_list'):
        print(f"Saving metadata to cache: {METADATA_CACHE_PATH}")
        # Convert sets to lists for JSON serialization
        metadata_to_save = []
        for m in hybrid_retriever.metadata_list:
            m_copy = m.copy()
            m_copy['dietary_tags'] = list(m['dietary_tags'])
            m_copy['dish_types'] = list(m['dish_types'])
            m_copy['cooking_methods'] = list(m['cooking_methods'])
            m_copy['contains_ingredients'] = list(m['contains_ingredients'])
            m_copy['excludes_ingredients'] = list(m['excludes_ingredients'])
            metadata_to_save.append(m_copy)
        
        with open(METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        print(f"Metadata cache saved")

results = []
for q in tqdm(queries, desc="Retrieving"):
    
    if USE_HYBRID_RETRIEVAL:
        # Use hybrid retrieval (metadata filtering + quantity filtering + vector search)
        stage1_items = hybrid_retriever.retrieve(
            query=q["query"],
            top_k=TOP_K,
            use_metadata_filter=USE_METADATA_FILTER,
            use_quantity_filter=USE_QUANTITY_FILTER
        )
    else:
        # Use original FAISS vector retrieval
        q_emb = model.encode(q["query"], convert_to_numpy=True)
        q_emb = np.expand_dims(q_emb, axis=0)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, TOP_K)

        stage1_items = [
            {
                "rank": int(rank + 1),
                "score": float(D[0][rank]),
                "recipe": recipes[int(I[0][rank])]
            }
            for rank in range(TOP_K)
        ]

    # print(stage1_items[0])
    # ===== Two-stage retrieval with LLM reranking =====
    if USE_LLM_RERANK:
        final_items = llm_rerank_results(
            q["query"],
            stage1_items,
            top_k=SECOND_STAGE_TOP_K,
            model="gpt-4o-mini"
        )
    else:
        # Just take top 5 if LLM reranking is disabled
        final_items = stage1_items[:SECOND_STAGE_TOP_K]

    # Build result entry
    result_entry = {
        "query_id": q["id"],
        "query": q["query"],
        "results": final_items
    }

    # Keep original query if structured by LLM
    if "original_query" in q:
        result_entry["original_query"] = q["original_query"]

    results.append(result_entry)



# ======== Step 5: Save ========
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Retrieval results saved to {OUTPUT_PATH}")
# ======== TIMER OUTPUT ========
end_time = time.time()
elapsed = end_time - start_time
print(f"Total retrieval pipeline time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")