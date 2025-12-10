import json
import faiss
import numpy as np
import torch
import random
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llm_query_processor import process_queries_with_llm
from llm_query_processor import rewrite_query_to_structured, rewrite_query_text
from llm_reranker import llm_rerank_results
from hybrid_retrieval import HybridRetriever
from hybrid_advanced_retrieval import AdvancedRetriever, RetrievalConfig
import time    #Add timer
# ======== Config ========
DATA_1_PATH = "dataset/RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "dataset/Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "dataset/manual_queries.json"
OUTPUT_PATH = "retrieval_results/hybrid_retrieval_advanced_LLM_Rerank_result.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_PATH = "cache/model"
EMBEDDINGS_CACHE_PATH = "cache/embeddings_cache.npy"
FAISS_INDEX_CACHE_PATH = "cache/faiss_index_cache.bin"
METADATA_CACHE_PATH = "cache/metadata_cache.json"  # Cache for recipe metadata
USE_LLM_STRUCTURING = False # Set to False to skip LLM query structuring
USE_QUERY_REWRITING = True  # New: use LLM to rewrite/structure queries at retrieval time

SEED = 42

# ===== Hybrid Retrieval Config =====
USE_HYBRID_RETRIEVAL = True     # Use hybrid retrieval (metadata filtering + vector search)
HYBRID_MODE = "advanced"        # "old" -> HybridRetriever, "advanced" -> AdvancedRetriever
USE_METADATA_FILTER = True      # Enable metadata filtering
USE_QUANTITY_FILTER = True      # Enable quantity filtering

# ===== Two-stage Retrieval Config =====
USE_LLM_RERANK = True       # Turn re-ranking on/off
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
    print("Structuring queries with LLM (offline pre-processing)...")
    queries = process_queries_with_llm(queries, model="gpt-4o-mini", use_llm=True)
    print(queries[0])
    print(f"Processed {len(queries)} queries with LLM.")

# Small demo: show original vs structured vs rewritten for first few queries
# if USE_QUERY_REWRITING and queries:
#     demo_n = min(1, len(queries))
#     print("\n[Query Rewriting Demo] Showing first", demo_n, "queries:")
#     for i in range(demo_n):
#         q_demo = queries[i]["query"]
#         structured_demo = rewrite_query_to_structured(q_demo, llm_model="gpt-4o-mini", use_llm=True)
#         rewritten_demo = rewrite_query_text(q_demo, structured_demo)
#         print(f"Original Query: {q_demo}")
#         print(f"Structured Query: {structured_demo}")
#         print(f"Rewritten Query: {rewritten_demo}")

# ===== Initialize Hybrid Retriever (if enabled) =====
if USE_HYBRID_RETRIEVAL:
    print(f"Hybrid mode: {HYBRID_MODE}")
    print("Initializing hybrid retriever(s)...")

    # Load or extract metadata
    metadata_list = None
    if os.path.exists(METADATA_CACHE_PATH):
        print(f"Loading metadata from cache: {METADATA_CACHE_PATH}")
        with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
            metadata_cache = json.load(f)
            metadata_list = []
            for m in metadata_cache:
                m["dietary_tags"] = set(m["dietary_tags"])
                m["dish_types"] = set(m["dish_types"])
                m["cooking_methods"] = set(m["cooking_methods"])
                m["contains_ingredients"] = set(m["contains_ingredients"])
                m["excludes_ingredients"] = set(m["excludes_ingredients"])
                metadata_list.append(m)
        print(f"Loaded metadata for {len(metadata_list)} recipes")

    advanced_retriever = None
    hybrid_retriever = None

    if HYBRID_MODE == "advanced":
        print("Using AdvancedRetriever (BM25 + dense + fusion)...")
        advanced_retriever = AdvancedRetriever(
            recipes=recipes,
            model=model,
            embeddings=embeddings,
            metadata_list=metadata_list,
            enable_cache=True,
        )

        # Save metadata cache if it was newly extracted
        if metadata_list is None and hasattr(advanced_retriever, "metadata_list"):
            print(f"Saving metadata to cache: {METADATA_CACHE_PATH}")
            metadata_to_save = []
            for m in advanced_retriever.metadata_list:
                m_copy = m.copy()
                m_copy["dietary_tags"] = list(m["dietary_tags"])
                m_copy["dish_types"] = list(m["dish_types"])
                m_copy["cooking_methods"] = list(m["cooking_methods"])
                m_copy["contains_ingredients"] = list(m["contains_ingredients"])
                m_copy["excludes_ingredients"] = list(m["excludes_ingredients"])
                m_copy["general_ingredient_tags"] = list(m["general_ingredient_tags"])
                metadata_to_save.append(m_copy)

            with open(METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
            print("Metadata cache saved")

    elif HYBRID_MODE == "old":
        print("Using original HybridRetriever (metadata + quantity + dense)...")
        hybrid_retriever = HybridRetriever(
            recipes,
            model,
            embeddings=embeddings,
            metadata_list=metadata_list,
        )

        if metadata_list is None and hasattr(hybrid_retriever, "metadata_list"):
            print(f"Saving metadata to cache: {METADATA_CACHE_PATH}")
            metadata_to_save = []
            for m in hybrid_retriever.metadata_list:
                m_copy = m.copy()
                m_copy["dietary_tags"] = list(m["dietary_tags"])
                m_copy["dish_types"] = list(m["dish_types"])
                m_copy["cooking_methods"] = list(m["cooking_methods"])
                m_copy["contains_ingredients"] = list(m["contains_ingredients"])
                m_copy["excludes_ingredients"] = list(m["excludes_ingredients"])
                m_copy["general_ingredient_tags"] = list(m["general_ingredient_tags"])
                metadata_to_save.append(m_copy)

            with open(METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
            print("Metadata cache saved")

results = []
for q in tqdm(queries, desc="Retrieving"):

    user_query = q["query"]

    # Optional online query rewriting
    if USE_QUERY_REWRITING:
        structured = rewrite_query_to_structured(user_query, llm_model="gpt-4o-mini", use_llm=True)
        rewritten_query = rewrite_query_text(user_query, structured)
    else:
        structured = None
        rewritten_query = user_query

    if USE_HYBRID_RETRIEVAL and HYBRID_MODE == "advanced":
        # Advanced hybrid retrieval (metadata + quantity + dense + sparse + fusion)
        cfg = RetrievalConfig()
        cfg.final_top_k = TOP_K
        cfg.use_metadata_filter = USE_METADATA_FILTER
        cfg.use_quantity_filter = USE_QUANTITY_FILTER
        if USE_QUERY_REWRITING:
            adv_result = advanced_retriever.retrieve(query=rewritten_query, config=cfg, structured_query=structured)
        else:
            adv_result = advanced_retriever.retrieve(query=rewritten_query, config=cfg)

        stage1_items = [
            {
                "rank": item.rank,
                "score": float(item.score),
                "recipe": recipes[item.recipe_index],
                "source": item.source,
            }
            for item in adv_result.items
        ]

    elif USE_HYBRID_RETRIEVAL and HYBRID_MODE == "old":
        # Original HybridRetriever behaviour
        stage1_items = hybrid_retriever.retrieve(
            rewritten_query,
            top_k=TOP_K,
            use_metadata_filter=USE_METADATA_FILTER,
            use_quantity_filter=USE_QUANTITY_FILTER,
        )

    else:
        # Original FAISS vector retrieval baseline
        q_emb = model.encode(rewritten_query, convert_to_numpy=True)
        q_emb = np.expand_dims(q_emb, axis=0)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, TOP_K)

        stage1_items = [
            {
                "rank": int(rank + 1),
                "score": float(D[0][rank]),
                "recipe": recipes[int(I[0][rank])],
            }
            for rank in range(TOP_K)
        ]

    # ===== Two-stage retrieval with LLM reranking =====
    if USE_LLM_RERANK:
        final_items = llm_rerank_results(
            rewritten_query,
            stage1_items,
            top_k=SECOND_STAGE_TOP_K,
            model="gpt-4o-mini"
        )
    else:
        final_items = stage1_items[:SECOND_STAGE_TOP_K]

    # Build result entry
    result_entry = {
        "query_id": q["id"],
        "query": user_query,
        "rewritten_query": rewritten_query if USE_QUERY_REWRITING else "LLM Query Rewriting not used",
        "structured_query": structured if USE_QUERY_REWRITING else "LLM Query Structuring not used",
        "results": final_items
    }

    if USE_QUERY_REWRITING:
        result_entry["used_query_rewriting"] = True

    # Keep original query if structured by offline LLM pass
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