"""
Advanced RAG Technique: Embedding-Based Query Expansion + Reranking
Milestone 3.0 - Handles queries with domain vocabulary using semantic embeddings
Enhanced with constraint-aware metadata filtering, FAISS retrieval, and CrossEncoder reranking
"""

import json
import faiss
import numpy as np
import torch
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any
import re
import sys

# Import from hybrid retrieval
sys.path.append('.')
from hybrid_retrieval import MetadataExtractor, MetadataFilter, HybridRetriever
from quantity_filter import QuantityFilter

# ======== Config ========
DATA_1_PATH = "RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "manual_queries.json"
OUTPUT_PATH = "retrieval_results/embedding_results_reranked.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
RERANK_K = 25  # number of top FAISS results to rerank
SEED = 42

# ======== Reproducibility ========
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ======== Query Expander ========
class QueryExpander:
    EXPANSIONS = {
        # Health-related
        'healthy': [
            'nutritious', 'low-calorie', 'low-fat', 'wholesome', 'fresh', 
            'vegetables', 'lean', 'grilled', 'steamed', 'salad'
        ],
        'light': [
            'fresh', 'salad', 'vegetables', 'low-calorie', 'grilled'
        ],
        'low-carb': [
            'protein', 'vegetables', 'meat', 'fish', 'cauliflower', 'zucchini'
        ],
        
        # Flavor profiles
        'spicy': [
            'hot', 'chili', 'pepper', 'jalapeño', 'cayenne', 'sriracha',
            'curry', 'paprika', 'red pepper flakes'
        ],
        'sweet': [
            'sugar', 'honey', 'maple syrup', 'chocolate', 'vanilla',
            'caramel', 'fruit', 'berry'
        ],
        'savory': [
            'garlic', 'onion', 'herbs', 'umami', 'soy sauce', 'cheese'
        ],
        
        # Cuisine types
        'irish': [
            'potato', 'cabbage', 'beef', 'stew', 'soda bread', 'bacon',
            'colcannon', 'shepherd', 'dublin'
        ],
        'italian': [
            'pasta', 'tomato', 'basil', 'mozzarella', 'parmesan', 'olive oil',
            'garlic', 'oregano', 'pizza'
        ],
        'mexican': [
            'tortilla', 'beans', 'salsa', 'avocado', 'cilantro', 'lime',
            'chili', 'cumin', 'taco'
        ],
        'asian': [
            'soy sauce', 'ginger', 'garlic', 'rice', 'noodles', 'sesame',
            'stir-fry', 'wok'
        ],
        
        # Occasion-based
        'birthday': [
            'cake', 'celebration', 'chocolate', 'festive', 'special',
            'dessert', 'party', 'frosting'
        ],
        'party': [
            'appetizer', 'snack', 'finger food', 'dip', 'platter', 'crowd'
        ],
        'special occasion': [
            'elegant', 'gourmet', 'impressive', 'fancy', 'celebration'
        ],
        
        # Time/Complexity
        'quick': [
            'easy', 'fast', 'simple', '15 minute', '20 minute', 'no bake'
        ],
        'simple': [
            'easy', 'basic', 'beginner', 'straightforward', 'quick'
        ],
        
        # Weather/Season
        'cold weather': [
            'warm', 'hot', 'comfort', 'hearty', 'soup', 'stew', 'roasted'
        ],
        'summer': [
            'fresh', 'light', 'cold', 'refreshing', 'salad', 'grilled'
        ],
        
        # Texture/Preparation
        'crispy': [
            'fried', 'baked', 'crunchy', 'golden', 'breaded'
        ],
        'creamy': [
            'cream', 'milk', 'smooth', 'rich', 'sauce'
        ],
    }
    @staticmethod
    def should_expand_query(query: str) -> bool:
        query_lower = query.lower()
        if re.search(r'(only have|have)\s+\w+\s+\w+', query_lower):
            return False
        if query.count(',') >= 3:
            return False
        semantic_words = ['traditional', 'authentic', 'classic', 'spicy', 'healthy', 'special']
        return any(word in query_lower for word in semantic_words)

    @staticmethod
    def expand_query(query: str) -> str:
        if not QueryExpander.should_expand_query(query):
            return query
        query_lower = query.lower()
        expanded_terms = []
        for key, expansions in QueryExpander.EXPANSIONS.items():
            if key in query_lower:
                expanded_terms.extend(expansions[:5])
        if expanded_terms:
            expansion_str = ' '.join(set(expanded_terms))
            return f"{query} {expansion_str}"
        return query

    @staticmethod
    def get_expansion_info(query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        expansions_used = {}
        for key, expansions in QueryExpander.EXPANSIONS.items():
            if key in query_lower:
                expansions_used[key] = expansions[:5]
        return expansions_used

# ======== CrossEncoder Reranker ========
class CrossEncoderReranker:
    def __init__(self, model_name=CROSS_ENCODER_MODEL):
        print(f"Loading CrossEncoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        texts = []
        for c in candidates:
            doc_text = c["recipe"]["text"]  # Use precomputed text field
            texts.append((query, doc_text))
        scores = self.model.predict(texts)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

# ======== Advanced Retriever ========
class AdvancedRetriever(HybridRetriever):
    def __init__(self, recipes: List[Dict], model: SentenceTransformer):
        super().__init__(recipes, model)
        self.query_expander = QueryExpander()
        self.reranker = CrossEncoderReranker()
        self._preprocess_recipes_text()

    def _preprocess_recipes_text(self):
        # Create a 'text' field combining title + ingredients + directions
        for r in self.recipes:
            parts = [r.get("title", "")]
            ingredients = r.get("ingredients", [])
            parts.extend(ingredients)
            directions = r.get("directions", []) or r.get("instructions", [])
            if isinstance(directions, str):
                directions = [directions]
            parts.extend(directions)
            r["text"] = " ".join(parts)

    def retrieve_with_expansion(
        self, query: str, top_k: int = TOP_K,
        use_metadata_filter: bool = True,
        use_query_expansion: bool = True,
        rerank_top_k: int = RERANK_K
    ) -> Dict[str, Any]:

        # Query expansion
        expansion_info = self.query_expander.get_expansion_info(query)
        if use_query_expansion and expansion_info:
            expanded_query = self.query_expander.expand_query(query)
            print(f"  Expanded query: {query} → {expanded_query[:100]}...")
        else:
            expanded_query = query

        # Metadata filtering
        requirements = MetadataFilter.parse_query_requirements(query)
        if use_metadata_filter:
            candidate_indices = [
                i for i, metadata in enumerate(self.metadata_list)
                if MetadataFilter.matches_requirements(metadata, requirements)
            ]
            if not candidate_indices:
                print("  Warning: No metadata matches. Using full search.")
                candidate_indices = list(range(len(self.recipes)))
        else:
            candidate_indices = list(range(len(self.recipes)))

        # FAISS search
        candidate_embeddings = self.embeddings[candidate_indices]
        dim = candidate_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(candidate_embeddings)

        query_embedding = self.model.encode(expanded_query, convert_to_numpy=True)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(candidate_indices))
        D, I = index.search(query_embedding, k)

        # Collect results
        results = []
        for rank in range(k):
            original_idx = candidate_indices[I[0][rank]]
            results.append({
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "recipe": self.recipes[original_idx],
                "metadata": {
                    "dietary_tags": list(self.metadata_list[original_idx]['dietary_tags']),
                    "dish_types": list(self.metadata_list[original_idx]['dish_types']),
                    "cooking_methods": list(self.metadata_list[original_idx]['cooking_methods']),
                }
            })

        # Rerank top results
        reranked_results = self.reranker.rerank(query, results[:rerank_top_k])
        results[:rerank_top_k] = reranked_results

        return {
            'original_query': query,
            'expanded_query': expanded_query if use_query_expansion else query,
            'expansion_used': expansion_info,
            'metadata_filtering': use_metadata_filter,
            'candidates_after_filtering': len(candidate_indices),
            'results': results
        }

# ======== Main Pipeline ========
def main():
    print("Loading datasets...")

    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")
            return []

    data1 = load_json(DATA_1_PATH)
    data2 = load_json(DATA_2_PATH)
    recipes = data1 + data2
    print(f"Loaded {len(recipes)} recipes ({len(data1)} + {len(data2)})")

    # Load queries
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    # Initialize retriever
    print("Initializing advanced retriever with query expansion and reranking...")
    retriever = AdvancedRetriever(recipes, model)

    # Retrieve for all queries
    results = []
    for q in tqdm(queries, desc="Retrieving with expansion and reranking"):
        retrieval_result = retriever.retrieve_with_expansion(
            q["query"], top_k=TOP_K, use_metadata_filter=True, use_query_expansion=True, rerank_top_k=RERANK_K
        )
        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "expanded_query": retrieval_result['expanded_query'],
            "expansion_info": retrieval_result['expansion_used'],
            "candidates_after_filtering": retrieval_result['candidates_after_filtering'],
            "results": retrieval_result['results']
        })

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAdvanced retrieval results saved to {OUTPUT_PATH}")

    # Example print
    print("\n" + "="*80)
    print("Example retrieval result:")
    print("="*80)
    if results:
        example = next((r for r in results if r['expansion_info']), results[0])
        print(f"Query: {example['query']}")
        if example['expansion_info']:
            print(f"Expanded with: {example['expansion_info']}")
        print(f"Expanded query: {example['expanded_query'][:150]}...")
        print(f"Candidates after filtering: {example['candidates_after_filtering']}")
        top_result = example['results'][0]
        print(f"  Title: {top_result['recipe']['title']}")
        print(f"  Score: {top_result['score']:.4f}")
        if 'rerank_score' in top_result:
            print(f"  Rerank score: {top_result['rerank_score']:.4f}")
        print(f"  Metadata: {top_result['metadata']}")

if __name__ == "__main__":
    main()
