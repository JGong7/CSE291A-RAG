"""
Advanced RAG Technique: Smart Query Expansion for Semantic Understanding
Milestone 2.2 - Handles queries like "healthy", "spicy", "traditional Irish"
Enhanced with constraint-aware expansion to avoid diluting precise queries
"""

import json
import faiss
import numpy as np
import torch
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import re

# Import from hybrid retrieval
import sys
sys.path.append('.')
from hybrid_retrieval import MetadataExtractor, MetadataFilter, HybridRetriever
from quantity_filter import QuantityFilter

# ======== Config ========
DATA_1_PATH = "RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "manual_queries.json"
OUTPUT_PATH = "retrieval_results/advanced_results.json"

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


class QueryExpander:
    """
    Expands semantic queries with domain-specific knowledge
    Handles: "healthy", "spicy", "traditional", "quick", "special occasion"
    """
    
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
        """
        Determine if query should be expanded
        
        DO NOT expand if (PRIORITY ORDER):
        1. Has quantity constraints ("I only have X") - HIGHEST PRIORITY
        2. Has multiple specific ingredients (complex query)
        3. Is already very specific
        
        DO expand if:
        - Has semantic adjectives (healthy, spicy, traditional)
        - Is open-ended
        """
        query_lower = query.lower()
        
        # PRIORITY 1: Don't expand if has quantity constraints
        if re.search(r'(only have|have)\s+\w+\s+\w+', query_lower):
            print(f"  Skipping expansion: Query has quantity constraint")
            return False
        
        # PRIORITY 2: Don't expand if has specific dietary + ingredient constraints
        dietary_count = sum(1 for word in ['vegan', 'vegetarian', 'lactose', 'gluten-free', 'low-carb'] 
                          if word in query_lower)
        specific_ingredients = ['chicken', 'beef', 'tofu', 'shrimp', 'pasta', 'coconut milk', 'potato']
        ingredient_count = sum(1 for word in specific_ingredients if word in query_lower)
        
        if dietary_count >= 1 and ingredient_count >= 1:
            print(f"  Skipping expansion: Query has specific dietary + ingredient constraints")
            return False
        
        # PRIORITY 3: Don't expand if has many specific ingredients (>= 3)
        if ingredient_count >= 3:
            print(f"  Skipping expansion: Query has {ingredient_count} specific ingredients")
            return False
        
        # PRIORITY 4: Don't expand if has many commas (complex multi-part query)
        if query.count(',') >= 3:
            print(f"  Skipping expansion: Complex multi-part query")
            return False
        
        # DO expand if has semantic adjectives WITHOUT hard constraints
        semantic_words = [
            'traditional', 'authentic', 'classic',  # Cuisine style
            'spicy', 'mild', 'hot',                 # Flavor (broad)
            'healthy', 'light', 'fresh',            # Health (broad)
            'special', 'birthday', 'celebration',   # Occasion
            # NOTE: Removed 'simple', 'easy', 'quick' - too generic
        ]
        has_semantic = any(word in query_lower for word in semantic_words)
        
        if has_semantic:
            print(f"  Enabling expansion: Query has semantic adjective")
        
        return has_semantic
    
    @staticmethod
    def expand_query(query: str) -> str:
        """
        Expand query with relevant domain terms
        ONLY if should_expand_query returns True
        Returns: expanded query string
        """
        # Check if we should expand
        if not QueryExpander.should_expand_query(query):
            return query
        
        query_lower = query.lower()
        expanded_terms = []
        
        # Check for each expandable term
        for key, expansions in QueryExpander.EXPANSIONS.items():
            if key in query_lower:
                # Add subset of expansions (not all, to avoid dilution)
                expanded_terms.extend(expansions[:5])
        
        if expanded_terms:
            # Add expansion to original query
            expansion_str = ' '.join(set(expanded_terms))  # Remove duplicates
            return f"{query} {expansion_str}"
        
        return query
    
    @staticmethod
    def get_expansion_info(query: str) -> Dict[str, Any]:
        """Get information about what was expanded"""
        query_lower = query.lower()
        expansions_used = {}
        
        for key, expansions in QueryExpander.EXPANSIONS.items():
            if key in query_lower:
                expansions_used[key] = expansions[:5]
        
        return expansions_used


class AdvancedRetriever(HybridRetriever):
    """
    Advanced retriever with query expansion
    Combines: Metadata filtering + Query expansion + Vector search
    """
    
    def __init__(self, recipes: List[Dict], model: SentenceTransformer):
        super().__init__(recipes, model)
        self.query_expander = QueryExpander()
    
    def retrieve_with_expansion(
        self, 
        query: str, 
        top_k: int = 5, 
        use_metadata_filter: bool = True,
        use_query_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve with query expansion
        Returns results with expansion information
        """
        
        # Get expansion info
        expansion_info = self.query_expander.get_expansion_info(query)
        
        # Expand query if applicable
        if use_query_expansion and expansion_info:
            expanded_query = self.query_expander.expand_query(query)
            print(f"  Expanded query: {query} → {expanded_query[:100]}...")
        else:
            expanded_query = query
        
        # Use parent class retrieve method with expanded query
        # Parse requirements from ORIGINAL query (not expanded)
        requirements = MetadataFilter.parse_query_requirements(query)
        
        # Filter candidates based on metadata
        if use_metadata_filter:
            candidate_indices = [
                i for i, metadata in enumerate(self.metadata_list)
                if MetadataFilter.matches_requirements(metadata, requirements)
            ]
            
            if len(candidate_indices) == 0:
                print(f"  Warning: No metadata matches. Using full search.")
                candidate_indices = list(range(len(self.recipes)))
        else:
            candidate_indices = list(range(len(self.recipes)))
        
        # Build FAISS index for candidates
        candidate_embeddings = self.embeddings[candidate_indices]
        
        dim = candidate_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(candidate_embeddings)
        
        # Encode EXPANDED query and search
        query_embedding = self.model.encode(expanded_query, convert_to_numpy=True)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k, len(candidate_indices))
        D, I = index.search(query_embedding, k)
        
        # Map back to original indices
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
        
        return {
            'original_query': query,
            'expanded_query': expanded_query if use_query_expansion else query,
            'expansion_used': expansion_info,
            'metadata_filtering': use_metadata_filter,
            'candidates_after_filtering': len(candidate_indices),
            'results': results
        }


def main():
    """Main pipeline for advanced retrieval"""
    
    # ======== Step 1: Load data ========
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
    
    print(f"Loaded {len(recipes)} total recipes ({len(data1)} from RecipeNLG, {len(data2)} from Spoonacular).")
    
    # Load queries
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    # ======== Step 2: Initialize model and retriever ========
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)
    
    print("Initializing advanced retriever with query expansion...")
    retriever = AdvancedRetriever(recipes, model)
    
    # ======== Step 3: Retrieve for all queries ========
    results = []
    for q in tqdm(queries, desc="Retrieving with expansion"):
        retrieval_result = retriever.retrieve_with_expansion(
            q["query"], 
            top_k=TOP_K, 
            use_metadata_filter=True,
            use_query_expansion=True
        )
        
        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "expanded_query": retrieval_result['expanded_query'],
            "expansion_info": retrieval_result['expansion_used'],
            "candidates_after_filtering": retrieval_result['candidates_after_filtering'],
            "results": retrieval_result['results']
        })
    
    # ======== Step 4: Save results ========
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAdvanced retrieval results saved to {OUTPUT_PATH}")
    
    # Print example
    print("\n" + "="*80)
    print("Example retrieval result with query expansion:")
    print("="*80)
    if results:
        # Find a query that was expanded
        example = None
        for r in results:
            if r['expansion_info']:
                example = r
                break
        
        if not example:
            example = results[0]
        
        print(f"Query: {example['query']}")
        if example['expansion_info']:
            print(f"Expanded with: {example['expansion_info']}")
        print(f"Expanded query: {example['expanded_query'][:150]}...")
        print(f"Candidates after filtering: {example['candidates_after_filtering']}")
        print(f"\nTop result:")
        top_result = example['results'][0]
        print(f"  Title: {top_result['recipe']['title']}")
        print(f"  Score: {top_result['score']:.4f}")
        print(f"  Metadata: {top_result['metadata']}")


if __name__ == "__main__":
    main()
