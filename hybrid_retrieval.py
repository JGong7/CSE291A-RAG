"""
Hybrid Retrieval System for Recipe RAG
Implements metadata filtering + vector search for improved retrieval quality
Enhanced with ingredient quantity constraints
"""

import json
import faiss
import numpy as np
import torch
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Set
import re

# Import quantity filter
from quantity_filter import QuantityFilter

# ======== Config ========
DATA_1_PATH = "RecipeNLG_dataset/recipes_nlg_clean.json"
DATA_2_PATH = "Spoonacular_API/spoonacular_dataset.json"
QUERIES_PATH = "manual_queries.json"
OUTPUT_PATH = "retrieval_results/hybrid_results.json"

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


class MetadataExtractor:
    """Extract and normalize metadata from recipes for filtering"""
    
    # Common dietary restrictions and their indicators
    DIETARY_PATTERNS = {
        'vegan': ['vegan'],
        'vegetarian': ['vegetarian'],
        'lactose-free': ['lactose-free', 'lactose free', 'dairy-free', 'dairy free'],
        'gluten-free': ['gluten-free', 'gluten free'],
        'nut-free': ['nut-free', 'nut free'],
    }
    
    # Ingredient categories for filtering
    INGREDIENT_EXCLUDES = {
        'lactose': ['milk', 'cheese', 'cream', 'butter', 'yogurt', 'cream cheese', 'sour cream', 
                    'parmesan', 'mozzarella', 'cheddar', 'ricotta', 'whey'],
        'gluten': ['flour', 'wheat', 'bread', 'pasta', 'noodles', 'soy sauce'],
        'meat': ['beef', 'pork', 'chicken', 'turkey', 'lamb', 'steak', 'bacon', 'ham', 
                 'sausage', 'meat'],
        'seafood': ['fish', 'salmon', 'tuna', 'shrimp', 'crab', 'lobster', 'shellfish'],
        'nuts': ['peanut', 'almond', 'walnut', 'cashew', 'pistachio', 'pecan', 'hazelnut'],
        'eggs': ['egg', 'eggs'],
    }
    
    # Dish type patterns
    DISH_TYPES = {
        'breakfast': ['breakfast', 'pancake', 'waffle', 'omelet', 'omelette', 'muffin'],
        'lunch': ['lunch', 'sandwich', 'wrap', 'salad'],
        'dinner': ['dinner', 'main', 'entree'],
        'dessert': ['dessert', 'cake', 'cookie', 'pie', 'brownie', 'pudding', 'ice cream', 
                    'chocolate', 'sweet'],
        'soup': ['soup', 'stew', 'chowder', 'broth'],
        'pasta': ['pasta', 'spaghetti', 'linguine', 'fettuccine', 'penne', 'noodle', 'lasagna'],
        'salad': ['salad'],
        'appetizer': ['appetizer', 'starter', 'snack'],
    }
    
    # Cooking methods
    COOKING_METHODS = {
        'baked': ['bake', 'baked', 'baking', 'oven', 'roast', 'roasted'],
        'grilled': ['grill', 'grilled', 'grilling', 'barbecue', 'bbq'],
        'fried': ['fry', 'fried', 'frying', 'pan-fry'],
        'boiled': ['boil', 'boiled', 'boiling'],
        'slow-cooked': ['slow cooker', 'crock pot', 'crockpot'],
    }
    
    @staticmethod
    def extract_metadata(recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a recipe"""
        title = recipe.get('title', '').lower()
        ingredients_list = recipe.get('ingredients', [])
        ingredients = ' '.join([str(i).lower() for i in ingredients_list])
        
        # Get directions/instructions
        directions = recipe.get('directions', recipe.get('instructions', ''))
        if isinstance(directions, list):
            directions = ' '.join(directions).lower()
        else:
            directions = str(directions).lower()
        
        # Get NER if available
        ner = recipe.get('ner', [])
        ner_text = ' '.join([str(n).lower() for n in ner])
        
        # Combine all text for analysis
        full_text = f"{title} {ingredients} {directions} {ner_text}"
        
        metadata = {
            'title': recipe.get('title', ''),
            'ingredients_list': ingredients_list,
            'ingredients_text': ingredients,
            'directions': directions,
            'ner': ner,
            'source': recipe.get('source', ''),
            'dietary_tags': set(),
            'dish_types': set(),
            'cooking_methods': set(),
            'contains_ingredients': set(),
            'excludes_ingredients': set(),
        }
        
        # Extract dietary tags
        for diet, patterns in MetadataExtractor.DIETARY_PATTERNS.items():
            if any(pattern in full_text for pattern in patterns):
                metadata['dietary_tags'].add(diet)
        
        # Check for ingredient exclusions (e.g., lactose-free if no dairy)
        for category, items in MetadataExtractor.INGREDIENT_EXCLUDES.items():
            has_item = any(item in ingredients or item in ner_text for item in items)
            if not has_item and category == 'lactose':
                metadata['dietary_tags'].add('lactose-free')
            if not has_item and category == 'gluten':
                metadata['dietary_tags'].add('gluten-free')
            if has_item and category == 'meat':
                metadata['contains_ingredients'].add('meat')
            if has_item and category == 'seafood':
                metadata['contains_ingredients'].add('seafood')
        
        # Check if vegan/vegetarian based on ingredients
        has_meat = any(item in ingredients or item in ner_text 
                      for item in MetadataExtractor.INGREDIENT_EXCLUDES['meat'])
        has_seafood = any(item in ingredients or item in ner_text 
                         for item in MetadataExtractor.INGREDIENT_EXCLUDES['seafood'])
        has_eggs = any(item in ingredients or item in ner_text 
                      for item in MetadataExtractor.INGREDIENT_EXCLUDES['eggs'])
        has_dairy = any(item in ingredients or item in ner_text 
                       for item in MetadataExtractor.INGREDIENT_EXCLUDES['lactose'])
        
        if not has_meat and not has_seafood and not has_eggs and not has_dairy:
            metadata['dietary_tags'].add('vegan')
        if not has_meat and not has_seafood:
            metadata['dietary_tags'].add('vegetarian')
        
        # Extract dish types
        for dish_type, patterns in MetadataExtractor.DISH_TYPES.items():
            if any(pattern in title or pattern in full_text for pattern in patterns):
                metadata['dish_types'].add(dish_type)
        
        # Extract cooking methods
        for method, patterns in MetadataExtractor.COOKING_METHODS.items():
            if any(pattern in directions for pattern in patterns):
                metadata['cooking_methods'].add(method)
        
        # Extract specific ingredients mentioned
        for ingredient in ingredients_list:
            ing_lower = str(ingredient).lower()
            # Extract key ingredients (simple heuristic)
            words = re.findall(r'\b[a-z]{3,}\b', ing_lower)
            for word in words:
                if word not in ['cup', 'cups', 'tablespoon', 'teaspoon', 'ounce', 
                               'pound', 'gram', 'the', 'and', 'or']:
                    metadata['contains_ingredients'].add(word)
        
        return metadata


class MetadataFilter:
    """Filter recipes based on metadata criteria"""
    
    @staticmethod
    def parse_query_requirements(query: str) -> Dict[str, Any]:
        """Parse query to extract filtering requirements"""
        query_lower = query.lower()
        
        requirements = {
            'dietary_requirements': set(),
            'required_ingredients': set(),
            'excluded_ingredients': set(),
            'dish_type': set(),
            'cooking_method': set(),
        }
        
        # Detect dietary requirements
        if 'vegan' in query_lower:
            requirements['dietary_requirements'].add('vegan')
        if 'vegetarian' in query_lower:
            requirements['dietary_requirements'].add('vegetarian')
        if 'lactose' in query_lower or 'dairy-free' in query_lower or 'lactose intolerance' in query_lower:
            requirements['dietary_requirements'].add('lactose-free')
            requirements['excluded_ingredients'].update(['milk', 'cheese', 'cream', 'butter', 'yogurt'])
        if 'gluten-free' in query_lower or 'gluten' in query_lower:
            requirements['dietary_requirements'].add('gluten-free')
        if 'nut-free' in query_lower:
            requirements['dietary_requirements'].add('nut-free')
        
        # Detect dish types
        for dish_type, patterns in MetadataExtractor.DISH_TYPES.items():
            if any(pattern in query_lower for pattern in patterns):
                requirements['dish_type'].add(dish_type)
        
        # Detect cooking methods
        for method, patterns in MetadataExtractor.COOKING_METHODS.items():
            if any(pattern in query_lower for pattern in patterns):
                requirements['cooking_method'].add(method)
        
        # Extract specific ingredients mentioned
        # Simple pattern matching for common ingredients
        ingredient_patterns = {
            'egg': ['egg', 'eggs'],
            'cheese': ['cheese'],
            'pasta': ['pasta'],
            'tofu': ['tofu'],
            'potato': ['potato', 'potatoes'],
            'chicken': ['chicken'],
            'peanut': ['peanut', 'peanuts'],
            'chocolate': ['chocolate'],
            'coconut': ['coconut'],
            'banana': ['banana'],
            'yogurt': ['yogurt'],
            'shrimp': ['shrimp'],
            'garlic': ['garlic'],
        }
        
        for ingredient, patterns in ingredient_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                requirements['required_ingredients'].add(ingredient)
        
        return requirements
    
    @staticmethod
    def matches_requirements(metadata: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Check if recipe metadata matches the requirements"""
        
        # Check dietary requirements
        for diet in requirements['dietary_requirements']:
            if diet not in metadata['dietary_tags']:
                return False
        
        # Check excluded ingredients
        for excluded in requirements['excluded_ingredients']:
            if excluded in metadata['ingredients_text']:
                return False
        
        # Check required ingredients (at least one should match if specified)
        if requirements['required_ingredients']:
            has_required = any(
                req in metadata['ingredients_text'] or req in metadata['contains_ingredients']
                for req in requirements['required_ingredients']
            )
            if not has_required:
                return False
        
        # Check dish type (at least one should match if specified)
        if requirements['dish_type']:
            if not requirements['dish_type'].intersection(metadata['dish_types']):
                return False
        
        # Check cooking method (at least one should match if specified)
        if requirements['cooking_method']:
            if not requirements['cooking_method'].intersection(metadata['cooking_methods']):
                return False
        
        return True


class HybridRetriever:
    """Hybrid retrieval combining metadata filtering and vector search"""
    
    def __init__(self, recipes: List[Dict], model: SentenceTransformer):
        self.recipes = recipes
        self.model = model
        
        # Extract metadata for all recipes
        print("Extracting metadata from recipes...")
        self.metadata_list = [MetadataExtractor.extract_metadata(r) for r in tqdm(recipes)]
        
        # Build embeddings
        print("Building embeddings...")
        self.texts = [self._build_text(r) for r in recipes]
        self.embeddings = model.encode(
            self.texts, 
            show_progress_bar=True, 
            batch_size=64, 
            convert_to_numpy=True
        )
        
        # Normalize embeddings
        faiss.normalize_L2(self.embeddings)
        
        print(f"Initialized retriever with {len(recipes)} recipes")
    
    def _build_text(self, recipe: Dict) -> str:
        """Build text representation of recipe for embedding"""
        title = recipe.get("title", "")
        ingredients = " ".join([str(i) for i in recipe.get("ingredients", [])])
        
        # Get instructions
        instructions = recipe.get("instructions", recipe.get("directions", ""))
        if isinstance(instructions, list):
            instructions = " ".join(instructions)
        
        return f"{title} {ingredients} {instructions}".strip()
    
    def retrieve(self, query: str, top_k: int = 5, use_metadata_filter: bool = True, use_quantity_filter: bool = True) -> List[Dict]:
        """
        Retrieve recipes using hybrid approach:
        1. Parse query for metadata requirements
        2. Filter candidates based on metadata
        3. Filter candidates based on quantity constraints (NEW)
        4. Perform vector search on filtered candidates
        """
        
        # Parse query requirements
        requirements = MetadataFilter.parse_query_requirements(query)
        
        # Filter candidates based on metadata
        if use_metadata_filter:
            candidate_indices = [
                i for i, metadata in enumerate(self.metadata_list)
                if MetadataFilter.matches_requirements(metadata, requirements)
            ]
            
            print(f"Metadata filtering: {len(candidate_indices)} / {len(self.recipes)} recipes matched")
            
            if len(candidate_indices) == 0:
                print("Warning: No recipes matched metadata filters. Falling back to full search.")
                candidate_indices = list(range(len(self.recipes)))
        else:
            candidate_indices = list(range(len(self.recipes)))
        
        # NEW: Apply quantity filtering
        if use_quantity_filter:
            before_quantity = len(candidate_indices)
            candidate_indices = QuantityFilter.filter_recipes_by_quantity(
                self.recipes,
                candidate_indices,
                query,
                tolerance=0.0  # Strict: must have enough ingredients
            )
            after_quantity = len(candidate_indices)
            
            if after_quantity < before_quantity:
                print(f"Quantity filtering: {after_quantity} / {before_quantity} recipes satisfy constraints")
            
            # Fallback if no recipes satisfy quantity constraints
            if len(candidate_indices) == 0:
                print("Warning: No recipes matched quantity constraints. Relaxing filter...")
                # Retry with some tolerance
                candidate_indices = QuantityFilter.filter_recipes_by_quantity(
                    self.recipes,
                    list(range(len(self.recipes))),
                    query,
                    tolerance=1.0  # Allow recipes needing 1 more ingredient
                )
                if len(candidate_indices) == 0:
                    # Still nothing? Fall back to metadata-only results
                    if use_metadata_filter:
                        candidate_indices = [
                            i for i, metadata in enumerate(self.metadata_list)
                            if MetadataFilter.matches_requirements(metadata, requirements)
                        ]
                    else:
                        candidate_indices = list(range(len(self.recipes)))
        
        # Build FAISS index for candidates
        candidate_embeddings = self.embeddings[candidate_indices]
        
        dim = candidate_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(candidate_embeddings)
        
        # Encode query and search
        query_embedding = self.model.encode(query, convert_to_numpy=True)
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
        
        return results


def main():
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
    
    retriever = HybridRetriever(recipes, model)
    
    # ======== Step 3: Retrieve for all queries ========
    results = []
    for q in tqdm(queries, desc="Retrieving"):
        matched = retriever.retrieve(q["query"], top_k=TOP_K, use_metadata_filter=True)
        
        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "results": matched
        })
    
    # ======== Step 4: Save results ========
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Hybrid retrieval results saved to {OUTPUT_PATH}")
    
    # Print example
    print("\n" + "="*80)
    print("Example retrieval result:")
    print("="*80)
    if results:
        example = results[0]
        print(f"Query: {example['query']}")
        print(f"\nTop result:")
        top_result = example['results'][0]
        print(f"  Title: {top_result['recipe']['title']}")
        print(f"  Score: {top_result['score']:.4f}")
        print(f"  Metadata: {top_result['metadata']}")


if __name__ == "__main__":
    main()
