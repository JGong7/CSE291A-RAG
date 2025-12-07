"""
Nutrition Filter
Handles nutrition-based filtering (carb, protein, fat levels)
"""

import re
from typing import Dict, List, Any, Set


class NutritionFilter:
    """Extract and filter recipes based on nutritional requirements"""
    
    # High-carbohydrate ingredients (for nutrition filtering)
    HIGH_CARB_INGREDIENTS = {
        # Grains & Starches
        'rice', 'pasta', 'noodles', 'spaghetti', 'linguine', 'fettuccine', 'penne', 
        'macaroni', 'orzo', 'couscous', 'quinoa', 'barley', 'oats', 'oatmeal',
        'vermicelli', 'angel hair', 'lasagna', 'rigatoni', 'ziti', 'rotini',
        'farfalle', 'ravioli', 'tortellini', 'gnocchi', 'udon', 'ramen',
        
        # Flour & Baking
        'flour', 'wheat', 'all-purpose flour', 'whole wheat flour', 'bread flour',
        'cornmeal', 'cornstarch', 'breadcrumbs', 'panko',
        
        # Breads
        'bread', 'baguette', 'roll', 'bun', 'biscuit', 'croissant', 'tortilla',
        'pita', 'bagel', 'english muffin', 'cracker',
        
        # Potatoes & Root Vegetables (starchy)
        'potato', 'potatoes', 'sweet potato', 'yam', 'taro',
        
        # Sugars & Sweeteners
        'sugar', 'brown sugar', 'powdered sugar', 'confectioners sugar',
        'honey', 'maple syrup', 'corn syrup', 'molasses', 'agave',
        
        # Legumes (moderate-high carbs)
        'beans', 'kidney beans', 'black beans', 'pinto beans', 'chickpeas',
        'lentils', 'peas',
        
        # Others
        'corn', 'cornflakes', 'cereal', 'granola', 'chocolate chips',
    }
    
    # Serving carb sources - these are main dishes (red flags for low-carb)
    SERVING_CARB_SOURCES = {
        # Grains
        'rice', 'quinoa', 'couscous', 'barley',
        
        # Pasta (all types)
        'pasta', 'noodles', 'spaghetti', 'linguine', 'fettuccine', 'penne', 
        'macaroni', 'angel hair', 'rigatoni', 'ziti', 'rotini', 'farfalle',
        'ravioli', 'tortellini', 'lasagna', 'gnocchi', 'orzo', 'vermicelli',
        'udon', 'ramen',
        
        # Bread products
        'bread', 'tortilla', 'baguette', 'roll', 'bun', 'pita', 'bagel',
        
        # Potatoes
        'potato', 'potatoes', 'sweet potato', 'yam',
    }
    
    # High-protein ingredients (for protein-rich diets)
    HIGH_PROTEIN_INGREDIENTS = {
        # Meat & Poultry (20-30g protein/100g)
        'chicken', 'chicken breast', 'turkey', 'turkey breast', 'beef', 'lean beef',
        'pork', 'pork chop', 'lamb', 'steak', 'ground beef', 'ground turkey',
        
        # Fish & Seafood (18-25g protein/100g)
        'fish', 'salmon', 'tuna', 'cod', 'halibut', 'tilapia', 'trout', 'sardines',
        'shrimp', 'prawns', 'crab', 'lobster', 'scallops', 'mussels', 'clams',
        
        # Eggs & Dairy (12-30g protein/100g)
        'egg', 'eggs', 'egg whites', 'greek yogurt', 'cottage cheese', 
        'protein powder', 'whey protein',
        
        # Plant-based (8-20g protein/100g)
        'tofu', 'tempeh', 'seitan', 'edamame', 'lentils', 'chickpeas', 
        'black beans', 'kidney beans', 'pinto beans', 'quinoa',
        
        # Nuts & Seeds (moderate protein, 15-25g/100g)
        'almonds', 'peanuts', 'peanut butter', 'almond butter', 'pumpkin seeds',
        'chia seeds', 'hemp seeds',
    }
    
    # High-fat ingredients (for keto/high-fat diets)
    HIGH_FAT_INGREDIENTS = {
        # Healthy fats
        'avocado', 'olive oil', 'coconut oil', 'avocado oil', 'butter',
        'ghee', 'heavy cream', 'cream cheese', 'full-fat cheese',
        
        # Nuts & Seeds
        'almonds', 'walnuts', 'pecans', 'macadamia nuts', 'cashews',
        'pine nuts', 'sunflower seeds', 'flax seeds', 'chia seeds',
        
        # Animal fats
        'bacon', 'pork belly', 'salmon', 'sardines', 'egg yolks',
        
        # Dairy
        'whole milk', 'full-fat yogurt', 'sour cream', 'mascarpone',
    }
    
    # Low-fat indicators
    LOW_FAT_KEYWORDS = {
        'low-fat', 'fat-free', 'skim', 'light', 'lean', 'skinless',
        'steamed', 'boiled', 'grilled', 'baked',
    }
    
    @staticmethod
    def analyze_nutrition(recipe: Dict[str, Any], ingredients_text: str, title: str) -> Dict[str, Any]:
        """
        Analyze nutritional properties of a recipe based on ingredients
        
        Args:
            recipe: Recipe dictionary with ingredients list
            ingredients_text: Lowercase string of all ingredients
            title: Recipe title (lowercase)
        
        Returns:
            Dictionary with nutrition metadata including carb, protein, and fat levels
        """
        ingredients_list = recipe.get('ingredients', [])
        
        high_carb_found = []
        serving_carb_sources = []
        high_protein_found = []
        high_fat_found = []
        
        # Check each ingredient for nutritional content
        for ingredient in ingredients_list:
            ing_lower = str(ingredient).lower()
            
            # Check for high-carb ingredients
            for hc_ing in NutritionFilter.HIGH_CARB_INGREDIENTS:
                if re.search(rf'\b{re.escape(hc_ing)}\b', ing_lower):
                    high_carb_found.append(hc_ing)
                    
                    # Check if it's a serving carb source (main dish carbs)
                    if hc_ing in NutritionFilter.SERVING_CARB_SOURCES:
                        serving_carb_sources.append(hc_ing)
                    break
            
            # Check for high-protein ingredients
            for hp_ing in NutritionFilter.HIGH_PROTEIN_INGREDIENTS:
                if re.search(rf'\b{re.escape(hp_ing)}\b', ing_lower):
                    high_protein_found.append(hp_ing)
                    break
            
            # Check for high-fat ingredients
            for hf_ing in NutritionFilter.HIGH_FAT_INGREDIENTS:
                if re.search(rf'\b{re.escape(hf_ing)}\b', ing_lower):
                    high_fat_found.append(hf_ing)
                    break
        
        # Determine carb level
        carb_level = 'moderate'
        if serving_carb_sources:
            carb_level = 'high'
        elif len(high_carb_found) >= 3:
            carb_level = 'high'
        elif any(ing in high_carb_found for ing in ['flour', 'sugar', 'bread']) and \
             any(word in title for word in ['cake', 'cookie', 'pie', 'bread', 'muffin']):
            carb_level = 'high'
        elif len(high_carb_found) <= 1:
            carb_level = 'low'
        
        # Determine protein level
        protein_level = 'moderate'
        if len(high_protein_found) >= 2:
            protein_level = 'high'  # Multiple protein sources
        elif len(high_protein_found) >= 1:
            # Check if it's a substantial amount (main ingredient)
            main_proteins = {'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 
                           'shrimp', 'tofu', 'lentils', 'chickpeas'}
            if any(p in high_protein_found for p in main_proteins):
                protein_level = 'high'
            else:
                protein_level = 'moderate'
        else:
            protein_level = 'low'
        
        # Determine fat level
        fat_level = 'moderate'
        if len(high_fat_found) >= 3:
            fat_level = 'high'
        elif len(high_fat_found) >= 1:
            # Check for substantial fat sources
            substantial_fats = {'avocado', 'olive oil', 'coconut oil', 'butter', 
                              'bacon', 'salmon', 'nuts'}
            if any(f in high_fat_found for f in substantial_fats):
                fat_level = 'high'
            else:
                fat_level = 'moderate'
        else:
            # Check for low-fat indicators
            if any(kw in ingredients_text or kw in title for kw in NutritionFilter.LOW_FAT_KEYWORDS):
                fat_level = 'low'
            else:
                fat_level = 'low'
        
        return {
            'carb_level': carb_level,
            'high_carb_count': len(set(high_carb_found)),
            'high_carb_ingredients': list(set(high_carb_found)),
            'serving_carb_sources': list(set(serving_carb_sources)),
            'protein_level': protein_level,
            'high_protein_count': len(set(high_protein_found)),
            'high_protein_ingredients': list(set(high_protein_found)),
            'fat_level': fat_level,
            'high_fat_count': len(set(high_fat_found)),
            'high_fat_ingredients': list(set(high_fat_found)),
        }
    
    @staticmethod
    def is_low_carb(nutrition: Dict[str, Any], strict: bool = True) -> bool:
        """
        Determine if a recipe is suitable for low-carb diets
        
        Args:
            nutrition: Nutrition metadata dictionary
            strict: If True, reject any recipe with serving carb sources
        
        Returns:
            True if recipe is low-carb friendly
        """
        if strict:
            # Strict mode: No serving carb sources, must be 'low' carb
            return (
                nutrition.get('carb_level') == 'low' and 
                len(nutrition.get('serving_carb_sources', [])) == 0
            )
        else:
            # Relaxed mode: Allow 'low' or 'moderate' without serving carbs
            return (
                nutrition.get('carb_level') in ['low', 'moderate'] and
                len(nutrition.get('serving_carb_sources', [])) == 0
            )
    
    @staticmethod
    def is_high_protein(nutrition: Dict[str, Any]) -> bool:
        """Determine if a recipe is high in protein"""
        return nutrition.get('protein_level') == 'high'
    
    @staticmethod
    def is_low_fat(nutrition: Dict[str, Any]) -> bool:
        """Determine if a recipe is low in fat"""
        return nutrition.get('fat_level') == 'low'
    
    @staticmethod
    def is_high_fat(nutrition: Dict[str, Any]) -> bool:
        """Determine if a recipe is high in fat (e.g., for keto)"""
        return nutrition.get('fat_level') == 'high'
    
    @staticmethod
    def parse_nutrition_requirements(query: str) -> Dict[str, bool]:
        """
        Parse query to extract nutrition requirements
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with nutrition requirement flags
        """
        query_lower = query.lower()
        
        requirements = {
            'low_carb': False,
            'keto': False,
            'high_protein': False,
            'low_fat': False,
            'high_fat': False,
        }
        
        # Low carb keywords
        low_carb_patterns = [
            r'\blow[\s-]?carb\b',
            r'\breduce.*carb',
            r'\bcut.*carb',
            r'\bfewer.*carb',
            r'\bno.*carb',
            r'\bcarb.*free',
        ]
        
        for pattern in low_carb_patterns:
            if re.search(pattern, query_lower):
                requirements['low_carb'] = True
                break
        
        # Keto diet (implies low-carb + high-fat)
        if 'keto' in query_lower or 'ketogenic' in query_lower:
            requirements['keto'] = True
            requirements['low_carb'] = True
            requirements['high_fat'] = True
        
        # High protein keywords
        if re.search(r'\bhigh[\s-]?protein\b', query_lower) or \
           re.search(r'\bprotein[\s-]?rich\b', query_lower) or \
           re.search(r'\bmore protein\b', query_lower):
            requirements['high_protein'] = True
        
        # Low fat keywords
        if re.search(r'\blow[\s-]?fat\b', query_lower) or \
           re.search(r'\bfat[\s-]?free\b', query_lower) or \
           re.search(r'\breduce.*fat\b', query_lower) or \
           'lean' in query_lower:
            requirements['low_fat'] = True
        
        # High fat keywords (for keto, etc.)
        if re.search(r'\bhigh[\s-]?fat\b', query_lower) or \
           re.search(r'\bfat[\s-]?rich\b', query_lower):
            requirements['high_fat'] = True
        
        return requirements
    
    @staticmethod
    def matches_nutrition_requirements(nutrition: Dict[str, Any], requirements: Dict[str, bool]) -> bool:
        """
        Check if recipe nutrition matches the requirements
        
        Args:
            nutrition: Recipe nutrition metadata
            requirements: Nutrition requirements from query
        
        Returns:
            True if recipe matches all nutrition requirements
        """
        # Low-carb/keto
        if requirements.get('low_carb') or requirements.get('keto'):
            if not NutritionFilter.is_low_carb(nutrition, strict=True):
                return False
        
        # High protein
        if requirements.get('high_protein'):
            if not NutritionFilter.is_high_protein(nutrition):
                return False
        
        # Low fat
        if requirements.get('low_fat'):
            if not NutritionFilter.is_low_fat(nutrition):
                return False
        
        # High fat (for keto, etc.)
        if requirements.get('high_fat'):
            if not NutritionFilter.is_high_fat(nutrition):
                return False
        
        # Special case: Keto diet requires both low-carb AND high-fat
        if requirements.get('keto'):
            if not (NutritionFilter.is_low_carb(nutrition, strict=True) and 
                   NutritionFilter.is_high_fat(nutrition)):
                return False
        
        return True


# ======== Unit Tests ========
if __name__ == "__main__":
    print("Testing NutritionFilter...")
    
    # Test nutrition requirement parsing
    print("\nTesting nutrition requirement parsing:")
    test_queries = [
        "I want a low-carb dinner",
        "Find me a keto-friendly recipe",
        "High protein breakfast ideas",
        "Low-fat chicken recipe",
        "I need a protein-rich, low-carb meal",
    ]
    
    for query in test_queries:
        requirements = NutritionFilter.parse_nutrition_requirements(query)
        print(f"  Query: {query}")
        print(f"  Requirements: {requirements}")
    
    # Test nutrition analysis
    print("\nTesting nutrition analysis:")
    test_recipes = [
        {
            'title': 'Grilled Chicken Salad',
            'ingredients': ['2 chicken breasts', '1 cup lettuce', 'olive oil', 'lemon juice']
        },
        {
            'title': 'Spaghetti Carbonara',
            'ingredients': ['1 lb spaghetti', '4 eggs', '1 cup parmesan', 'bacon']
        },
        {
            'title': 'Keto Fat Bombs',
            'ingredients': ['coconut oil', 'butter', 'cream cheese', 'cocoa powder']
        },
    ]
    
    for recipe in test_recipes:
        ingredients_text = ' '.join([str(i).lower() for i in recipe['ingredients']])
        title = recipe['title'].lower()
        nutrition = NutritionFilter.analyze_nutrition(recipe, ingredients_text, title)
        print(f"\n  Recipe: {recipe['title']}")
        print(f"  Nutrition: {nutrition}")
        print(f"  Is low-carb? {NutritionFilter.is_low_carb(nutrition)}")
        print(f"  Is high-protein? {NutritionFilter.is_high_protein(nutrition)}")
        print(f"  Is high-fat? {NutritionFilter.is_high_fat(nutrition)}")
    
    print("\n✓ NutritionFilter tests complete")
