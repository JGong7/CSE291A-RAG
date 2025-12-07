"""
Ingredient Quantity Filter
Handles queries with ingredient quantity constraints like "I only have two eggs"
"""

import re
from typing import Dict, List, Tuple, Optional


class QuantityFilter:
    """Extract and filter based on ingr    # Test recipe requirement extraction
    print("\nTesting recipe requirement extraction:")
    test_ingredients = [
        ["6 eggs", "1 c. milk", "2 cups flour"],
        ["1 egg", "3 tablespoons butter"],
        ["2 lb chicken breast", "1/2 cup cheese"],
        ["2 or more eggs", "2 or more tomatoes", "grated cheese"],  # NEW TEST
        ["7 large eggs", "1/2 cup cheddar cheese"],  # NEW TEST
    ]
    for ingredients in test_ingredients:
        requirements = QuantityFilter.extract_recipe_requirements(ingredients)
        print(f"  Ingredients: {ingredients}")
        print(f"  Requirements: {requirements}")
    
    # Test constraint satisfaction
    print("\nTesting constraint satisfaction:")
    test_cases = [
        ({'egg': 2.0}, {'egg': 1.0}, "User has 2 eggs, recipe needs 1"),
        ({'egg': 2.0}, {'egg': 6.0}, "User has 2 eggs, recipe needs 6"),
        ({'egg': 2.0}, {'egg': 2.5}, "User has 2 eggs, recipe needs 2.5"),
        ({'egg': 2.0}, {'egg': 2.0}, "User has 2 eggs, recipe needs 2 (exact)"),  # NEW TEST
        ({'egg': 2.0}, {'egg': 7.0}, "User has 2 eggs, recipe needs 7"),  # NEW TEST
    ]straints"""
    
    # Number word to digit mapping
    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'a': 1, 'an': 1, 'couple': 2, 'few': 3
    }
    
    # Common ingredient patterns for quantity extraction
    # Updated to handle modifiers like "jumbo", "large", "fresh" and "or more" patterns
    INGREDIENT_PATTERNS = {
        'egg': [  # Use singular form as canonical key
            r'(\d+(?:\.\d+)?)\s*(?:or\s+more\s+)?(?:\w+\s+)?eggs?',  # "2 or more eggs", "2 large eggs"
            r'eggs?\s*(\d+)',
        ],
        'chicken': [
            r'(\d+(?:\.\d+)?)\s*(?:lb\.?|lbs?\.?|pound|pounds)?\s*(?:or\s+more\s+)?(?:\w+\s+)?chicken',
            r'(\d+)\s*chicken\s*(?:breast|thigh|wing)',
        ],
        'butter': [
            r'(\d+(?:\.\d+)?)\s*(?:stick|sticks|cup|cups|tbsp|tablespoon)?\s*(?:or\s+more\s+)?(?:\w+\s+)?butter',
        ],
        'milk': [
            r'(\d+(?:\.\d+)?)\s*(?:c\.?|cup|cups|pt\.?|pint)?\s*(?:or\s+more\s+)?(?:\w+\s+)?milk',
        ],
        'cheese': [
            r'(\d+(?:\.\d+)?)\s*(?:c\.?|cup|cups|oz\.?|ounce)?\s*(?:or\s+more\s+)?(?:\w+\s+)?cheese',
        ],
        'flour': [
            r'(\d+(?:\.\d+)?)\s*(?:c\.?|cup|cups)?\s*(?:or\s+more\s+)?(?:\w+\s+)?flour',
        ],
        'sugar': [
            r'(\d+(?:\.\d+)?)\s*(?:c\.?|cup|cups)?\s*(?:or\s+more\s+)?(?:\w+\s+)?sugar',
        ],
    }
    
    @staticmethod
    def parse_number_word(word: str) -> Optional[float]:
        """Convert number words to digits: 'two' -> 2"""
        word_lower = word.lower().strip()
        
        # Direct digit
        try:
            return float(word_lower)
        except ValueError:
            pass
        
        # Word to number
        return QuantityFilter.NUMBER_WORDS.get(word_lower)
    
    @staticmethod
    def extract_user_constraints(query: str) -> Dict[str, float]:
        """
        Extract user's ingredient quantity constraints from query
        
        Examples:
            "I only have two eggs" -> {'eggs': 2.0}
            "I have 1 lb chicken" -> {'chicken': 1.0}
            "I only have one egg and some cheese" -> {'eggs': 1.0}
        
        Returns:
            Dict mapping ingredient name to max available quantity
        """
        constraints = {}
        query_lower = query.lower()
        
        # Pattern 1: "I (only) have X ingredient"
        patterns = [
            r'(?:only\s+)?have\s+(\w+)\s+(\w+)',  # "have two eggs"
            r'(?:only\s+)?have\s+(\d+(?:\.\d+)?)\s+(\w+)',  # "have 2 eggs"
            r'(?:with|using)\s+(\w+)\s+(\w+)',  # "with two eggs"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                quantity_str = match.group(1)
                ingredient = match.group(2)
                
                # Try to parse quantity
                quantity = QuantityFilter.parse_number_word(quantity_str)
                
                if quantity is not None:
                    # Normalize ingredient name (remove plural 's')
                    ingredient_key = ingredient.rstrip('s')
                    constraints[ingredient_key] = quantity
        
        return constraints
    
    @staticmethod
    def extract_recipe_requirements(ingredients: List[str]) -> Dict[str, float]:
        """
        Extract ingredient quantities required by a recipe
        
        Args:
            ingredients: List of ingredient strings from recipe
        
        Returns:
            Dict mapping ingredient name to required quantity
        
        Examples:
            ["6 eggs", "1 cup milk"] -> {'eggs': 6.0, 'milk': 1.0}
            ["2 or more eggs"] -> {'eggs': 2.0}  # Treat as minimum
        """
        requirements = {}
        
        for ingredient_str in ingredients:
            ingredient_lower = ingredient_str.lower()
            
            # Try each ingredient pattern
            for ingredient_name, patterns in QuantityFilter.INGREDIENT_PATTERNS.items():
                for pattern in patterns:
                    match = re.search(pattern, ingredient_lower)
                    if match:
                        try:
                            quantity = float(match.group(1))
                            # Keep the maximum quantity if multiple mentions
                            if ingredient_name in requirements:
                                requirements[ingredient_name] = max(
                                    requirements[ingredient_name], 
                                    quantity
                                )
                            else:
                                requirements[ingredient_name] = quantity
                        except (ValueError, IndexError):
                            pass
        
        return requirements
    
    @staticmethod
    def satisfies_constraints(
        recipe_requirements: Dict[str, float],
        user_constraints: Dict[str, float],
        tolerance: float = 0.0
    ) -> bool:
        """
        Check if recipe requirements satisfy user's constraints
        
        Args:
            recipe_requirements: What recipe needs
            user_constraints: What user has (maximum available)
            tolerance: Allow recipes needing slightly more (0.0 = strict)
        
        Returns:
            True if recipe can be made with user's available ingredients
        
        Examples:
            User has 2 eggs, recipe needs 1 egg -> True
            User has 2 eggs, recipe needs 6 eggs -> False
            User has 2 eggs, recipe needs 2.5 eggs, tolerance=0.5 -> True
        """
        for ingredient, user_max in user_constraints.items():
            # Check if recipe requires this ingredient
            recipe_need = recipe_requirements.get(ingredient, 0)
            
            # Recipe needs more than user has (beyond tolerance)
            if recipe_need > user_max + tolerance:
                return False
        
        return True
    
    @staticmethod
    def filter_recipes_by_quantity(
        recipes: List[Dict],
        recipe_indices: List[int],
        query: str,
        tolerance: float = 0.0
    ) -> List[int]:
        """
        Filter recipe indices based on quantity constraints in query
        
        Args:
            recipes: Full list of recipes
            recipe_indices: Candidate recipe indices to filter
            query: User query with potential quantity constraints
            tolerance: Tolerance for quantity matching
        
        Returns:
            Filtered list of recipe indices that satisfy constraints
        """
        # Extract user constraints from query
        user_constraints = QuantityFilter.extract_user_constraints(query)
        
        # No constraints found - return all candidates
        if not user_constraints:
            return recipe_indices
        
        # Filter recipes
        filtered_indices = []
        
        for idx in recipe_indices:
            recipe = recipes[idx]
            
            # Extract recipe requirements
            ingredients = recipe.get('ingredients', [])
            recipe_requirements = QuantityFilter.extract_recipe_requirements(ingredients)
            
            # Check if recipe satisfies constraints
            if QuantityFilter.satisfies_constraints(
                recipe_requirements, 
                user_constraints, 
                tolerance
            ):
                filtered_indices.append(idx)
        
        return filtered_indices


# ======== Testing ========
if __name__ == "__main__":
    # Test number word parsing
    print("Testing number word parsing:")
    test_words = ['two', 'three', '2', '1.5', 'few', 'couple']
    for word in test_words:
        print(f"  '{word}' -> {QuantityFilter.parse_number_word(word)}")
    
    # Test constraint extraction
    print("\nTesting constraint extraction:")
    test_queries = [
        "I only have two eggs, give me a quick breakfast",
        "I have 1 lb chicken and some vegetables",
        "Using three cups of flour, make bread",
    ]
    for query in test_queries:
        constraints = QuantityFilter.extract_user_constraints(query)
        print(f"  Query: {query}")
        print(f"  Constraints: {constraints}")
    
    # Test recipe requirement extraction
    print("\nTesting recipe requirement extraction:")
    test_ingredients = [
        ["6 eggs", "1 c. milk", "2 cups flour"],
        ["1 egg", "3 tablespoons butter"],
        ["2 lb chicken breast", "1/2 cup cheese"],
        ["2 or more eggs", "2 or more tomatoes", "grated cheese"],  # NEW TEST
        ["7 large eggs", "1/2 cup cheddar cheese"],  # NEW TEST
        ["6 jumbo eggs", "1 cup milk"],  # NEW TEST
    ]
    for ingredients in test_ingredients:
        requirements = QuantityFilter.extract_recipe_requirements(ingredients)
        print(f"  Ingredients: {ingredients}")
        print(f"  Requirements: {requirements}")
    
    # Test constraint satisfaction
    print("\nTesting constraint satisfaction:")
    test_cases = [
        ({'egg': 2.0}, {'egg': 1.0}, "User has 2 eggs, recipe needs 1"),
        ({'egg': 2.0}, {'egg': 6.0}, "User has 2 eggs, recipe needs 6"),
        ({'egg': 2.0}, {'egg': 2.5}, "User has 2 eggs, recipe needs 2.5"),
        ({'egg': 2.0}, {'egg': 2.0}, "User has 2 eggs, recipe needs 2 (exact)"),  # NEW TEST
        ({'egg': 2.0}, {'egg': 7.0}, "User has 2 eggs, recipe needs 7"),  # NEW TEST
    ]
    for user_c, recipe_r, desc in test_cases:
        result = QuantityFilter.satisfies_constraints(recipe_r, user_c)
        print(f"  {desc}: {result}")
