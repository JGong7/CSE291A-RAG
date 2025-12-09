"""
Ingredient Quantity Filter
Handles queries with ingredient quantity constraints like "I only have two eggs"

Updated design:
- Add a lightweight unit normalization layer so that recipe requirements
  and user constraints expressed in different units (g/kg/oz/lb, ml/l/cup/tbsp/tsp)
  can be compared in a common canonical unit where possible.
- Public API remains the same so both the old HybridRetriever and the
  new AdvancedRetriever can share this improved robustness.
"""

import re
from typing import Dict, List, Tuple, Optional, Any


class UnitNormalizer:
    """Utility for normalizing common cooking units.

    We distinguish three kinds:
      - "weight": grams
      - "volume": milliliters
      - "count": unit-less counts (eggs, pieces, etc.)
    """

    WEIGHT_IN_GRAMS = {
        "g": 1.0,
        "gram": 1.0,
        "grams": 1.0,
        "kg": 1000.0,
        "kilogram": 1000.0,
        "kilograms": 1000.0,
        "oz": 28.35,
        "ounce": 28.35,
        "ounces": 28.35,
        "lb": 453.6,
        "lbs": 453.6,
        "pound": 453.6,
        "pounds": 453.6,
    }

    VOLUME_IN_ML = {
        "ml": 1.0,
        "milliliter": 1.0,
        "milliliters": 1.0,
        "l": 1000.0,
        "liter": 1000.0,
        "liters": 1000.0,
        "cup": 240.0,
        "cups": 240.0,
        "c": 240.0,
        "tbsp": 15.0,
        "tablespoon": 15.0,
        "tablespoons": 15.0,
        "tsp": 5.0,
        "teaspoon": 5.0,
        "teaspoons": 5.0,
    }

    @staticmethod
    def normalize(quantity: float, unit: Optional[str]) -> Tuple[float, str]:
        """Normalize a quantity with an optional unit.

        Returns (normalized_value, kind) where kind is one of
        {"weight", "volume", "count"}.
        """

        if not unit:
            return quantity, "count"

        u = unit.lower().rstrip(". ")

        if u in UnitNormalizer.WEIGHT_IN_GRAMS:
            return quantity * UnitNormalizer.WEIGHT_IN_GRAMS[u], "weight"
        if u in UnitNormalizer.VOLUME_IN_ML:
            return quantity * UnitNormalizer.VOLUME_IN_ML[u], "volume"

        # Unknown unit: treat as count
        return quantity, "count"


class QuantityFilter:
    """Extract and filter based on ingredient quantity constraints"""

    # Number word to digit mapping
    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "a": 1,
        "an": 1,
        "couple": 2,
        "few": 3,
        "half": 0.5,
        "quarter": 0.25,
        "dozen": 12,
    }

    # Common ingredient patterns for quantity extraction
    # Updated to handle modifiers like "jumbo", "large", "fresh" and "or more" patterns
    # Where possible we also capture a unit group for normalization.
    INGREDIENT_PATTERNS = {
        "egg": [  # Use singular form as canonical key
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(?:or\s+more\s+)?(?:\w+\s+)?(eggs?)",  # supports fractions like 3/4 eggs
            r"eggs?\s*(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
        ],
        "chicken": [
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(lb\.?|lbs?\.?|pound|pounds)?\s*(?:or\s+more\s+)?(?:\w+\s+)?chicken",
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*chicken\s*(?:breast|thigh|wing)",
        ],
        "butter": [
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(stick|sticks|cup|cups|tbsp|tablespoon|tablespoons)?\s*(?:or\s+more\s+)?(?:\w+\s+)?butter",
        ],
        "milk": [
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(c\.?|cup|cups|pt\.?|pint|pints)?\s*(?:or\s+more\s+)?(?:\w+\s+)?milk",
        ],
        "cheese": [
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(c\.?|cup|cups|oz\.?|ounce|ounces)?\s*(?:or\s+more\s+)?(?:\w+\s+)?cheese",
        ],
        "flour": [
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(c\.?|cup|cups)?\s*(?:or\s+more\s+)?(?:\w+\s+)?flour",
        ],
        "sugar": [
            r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s*(c\.?|cup|cups)?\s*(?:or\s+more\s+)?(?:\w+\s+)?sugar",
        ],
    }

    @staticmethod
    def _parse_fraction(qty_str: str) -> Optional[float]:
        """Parse simple fractions like '1/2' into float.

        Supports plain numbers ("1", "1.5") and simple a/b forms ("3/4").
        Returns None if parsing fails.
        """
        if qty_str is None:
            return None
        qty_str = qty_str.strip()
        if not qty_str:
            return None

        if "/" in qty_str:
            try:
                num, denom = qty_str.split("/", 1)
                num = float(num.strip())
                denom = float(denom.strip())
                if denom == 0:
                    return None
                return num / denom
            except ValueError:
                return None

        try:
            return float(qty_str)
        except ValueError:
            return None

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
    def extract_user_constraints(query: str) -> Dict[str, Dict[str, Any]]:
        """Extract user's ingredient quantity constraints from query.

        Returns a mapping from ingredient -> {"value": float, "kind": str},
        where kind is one of {"weight", "volume", "count"}.
        """

        constraints: Dict[str, Dict[str, Any]] = {}
        query_lower = query.lower()

        # Pattern 1: "I (only) have X [unit] ingredient"
        patterns = [
            # quantity + ingredient, optional unit in between, e.g. "have 2 cups flour" or "have 1/2 cup flour"
            r"(?:only\s+)?have\s+(\S+)\s*(\w+)?\s+(\w+)",
            # word-number + ingredient: "have two eggs"
            r"(?:only\s+)?have\s+(\w+)\s+(\w+)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, query_lower):
                groups = match.groups()

                if len(groups) == 3:
                    # quantity (possibly fraction) + optional unit + ingredient
                    qty_str, unit, ingredient = groups
                    qty = QuantityFilter._parse_fraction(qty_str)
                    if qty is None:
                        continue
                else:
                    # word number + ingredient (no explicit unit)
                    qty_str, ingredient = groups
                    qty = QuantityFilter.parse_number_word(qty_str) or 0.0
                    unit = None

                ingredient_key = ingredient.rstrip("s")

                # If ingredient_key is actually a unit (cup, lb, g, etc.), skip it.
                if ingredient_key in UnitNormalizer.WEIGHT_IN_GRAMS or ingredient_key in UnitNormalizer.VOLUME_IN_ML:
                    continue

                value, kind = UnitNormalizer.normalize(qty, unit)
                constraints[ingredient_key] = {"value": value, "kind": kind}

        # Optional: keep only known ingredient keys we care about
        valid_keys = {"egg", "chicken", "butter", "milk", "cheese", "flour", "sugar"}
        constraints = {k: v for k, v in constraints.items() if k in valid_keys}

        return constraints

    @staticmethod
    def extract_recipe_requirements(ingredients: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract ingredient quantities required by a recipe.

        Returns mapping ingredient -> {"value": float, "kind": str}.
        """

        requirements: Dict[str, Dict[str, Any]] = {}

        for ingredient_str in ingredients:
            ingredient_lower = ingredient_str.lower()

            # Try each ingredient pattern
            for ingredient_name, patterns in QuantityFilter.INGREDIENT_PATTERNS.items():
                for pattern in patterns:
                    match = re.search(pattern, ingredient_lower)
                    if not match:
                        continue

                    try:
                        # group(1) is quantity (may be fraction), group(2) may be unit
                        qty_raw = match.group(1)
                        qty = QuantityFilter._parse_fraction(qty_raw) or 0.0
                        unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                        value, kind = UnitNormalizer.normalize(qty, unit)
                    except (IndexError, AttributeError):
                        continue

                    prev = requirements.get(ingredient_name)
                    if prev is None:
                        requirements[ingredient_name] = {"value": value, "kind": kind}
                    else:
                        # Only merge if kinds are compatible; otherwise keep the max by value
                        if prev["kind"] == kind:
                            prev["value"] = max(prev["value"], value)
                        else:
                            # Different kinds (e.g., weight vs volume) are hard to reconcile;
                            # keep the larger normalized value as a conservative estimate.
                            prev["value"] = max(prev["value"], value)

        return requirements

    @staticmethod
    def satisfies_constraints(
        recipe_requirements: Dict[str, Dict[str, Any]],
        user_constraints: Dict[str, Dict[str, Any]],
        tolerance: float = 0.0,
    ) -> bool:
        """Check if recipe requirements satisfy user's constraints.

        Both recipe_requirements and user_constraints must use the same
        canonical units (handled by UnitNormalizer).

        Supports optional min/max-style constraints via the following
        user constraint schema (backward compatible):
          {"value": v}              -> interpreted as max=v
          {"min": a, "max": b}     -> allowed range [a, b]

        This schema is deliberately aligned with the LLM structured
        output in llm_query_processor (quantity_constraints), so that
        LLM-produced constraints can be passed in directly.
        """

        for ingredient, user_info in user_constraints.items():
            recipe_info = recipe_requirements.get(ingredient)
            if recipe_info is None:
                # Recipe doesn't explicitly require this ingredient -> OK
                continue

            if recipe_info["kind"] != user_info.get("kind", recipe_info["kind"]):
                # Different dimensions, skip strict comparison
                continue

            recipe_need = recipe_info["value"]

            # Derive min/max
            user_min = user_info.get("min")
            user_max = user_info.get("max")
            if user_min is None and user_max is None:
                # Backward compatible path: use single value as max
                user_max = user_info.get("value", recipe_need)

            if user_min is not None and recipe_need < user_min - tolerance:
                return False
            if user_max is not None and recipe_need > user_max + tolerance:
                return False

        return True

    @staticmethod
    def filter_recipes_by_quantity(
        recipes: List[Dict],
        recipe_indices: List[int],
        query: str,
        tolerance: float = 0.0,
        structured_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[int]:
        """Filter recipe indices based on quantity constraints.

        If structured_constraints is provided (e.g. from LLM
        quantity_constraints), it is used directly as user_constraints.
        Otherwise, fall back to extracting constraints from the raw
        query text so old HybridRetriever experiments remain valid.
        """

        if structured_constraints:
            user_constraints = structured_constraints
        else:
            # Extract user constraints from query text (regex path)
            user_constraints = QuantityFilter.extract_user_constraints(query)

        # No constraints found - return all candidates
        if not user_constraints:
            return recipe_indices

        filtered_indices: List[int] = []

        for idx in recipe_indices:
            recipe = recipes[idx]
            ingredients = recipe.get("ingredients", [])
            recipe_requirements = QuantityFilter.extract_recipe_requirements(ingredients)

            if QuantityFilter.satisfies_constraints(
                recipe_requirements,
                user_constraints,
                tolerance,
            ):
                filtered_indices.append(idx)

        return filtered_indices


# ======== Testing ========
if __name__ == "__main__":
    # Test number word parsing
    print("Testing number word parsing:")
    test_words = ["two", "three", "2", "1.5", "few", "couple"]
    for word in test_words:
        print(f"  '{word}' -> {QuantityFilter.parse_number_word(word)}")

    # Test constraint extraction
    print("\nTesting constraint extraction:")
    test_queries = [
        "I only have two eggs, give me a quick breakfast",
        "I have 1 lb chicken and some vegetables",
        "Using three cups flour, make bread",
        "I have 500 g flour",
        "I have 1/2 cup sugar",
        "Using 3/4 cup milk",
    ]
    for query in test_queries:
        constraints = QuantityFilter.extract_user_constraints(query)
        print(f"  Query: {query}")
        print(f"  Constraints: {constraints}")

    # Test recipe requirement extraction
    print("\nTesting recipe requirement extraction:")
    test_ingredients = [
        ["6 jumbo eggs", "1 c. milk", "2 cups flour"],
        ["1 egg", "3 tablespoons butter"],
        ["1.5 egg", "3 tablespoons butter"],
        ["3/4 egg", "3 tablespoons butter"],
        ["2 lb chicken breast", "1/2 cup cheese"],  # contains fraction
    ]
    for ingredients in test_ingredients:
        reqs = QuantityFilter.extract_recipe_requirements(ingredients)
        print(f"  Ingredients: {ingredients}")
        print(f"  Requirements: {reqs}")
