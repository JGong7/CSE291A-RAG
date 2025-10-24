import json
import os

OUT_DIR = "manual_selection_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

def is_lactose_free(ingredients):
    """
    Return True if the recipe does NOT contain lactose ingredients.
    """
    lactose_keywords = [
        'milk', 'butter', 'cream', 'cheese', 'yogurt', 'ghee',
        'whey', 'custard', 'ricotta', 'mozzarella', 'parmesan',
        'sour cream', 'condensed milk', 'evaporated milk', 'milk powder'
    ]
    ingredients_lower = [i.lower() for i in ingredients]
    return not any(keyword in i for i in ingredients_lower for keyword in lactose_keywords)


# ========= RecipeNLG dataset =========
with open('RecipeNLG_dataset/recipes_nlg_clean.json', 'r') as f:
    recipes = json.load(f)

lactose_free_pasta = [
    r for r in recipes
    if 'pasta' in r.get('title', '').lower() and is_lactose_free(r.get('ingredients', []))
]

with open(os.path.join(OUT_DIR, 'lactose_free_pasta_RecipeNLG.json'), 'w') as f:
    json.dump(lactose_free_pasta, f, indent=4)


# ========= Spoonacular dataset =========
with open('Spoonacular_API/spoonacular_dataset.json', 'r') as f:
    recipes = json.load(f)

lactose_free_pasta = [
    r for r in recipes
    if 'pasta' in r.get('title', '').lower() and is_lactose_free(r.get('ingredients', []))
]

with open(os.path.join(OUT_DIR, 'lactose_free_pasta_Spoonacular.json'), 'w') as f:
    json.dump(lactose_free_pasta, f, indent=4)
