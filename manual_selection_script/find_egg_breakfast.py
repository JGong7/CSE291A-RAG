import json, os

OUT_DIR = "manual_selection_dataset"
os.makedirs(OUT_DIR, exist_ok=True)  # make sure the folder exists

def has_eggs_and_cheese(ingredients):
    ings = [str(i).lower() for i in ingredients]
    has_egg = any("egg" in i for i in ings)      # covers egg, eggs, egg yolk, etc.
    has_cheese = any("cheese" in i for i in ings)
    return has_egg and has_cheese

# ========= RecipeNLG =========
with open('RecipeNLG_dataset/recipes_nlg_clean.json', 'r') as f:
    recipes = json.load(f)

breakfast_recipes = [
    r for r in recipes
    if 'breakfast' in r.get('title', '').lower()
    and has_eggs_and_cheese(r.get('ingredients', []))
]

with open(os.path.join(OUT_DIR, 'eggs_cheese_breakfast_RecipeNLG.json'), 'w') as f:
    json.dump(breakfast_recipes, f, indent=4)

# ========= Spoonacular =========
with open('Spoonacular_API/spoonacular_dataset.json', 'r') as f:   # <-- fixed extension
    recipes = json.load(f)

breakfast_recipes = [
    r for r in recipes
    if 'breakfast' in r.get('title', '').lower()
    and has_eggs_and_cheese(r.get('ingredients', []))
]

with open(os.path.join(OUT_DIR, 'eggs_cheese_breakfast_Spoonacular.json'), 'w') as f:
    json.dump(breakfast_recipes, f, indent=4)

