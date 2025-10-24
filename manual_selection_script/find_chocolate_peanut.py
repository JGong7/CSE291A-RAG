
import json

# Define dessert keywords for birthday occasions (easily customizable)
dessert_keywords = ['dessert', 'cake', ' pie', 'cupcake', 'muffin']
baking_keywords = ['birthday']

# Load the recipes from the JSON file
with open('RecipeNLG_dataset/recipes_nlg_clean.json', 'r', encoding='utf-8') as file:
    recipes = json.load(file)

# Filter for peanut chocolate dessert recipes (based on query: "I love peanuts, give me a chocolate dessert recipe for special birthday occasions")
peanut_chocolate_dessert_recipes = [
    recipe for recipe in recipes
    if (any('peanut' in ingredient.lower() for ingredient in recipe.get('ingredients', [])) or 
        any('peanut' in ner_item.lower() for ner_item in recipe.get('ner', []))) and 
       (any('chocolate' in ingredient.lower() for ingredient in recipe.get('ingredients', [])) or 
        any('chocolate' in ner_item.lower() for ner_item in recipe.get('ner', []))) and
       (any(dessert_word in recipe.get('title', '').lower() for dessert_word in dessert_keywords) or
        any(dessert_word in ' '.join(recipe.get('directions', [])).lower() for dessert_word in baking_keywords))
]

# Save the peanut chocolate dessert recipes to a new JSON file
with open('RecipeNLG_dataset/peanut_chocolate_dessert.json', 'w', encoding='utf-8') as file:
    json.dump(peanut_chocolate_dessert_recipes, file, indent=4)

# Load the recipes from the JSON file
with open('Spoonacular_API/spoonacular_dataset.json', 'r', encoding='utf-8') as file:
    recipes = json.load(file)

# Filter for peanut chocolate dessert recipes (based on query: "I love peanuts, give me a chocolate dessert recipe for special birthday occasions")
peanut_chocolate_dessert_recipes = [
    recipe for recipe in recipes
    if any('peanut' in ingredient.lower() for ingredient in recipe.get('ingredients', [])) and 
       any('chocolate' in ingredient.lower() for ingredient in recipe.get('ingredients', [])) and
       (any(dessert_word in recipe.get('title', '').lower() for dessert_word in dessert_keywords) or
        any(dessert_word in recipe.get('directions', '').lower() for dessert_word in baking_keywords))
]

# Save the peanut chocolate dessert recipes to a new JSON file
with open('Spoonacular_API/peanut_chocolate_dessert.json', 'w', encoding='utf-8') as file:
    json.dump(peanut_chocolate_dessert_recipes, file, indent=4)