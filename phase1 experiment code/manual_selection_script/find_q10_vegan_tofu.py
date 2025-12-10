import json

keywords = ['vega', 'tofu']
exclude_keywords = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'lamb', 'turkey']

# Load the recipes from the JSON file
with open('RecipeNLG_dataset/recipes_nlg_clean.json', 'r', encoding='utf-8') as file:
    recipes = json.load(file)

# Filter for peanut chocolate dessert recipes (based on query: "I love peanuts, give me a chocolate dessert recipe for special birthday occasions")
peanut_chocolate_dessert_recipes = [
    recipe for recipe in recipes
    if (any('tofu' in ingredient.lower() for ingredient in recipe.get('ingredients', []))) and
       (any(keyword in ' '.join(recipe.get('directions', '')).lower() for keyword in keywords)) and
       (not any(exclude_keyword in ' '.join(recipe.get('directions', '')).lower() for exclude_keyword in exclude_keywords))

]

# Save the peanut chocolate dessert recipes to a new JSON file
with open('manual_selection_dataset/q10_vegan_tofu.json', 'w', encoding='utf-8') as file:
    json.dump(peanut_chocolate_dessert_recipes, file, indent=4)
