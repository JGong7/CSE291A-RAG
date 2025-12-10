
import json

def is_vegan(ingredients):
    non_vegan_keywords = ['meat', 'beef', 'chicken', 'pork', 'fish', 'seafood', 'shrimp', 'crab', 'lobster','turkey','ham','hamburger','sausage','goose','duck','pig']
    return not any(keyword in ingredient.lower() for ingredient in ingredients for keyword in non_vegan_keywords)

# Load the recipes from the JSON file
with open('RecipeNLG_dataset/soup.json', 'r') as file:
    recipes = json.load(file)

# Filter for vegan soup recipes
vegan_soup_recipes = [
    recipe for recipe in recipes 
    if 'soup' in recipe.get('title', '').lower() and is_vegan(recipe.get('ingredients', []))
]

# Save the vegan soup recipes to a new JSON file
with open('RecipeNLG_dataset/vegan_soup.json', 'w') as file:
    json.dump(vegan_soup_recipes, file, indent=4)

# Load the recipes from the JSON file
with open('Spoonacular_API/soup.json', 'r') as file:
    recipes = json.load(file)

# Filter for vegan soup recipes
vegan_soup_recipes = [
    recipe for recipe in recipes 
    if 'soup' in recipe.get('title', '').lower() and is_vegan(recipe.get('ingredients', []))
]

# Save the vegan soup recipes to a new JSON file
with open('Spoonacular_API/vegan_soup.json', 'w') as file:
    json.dump(vegan_soup_recipes, file, indent=4)