import pandas as pd
import json
import os
from ast import literal_eval

# ========== CONFIG ==========
INPUT_CSV = "RecipeNLG_dataset.csv"
OUTPUT_JSON = "recipes_nlg_clean.json"
SAMPLE_SIZE = 5000  # set to None to process all

# ========== STEP 1: Load ==========
print("Loading RecipeNLG dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} recipes.")

if SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df)} recipes for cleaning.")

# ['Title', 'Ingredients', 'Directions', 'NER', ...]
df.columns = [c.lower().strip() for c in df.columns]

# ========== STEP 2: Cleaning helper ==========
def parse_list_field(x):
    # paese a field that should be a list of strings
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        val = literal_eval(str(x))
        if isinstance(val, list):
            return [str(i).strip().lower() for i in val if len(str(i).strip()) > 1]
        else:
            return [str(val).strip().lower()]
    except Exception:
        return [t.strip().lower() for t in str(x).split(",") if t.strip()]

# ========== STEP 3: Clean & Format ==========
recipes = []
for i, row in df.iterrows():
    title = str(row.get("title") or "").strip()
    ingredients = parse_list_field(row.get("ingredients"))
    instructions = parse_list_field(row.get("directions"))
    ner = parse_list_field(row.get("ner"))

    if not title or not ingredients or not instructions:
        continue

    recipe = {
        "id": f"recnlg_{i}",
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions,
        "ner": ner,
        "source": "RecipeNLG"
    }
    recipes.append(recipe)

print(f"Cleaned {len(recipes)} usable recipes.")

# ========== STEP 4: Save ==========
os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(recipes, f, indent=2, ensure_ascii=False)

print(f"Saved cleaned dataset to {OUTPUT_JSON}")

print("Sample cleaned recipe:")
print(json.dumps(recipes[0], indent=2, ensure_ascii=False))