import requests
import json
import os
import time
from tqdm import tqdm

API_KEY = "0817dd0e857e4552b48d09988a9ee17f"
SAVE_PATH = "spoonacular_dataset.json"
N_RECIPES = 100
DELAY = 1.2  # seconds between requests
RETRY_LIMIT = 2  # max retry per recipe
SAVE_INTERVAL = 10  # auto-save every N recipes

def get_random_recipe():
    """Fetch one random recipe with retries and timeout."""
    for attempt in range(RETRY_LIMIT):
        try:
            url = f"https://api.spoonacular.com/recipes/random?number=1&apiKey={API_KEY}"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if "recipes" in data and len(data["recipes"]) > 0:
                    r = data["recipes"][0]
                    return {
                        "id": r.get("id", None),
                        "title": r.get("title", ""),
                        "ingredients": [ing["original"] for ing in r.get("extendedIngredients", [])],
                        "instructions": r.get("instructions", "") or "",
                        "source": "Spoonacular"
                    }
            elif response.status_code in [402, 429]:
                # 402 = quota exceeded, 429 = too many requests
                print(f"\n Rate limit reached ({response.status_code}).")
                time.sleep(3)
                print("Forced Exit due to API limits.")
                exit(1)
            else:
                print(f"Unexpected status {response.status_code}, retrying ({attempt+1}/{RETRY_LIMIT})...")
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"Timeout, retrying ({attempt+1}/{RETRY_LIMIT})...")
            time.sleep(3)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
    return None

# ---------- Load Existing ----------
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {r["id"] for r in existing if r.get("id")}
    print(f"Loaded {len(existing)} existing recipes.")
else:
    existing, existing_ids = [], set()
    print("Starting new dataset...")

# ---------- Fetch ----------
new_recipes = []
for i in tqdm(range(N_RECIPES), desc="Fetching recipes"):
    rec = get_random_recipe()
    if rec and rec.get("id") and rec["id"] not in existing_ids:
        new_recipes.append(rec)
        existing_ids.add(rec["id"])
    else:
        print("Duplicate or invalid recipe skipped.")
    # Auto-save every N_FETCH steps
    if (i + 1) % SAVE_INTERVAL == 0 and new_recipes:
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(existing + new_recipes, f, indent=2, ensure_ascii=False)
        print(f"Auto-saved progress ({len(existing) + len(new_recipes)} total).")
    time.sleep(DELAY)

# ---------- Final Save ----------
if new_recipes:
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(existing + new_recipes, f, indent=2, ensure_ascii=False)
    print(f"Finished run: {len(new_recipes)} new recipes. Total {len(existing) + len(new_recipes)}.")
else:
    print("No new unique recipes fetched.")
