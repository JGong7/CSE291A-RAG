# Recipe Retrieval

### Overview
This project implements a **Retrieval-Augmented Generation (RAG)** baseline system for the **recipe domain**, integrating two complementary data sources:

- **[RecipeNLG Dataset](https://www.kaggle.com/code/paultimothymooney/explore-recipe-nlg-dataset/input)** – a large open dataset of structured recipes  
- **[Spoonacular API](https://spoonacular.com/food-api)** – real-world, API-fetched recipes with ingredients and cooking steps  

The system encodes recipes into dense vector representations using **SentenceTransformers**, indexes them using **FAISS**, and retrieves top relevant recipes for given user queries.

---

### Project Structure

project_root/
- baseline.py  # Embedding and FAISS retrieval pipeline
- manual_queries.json # 10 manually created test queries
- requirements.txt
- RecipeNLG_dataset
    - recipes_nlg_clean.json # Cleaned RecipeNLG dataset JSON with 5000 recipes
    - clean.py # Script to clean RecipeNLG dataset
    - RecipeNLG_dataset.csv # Original RecipeNLG dataset CSV, Download if needed
- Spoonacular_API
    - spoonacular_dataset.json # Fetched Spoonacular recipes JSON
    - spoonacular_fetch.py # Script to fetch recipes from Spoonacular API
- retrieval_results
    - RecipeNLG_faiss_results.json
    - Spoonacular_faiss_results.json

---

### Setup Instructions
```bash
conda create -n recipe_rag python=3.12
conda activate recipe_rag
pip install -r requirements.txt

# Optional: fetch Spoonacular API and append to existing dataset
python Spoonacular_API/fetch_spoonacular.py
```
