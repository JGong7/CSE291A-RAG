# llm_reranker.py
import json
import os

# ======== Check OpenAI availability  ========
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")


import re

def extract_json_list(text):
    """
    Extracts a JSON list from any LLM output.
    Example:
        "Here are the results: [1, 3, 4, 2, 5]" → [1,3,4,2,5]

    Returns None if no valid list is found.
    """

    if not text or not isinstance(text, str):
        return None

    # Extract the first [...] block
    match = re.search(r"\[[^\]]*\]", text.strip())
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None
    
def llm_rerank_results(
    query_text,
    retrieved_items,
    top_k=5,
    model="gpt-4o-mini",
    api_key=None
):
    """
    Uses an LLM to rerank retrieved recipe candidates and select the best top_k.
    Follows same API loading + fallback pattern as llm_query_processor.py
    """

    # ----------- No OpenAI? Fallback directly -----------
    if not OPENAI_AVAILABLE:
        print("Warning: OpenAI not available, skipping rerank (using FAISS results).")
        return retrieved_items[:top_k]

    # ----------- Initialize Official Client -----------
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print("Warning: Failed to initialize OpenAI client:", str(e))
        print("Using FAISS top_k results instead.")
        return retrieved_items[:top_k]

    # Build prompt for reranking
    recipe_descriptions = []
    for i, item in enumerate(retrieved_items):
        recipe = item["recipe"]
        title = recipe.get("title", "")
        ingredients = ", ".join(recipe.get("ingredients", []))[:500]
        desc = f"({i+1}) Title: {title}\nIngredients: {ingredients}\n"
        recipe_descriptions.append(desc)

    prompt = f"""
User query:
"{query_text}"

You are to rerank the following retrieved recipes.
Return ONLY the indices (1-based) of the best {top_k} candidates in JSON array format.

Candidates:
{'\n'.join(recipe_descriptions)}

JSON Output (example): [3, 1, 7, 2, 4]
"""
    print(prompt)

    # ----------- LLM call with fallback -----------
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        # FIX: extract content from ChatCompletionMessage object
        raw = response.choices[0].message.content
        # selected = json.loads(raw)
        selected = extract_json_list(raw)
        if not selected or not isinstance(selected, list):
            raise ValueError("LLM output did not contain a valid JSON list.")


    except Exception as e:
        print("Warning: LLM reranking failed:", str(e))
        print("Using FAISS top_k results instead.")
        return retrieved_items[:top_k]

    # Convert 1-based indices → 0-based
    selected = [i - 1 for i in selected if 1 <= i <= len(retrieved_items)]

    # Slice to top_k (LLM might return more)
    reranked = [retrieved_items[i] for i in selected[:top_k]]

    return reranked