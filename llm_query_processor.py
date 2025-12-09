import os
import json
from typing import Any, Dict, List

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")


def structure_query_with_llm(query_text, model="gpt-4o-mini", api_key=None):
    """
    Use LLM to structure and optimize a recipe query for better retrieval.
    
    Args:
        query_text: The original user query
        model: OpenAI model to use (default: gpt-4o-mini)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    Returns:
        Structured query string optimized for recipe retrieval
    """
    if not OPENAI_AVAILABLE:
        print("Warning: OpenAI not available, using original query")
        return query_text
    
    # Try to load API key from .env.local file if not provided
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            with open('.env.local', 'r') as f:
                line = f.readline().strip()
                key, value = line.split('=', 1)
                if key.strip() == 'OPENAI_API_KEY':
                    api_key = value.strip().strip('"').strip("'")
        except Exception:
            pass
    
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, using original query")
        return query_text
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are a recipe search assistant. Your task is to transform a user's recipe query into an optimized search query that will help find the best matching recipes.

Original query: {query_text}

Transform this query to:
1. Extract key ingredients, dietary requirements, cooking methods, and preferences
2. Expand implicit requirements (e.g., "lactose intolerance" -> "dairy-free", "no milk", "no cheese")
3. Clarify ambiguous terms (e.g., "spicy" -> "hot", "chili", "pepper")
4. Include relevant synonyms and related terms
5. Keep the query concise but comprehensive

Return ONLY the optimized search query text, nothing else. Do not include explanations or additional text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful recipe search assistant that optimizes queries for better recipe retrieval."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        structured_query = response.choices[0].message.content.strip()
        return structured_query
    except Exception as e:
        print(f"Error structuring query with LLM: {e}, using original query")
        return query_text


def call_llm(prompt: str, model: str = "gpt-4o-mini", api_key: str | None = None) -> str:
    """Call the configured LLM with a simple prompt and return raw text.

    This reuses the same OpenAI client and API-key loading pattern as
    structure_query_with_llm so rewrite_query_to_structured can call it.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI package not installed; cannot call LLM")

    # Try to load API key from env or .env.local
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            with open(".env.local", "r") as f:
                line = f.readline().strip()
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    api_key = value.strip().strip("\"").strip("'")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call LLM")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that strictly outputs valid JSON when asked.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )

    return response.choices[0].message.content or ""


def process_queries_with_llm(queries, model="gpt-4o-mini", api_key=None, use_llm=True):
    """
    Process a list of queries with LLM structuring.
    
    Args:
        queries: List of query dictionaries with 'id' and 'query' keys
        model: OpenAI model to use
        api_key: OpenAI API key
        use_llm: Whether to use LLM structuring (if False, returns original queries)
    
    Returns:
        List of queries with structured 'query' field
    """
    if not use_llm:
        return queries
    
    processed_queries = []
    for q in queries:
        original_query = q["query"]
        structured_query = structure_query_with_llm(original_query, model, api_key)
        processed_queries.append({
            "id": q["id"],
            "query": structured_query,
            "original_query": original_query  # Keep original for reference
        })
    
    return processed_queries


def _build_structured_prompt(query: str) -> str:
    """Prompt template to ask LLM for a structured interpretation of the query."""

    return f"""You are a helpful assistant that converts a natural language recipe search query
into a strictly formatted JSON object that captures user intent.

Query: "{query}"

Return a JSON object with the following schema (no extra text):
{{
  "dietary_tags": ["vegan" | "vegetarian" | "lactose-free" | "gluten-free" | "nut-free"],
  "must_have_ingredients": ["egg", "chicken", "curry", ...],
  "avoid_ingredients": ["lactose", "gluten", "meat", "seafood", "nuts", "eggs"],
  "meal_types": ["breakfast" | "lunch" | "dinner" | "dessert" | "soup" | "pasta" | "salad" | "appetizer" | "beverage" | "smoothie"]  (optional if not specified),
  "quantity_constraints": {{
    "ingredient": {{
      "max": float (optional),
      "min": float (optional),
      "kind": "weight" | "volume" | "count" | null
    }},
    ...
  }},
  "general_ingredient_tags": ["vegetable" | "meat" | "fruit" | "dairy" | "spice"] (optional if not specified),
  "cooking_methods": ["baked" | "fried" | "grilled" | "boiled" | "slow-cooked"] (optional if not specified),
  "notes": "free-form explanation of how you interpreted the query"
}}

Rules:
- Always output valid JSON.
- If some field is unknown, use an empty list or empty object, not null.
- Try to understand cultural or regional terms (e.g., "halal", "Irish", etc.) and include or exclude the ingredients accordingly.
- For quantity_constraints, use canonical ingredient names (singular, lowercase),
  and only add entries if the query clearly expresses a limit such as "I only have",
  "at most", "at least", "no more than", "I have 500 g of flour", etc.
  "at least" means min is set and Don't set max!!!
  "at most" / "Using" / "I (only) have" means max is set and Don't set min!!!
- If "OR" logic appears in the query, use " or " connect ingredients in must_have_ingredients
- Do not include general ingredient tags (without plural form) in "must_have_ingredients".
- Do not include avoid_ingredients in "general_ingredient_tags".
- Do not invent ingredients or constraints that are not implied by the query.
"""  # noqa: E501


def rewrite_query_to_structured(query: str, llm_model: str = "gpt-4o-mini", use_llm: bool = True) -> Dict[str, Any]:
    """Rewrite a natural language query into a structured representation.

    Returns a dict with keys:
      - dietary_tags: List[str]
      - must_have_ingredients: List[str]
      - avoid_ingredients: List[str]
      - meal_types: List[str]
      - quantity_constraints: Dict[str, Dict[str, Any]]
      - notes: str

    When use_llm=False, returns a mostly-empty baseline structure that can be
    filled by rule-based components.
    """

    if not use_llm:
        return {
            "dietary_tags": [],
            "must_have_ingredients": [],
            "avoid_ingredients": [],
            "meal_types": [],
            "quantity_constraints": {},
            "cooking_methods": [],
            "notes": "rule-based fallback, no LLM used",
        }

    prompt = _build_structured_prompt(query)

    # Use the local call_llm defined above instead of self-importing this module
    raw = call_llm(prompt, model=llm_model)

    try:
        structured = json.loads(raw)
    except Exception:
        # Fallback: in case JSON is not strictly valid, return a safe skeleton
        structured = {
            "dietary_tags": [],
            "must_have_ingredients": [],
            "avoid_ingredients": [],
            "meal_types": [],
            "quantity_constraints": {},
            "cooking_methods": [],
            "notes": f"failed_to_parse_json: {raw[:200]}",
        }

    # Ensure required keys exist with expected types
    structured.setdefault("dietary_tags", [])
    structured.setdefault("must_have_ingredients", [])
    structured.setdefault("avoid_ingredients", [])
    structured.setdefault("meal_types", [])
    structured.setdefault("quantity_constraints", {})
    structured.setdefault("cooking_methods", [])
    structured.setdefault("notes", "")

    return structured


def rewrite_query_text(query: str, structured: Dict[str, Any]) -> str:
    """Optionally build a rewritten textual query from the structured view.

    This is a light-weight way to let existing rule-based filters (MetadataFilter,
    QuantityFilter) see a more explicit form, without changing their APIs.
    """

    parts: List[str] = [query]

    must = structured.get("must_have_ingredients") or []
    avoid = structured.get("avoid_ingredients") or []
    dietary = structured.get("dietary_tags") or []
    meal_types = structured.get("meal_types") or []
    quantity_constraints = structured.get("quantity_constraints") or {}
    cooking_methods = structured.get("cooking_methods") or []

    if must:
        parts.append("must contain: " + ", ".join(must))
    if avoid:
        parts.append("avoid: " + ", ".join(avoid))
    if dietary:
        parts.append("dietary: " + ", ".join(dietary))
    if meal_types:
        parts.append("meal types: " + ", ".join(meal_types))
    if quantity_constraints:
        for ingredient, constraint in quantity_constraints.items():
            q_parts = []
            if "max" in constraint and constraint["max"] is not None:
                add_part = f"have {constraint['max']}"
                if "kind" in constraint and constraint["kind"] != "count":
                    add_part += f" {constraint['kind']} {ingredient}"
                else:
                    add_part += f" {ingredient}"

                q_parts.append(add_part)
            if q_parts:
                parts.append(f"{ingredient}: " + ", ".join(q_parts))
    if cooking_methods:
        parts.append("cooking methods: " + ", ".join(cooking_methods))

    return "; ".join(p.strip() for p in parts if p and p.strip())

