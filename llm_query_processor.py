import os
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

