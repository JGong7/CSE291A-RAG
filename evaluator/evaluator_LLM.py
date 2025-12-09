#!/usr/bin/env python
"""

Usage:
    python compare_rag_results_llm_bool.py manual.json rag_results.json

Requirements:
    pip install --upgrade openai
    export OPENAI_API_KEY=...

"""

import json
import sys
import time
from typing import Dict, Set, List
from openai import OpenAI

client = OpenAI()

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gold_lookup(gold_data: List[dict]) -> Dict[int, Set[str]]:
    """
    Build a mapping: query_id -> set of gold recipe IDs.
    New format:
        results: [ { "id": "recnlg_xxx", ... }, ... ]
    """
    lookup: Dict[int, Set[str]] = {}
    for item in gold_data:
        qid = item["query_id"]
        gold_ids = {r["id"] for r in item["results"]}
        lookup[qid] = gold_ids
    return lookup


# ---------- LLM judge: returns True/False only ----------

def llm_batch_is_reasonable(query_text: str,
                            recipe_list: List[dict],
                            model: str = "gpt-4.1-mini") -> List[bool]:
    """
    Ask the LLM, in one shot, whether each recipe is a reasonable answer.

    Returns:
        A list of booleans of length len(recipe_list),
        where result[i] corresponds to recipe_list[i].
    """

    n = len(recipe_list)
    if n == 0:
        return []

    # Build a compact but clear prompt
    recipes_block = []
    for idx, r in enumerate(recipe_list, start=1):
        recipes_block.append(f"Recipe {idx}:\n{json.dumps(r, indent=2)}")
    recipes_block_str = "\n\n".join(recipes_block)

    prompt = f"""
    You are evaluating whether each candidate recipe is a reasonable answer to the user's query.

    Criteria:
    - The recipe must fit what the user asked for (e.g., breakfast, soup, quick snack).
    - It must respect important constraints in the query (e.g. "only two eggs", "quick", "vegetarian", "no bacon").
    - It should be something a normal user would accept as a sensible response.

    User query:
    {query_text}

    Candidate recipes:
    {recipes_block_str}

    For EACH recipe 1..{n}, output whether it is reasonable.
    Respond with EXACTLY one JSON array of booleans of length {n}, like:

    [true, false, true, ...]

    No extra text, no keys, no explanations.
    """.strip()

    # Make a single API call for all recipes for this query
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
        max_output_tokens=50,
    )

    raw = response.output[0].content[0].text.strip()

    # Try to extract the JSON array [ ... ]
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        json_str = raw[start:end]
        arr = json.loads(json_str)
        # Ensure it's the right length; if not, fallback
        if not isinstance(arr, list) or len(arr) != n:
            return [False] * n
        return [bool(x) for x in arr]
    except Exception:
        # If anything goes wrong, treat all as not reasonable
        return [False] * n



# ---------- Main evaluation: gold OR LLM (binary) ----------

def compare_results(gold_json_path: str, rag_json_path: str,
                    max_results_per_query: int = None):
    """
    Batched version:
    - One LLM call per query (instead of per recipe).
    - Avoids rate limit issues.
    - Adds a time.sleep() after each query to stay below 3 RPM.
    """
    gold_data = load_json(gold_json_path)
    rag_data = load_json(rag_json_path)

    gold_lookup = build_gold_lookup(gold_data)

    total = 0
    matched_total = 0      # gold OR LLM
    matched_gold = 0
    matched_llm_only = 0

    for item in rag_data:
        qid = item["query_id"]
        query_text = item.get("query", "")
        rag_results = item["results"]

        if max_results_per_query is not None:
            rag_results = rag_results[:max_results_per_query]

        gold_ids = gold_lookup.get(qid, set())
        retrieved_this_q = len(rag_results)

        # ---- Build recipe list for batching ----
        recipe_list = [r["recipe"] for r in rag_results]

        # ---- One LLM call per query ----
        llm_flags = llm_batch_is_reasonable(query_text, recipe_list)

        gold_hits_this_q = 0
        llm_hits_this_q = 0

        # ---- Evaluate each recipe ----
        for idx, r in enumerate(rag_results):
            total += 1
            rid = r["recipe"]["id"]
            llm_ok = llm_flags[idx] if idx < len(llm_flags) else False

            if rid in gold_ids:
                matched_total += 1
                matched_gold += 1
                gold_hits_this_q += 1
            elif llm_ok:
                matched_total += 1
                matched_llm_only += 1
                llm_hits_this_q += 1

        print(
            f"Query {qid}: "
            f"gold hits={gold_hits_this_q}, "
            f"LLM-only hits={llm_hits_this_q}, "
            f"matched={gold_hits_this_q + llm_hits_this_q} / {retrieved_this_q}"
        )

        # ---- SLEEP to avoid rate-limit ----
        time.sleep(21)

    print("\n=========== Gold OR LLM evaluation (batched) ===========")
    print(f"Total retrieved recipes:           {total}")
    print(f"Gold ID hits:                      {matched_gold}")
    print(f"LLM-only hits (not in gold set):   {matched_llm_only}")
    print(f"Total hits (gold ∪ LLM):           {matched_total}")
    gold_rate = matched_gold / total if total else 0.0
    total_rate = matched_total / total if total else 0.0
    print(f"Gold-only hit rate:                {gold_rate:.2%}")
    print(f"Augmented hit rate (gold ∪ LLM):   {total_rate:.2%}")
    print("========================================================\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_rag_results_llm_bool.py gold_answer.json rag_results.json")
        sys.exit(1)

    gold_file = sys.argv[1]
    rag_file = sys.argv[2]

    # single pass with the new gold+LLM 0/1 logic
    compare_results(
        gold_file,
        rag_file,
        max_results_per_query=None,   # set e.g. 5 for top-5 per query if you want
    )
