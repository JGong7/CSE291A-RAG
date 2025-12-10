import json
import sys

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_gold_lookup(gold_data):
    """
    Build a mapping: query_id -> set of gold recipe IDs.
    New format:
        results: [ { "id": "recnlg_xxx", ... }, ... ]
    """
    lookup = {}
    for item in gold_data:
        qid = item["query_id"]
        gold_ids = {r["id"] for r in item["results"]}
        lookup[qid] = gold_ids
    return lookup

def compare_results(gold_json_path, rag_json_path):
    gold_data = load_json(gold_json_path)
    rag_data = load_json(rag_json_path)

    gold_lookup = build_gold_lookup(gold_data)

    total = 0
    matched = 0

    for item in rag_data:
        qid = item["query_id"]
        
        # RAG format: results: [ {"recipe": {"id": xxx}}, ... ]
        rag_ids = {r["recipe"]["id"] for r in item["results"]}

        gold_ids = gold_lookup.get(qid, set())

        hits = rag_ids.intersection(gold_ids)
        matched += len(hits)
        total += len(rag_ids)

        print(f"Query {qid}: matched {len(hits)} / {len(rag_ids)}")

    print("\n=============================")
    print(f"Total RAG returned recipes: {total}")
    print(f"Matched with gold answers:  {matched}")
    print(f"Match rate: {matched/total:.2%}")
    print("=============================")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_rag_results.py gold_answer.json rag_results.json")
        sys.exit(1)

    gold_file = sys.argv[1]
    rag_file = sys.argv[2]

    compare_results(gold_file, rag_file)