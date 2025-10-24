import json, math
from typing import Dict, List

# ======== METRIC DEFINITIONS ========
METRIC_DEFINITIONS = {
    "Precision@k": (
        "Fraction of the top-k retrieved items that are relevant. "
        "Precision@1 tests if the very top result is correct; Precision@5 measures "
        "how many of the first five results are relevant."
    ),
    "HitRate@k": (
        "Binary metric indicating whether at least one relevant item appears within "
        "the top-k results. 1 if any relevant result is found, otherwise 0."
    ),
    "MRR@k": (
        "Mean Reciprocal Rank: the reciprocal of the rank position of the first "
        "relevant result within the top-k. If the first relevant result is at rank 1, "
        "MRR=1. If at rank 5, MRR=1/5."
    ),
    "AP@k": (
        "Average Precision: the mean of precisions computed at each position where a "
        "relevant item occurs, within the top-k. Captures both precision and the order "
        "of relevant items. Averaged across queries to form MAP (Mean Average Precision)."
    ),
    "NDCG@k": (
        "Normalized Discounted Cumulative Gain: measures ranking quality by assigning "
        "higher weights to relevant items appearing earlier. DCG discounts relevance "
        "scores logarithmically by rank; NDCG normalizes it by the ideal ranking's DCG."
    ),
}

# ======== CONFIG ========
JSON_PATH = "retrieval_results/faiss_fusion_results.json"
KS = (1, 3, 5)  # fixed cutoffs since you always retrieve 5
# ========================

# --------- Metric functions (no min(k, len(labels))) ---------
def precision_at_k(labels: List[int], k: int) -> float:
    return sum(labels[:k]) / k if k > 0 else 0.0

def hit_rate_at_k(labels: List[int], k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0

def mrr_at_k(labels: List[int], k: int) -> float:
    for i, y in enumerate(labels[:k], start=1):
        if int(y) == 1:
            return 1.0 / i
    return 0.0

def average_precision_at_k(labels: List[int], k: int) -> float:
    num_rel, ap = 0, 0.0
    for i in range(1, k + 1):
        if int(labels[i - 1]) == 1:
            num_rel += 1
            ap += num_rel / i
    return ap / num_rel if num_rel > 0 else 0.0

def dcg_at_k(labels: List[int], k: int) -> float:
    dcg = 0.0
    for i in range(1, k + 1):
        gain = int(labels[i - 1])
        dcg += gain / math.log2(i + 1)  # rank 1 -> /1
    return dcg

def ndcg_at_k(labels: List[int], k: int) -> float:
    ideal = sorted(labels[:k], reverse=True)
    idcg = dcg_at_k(ideal, k)
    return (dcg_at_k(labels, k) / idcg) if idcg > 0 else 0.0

# --------- Load + parse JSON into {query_id: [labels]} ---------
def load_labels_by_query(path: str) -> Dict[str, List[int]]:
    """
    Expects an array of items like:
      { "query_id": 1, "query": "...",
        "results": [ {"rank":1, "valid":1, ...}, ... ] }
    Returns: { "1": [1,0,1,0,0], ... } (results sorted by rank if present)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data
    if isinstance(data, dict):
        for key in ("queries", "results", "data"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
    if not isinstance(items, list):
        raise ValueError("Unexpected JSON structure; expected a list of query blocks.")

    labels_by_query: Dict[str, List[int]] = {}
    for q in items:
        qid = str(q.get("query_id", q.get("id", q.get("query", "unknown"))))
        results = q.get("results", [])
        if results and isinstance(results[0], dict) and "rank" in results[0]:
            results = sorted(results, key=lambda r: r.get("rank", 10**9))
        labels = [int(r.get("valid", 0)) for r in results]  # should be length 5
        if len(labels) != 5:
            raise ValueError(f"Query {qid} does not have exactly 5 results.")
        labels_by_query[qid] = labels
    return labels_by_query

# --------- Evaluation + printing ---------
def evaluate_and_print(labels_by_query: Dict[str, List[int]]):
    print("\nPer-query metrics")
    print("-" * 60)
    header = ["query_id"] + [f"P@{k}" for k in KS] + [f"HR@{k}" for k in KS] \
             + [f"MRR@{k}" for k in KS] + [f"AP@{k}" for k in KS] + [f"NDCG@{k}" for k in KS]
    print("\t".join(header))

    # macro sums
    sums = {
        "precision": {k: 0.0 for k in KS},
        "hit_rate":  {k: 0.0 for k in KS},
        "mrr":       {k: 0.0 for k in KS},
        "map":       {k: 0.0 for k in KS},
        "ndcg":      {k: 0.0 for k in KS},
    }

    for qid, labels in labels_by_query.items():
        row_vals = []
        for k in KS: row_vals.append(precision_at_k(labels, k)); sums["precision"][k] += row_vals[-1]
        for k in KS: row_vals.append(hit_rate_at_k(labels, k));  sums["hit_rate"][k]  += row_vals[-1]
        for k in KS: row_vals.append(mrr_at_k(labels, k));       sums["mrr"][k]       += row_vals[-1]
        for k in KS: row_vals.append(average_precision_at_k(labels, k)); sums["map"][k] += row_vals[-1]
        for k in KS: row_vals.append(ndcg_at_k(labels, k));      sums["ndcg"][k]      += row_vals[-1]

        # pretty print row
        row_str = [qid] + [f"{v:.3f}" for v in row_vals]
        print("\t".join(row_str))

    n = len(labels_by_query) or 1
    print("\nMacro-average summary")
    print("-" * 60)
    print("PRECISION: " + ", ".join(f"@{k}={sums['precision'][k]/n:.3f}" for k in KS))
    print("HIT_RATE : " + ", ".join(f"@{k}={sums['hit_rate'][k]/n:.3f}"  for k in KS))
    print("MRR      : " + ", ".join(f"@{k}={sums['mrr'][k]/n:.3f}"       for k in KS))
    print("MAP      : " + ", ".join(f"@{k}={sums['map'][k]/n:.3f}"       for k in KS))
    print("NDCG     : " + ", ".join(f"@{k}={sums['ndcg'][k]/n:.3f}"      for k in KS))

def main():
    labels_by_query = load_labels_by_query(JSON_PATH)
    evaluate_and_print(labels_by_query)

if __name__ == "__main__":
    main()
