"""
Comprehensive Evaluation Script for Hybrid Retrieval System
Generates metrics required for Phase 2A evaluation
"""

import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict


def load_manual_labels(manual_json_path: str) -> Dict[str, Dict]:
    """Load manual ground truth labels"""
    with open(manual_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping: query_id -> {recipe_id: valid}
    ground_truth = {}
    for item in data:
        qid = str(item['query_id'])
        recipe_labels = {}
        for result in item['results']:
            recipe_id = result.get('id', '')
            valid = int(result.get('valid', 0))
            recipe_labels[recipe_id] = valid
        ground_truth[qid] = recipe_labels
    
    return ground_truth


def label_retrieval_results(retrieval_path: str, ground_truth: Dict, output_path: str):
    """
    Label retrieval results based on ground truth
    Saves labeled results for evaluator
    """
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_results = json.load(f)
    
    labeled_results = []
    
    for query_result in retrieval_results:
        qid = str(query_result['query_id'])
        query_text = query_result['query']
        
        # Get ground truth for this query
        gt_labels = ground_truth.get(qid, {})
        
        labeled_recipes = []
        for result in query_result['results']:
            recipe = result['recipe']
            recipe_id = recipe.get('id', '')
            
            # Check if this recipe is valid according to ground truth
            valid = gt_labels.get(recipe_id, 0)
            
            labeled_recipes.append({
                'rank': result['rank'],
                'score': result['score'],
                'id': recipe_id,
                'title': recipe.get('title', ''),
                'valid': valid,
                'recipe': recipe  # Keep full recipe for reference
            })
        
        labeled_results.append({
            'query_id': int(qid),
            'query': query_text,
            'results': labeled_recipes
        })
    
    # Save labeled results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_results, f, indent=2, ensure_ascii=False)
    
    return labeled_results


def calculate_metrics(labeled_results: List[Dict], k_values=[1, 3, 5]) -> Dict:
    """Calculate all evaluation metrics"""
    
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
        import math
        dcg = 0.0
        for i in range(1, k + 1):
            gain = int(labels[i - 1])
            dcg += gain / math.log2(i + 1)
        return dcg
    
    def ndcg_at_k(labels: List[int], k: int) -> float:
        ideal = sorted(labels[:k], reverse=True)
        idcg = dcg_at_k(ideal, k)
        return (dcg_at_k(labels, k) / idcg) if idcg > 0 else 0.0
    
    # Calculate per-query metrics
    per_query_metrics = []
    
    # Aggregate sums
    sums = {
        "precision": {k: 0.0 for k in k_values},
        "hit_rate":  {k: 0.0 for k in k_values},
        "mrr":       {k: 0.0 for k in k_values},
        "map":       {k: 0.0 for k in k_values},
        "ndcg":      {k: 0.0 for k in k_values},
    }
    
    for query_result in labeled_results:
        qid = query_result['query_id']
        labels = [r['valid'] for r in query_result['results']]
        
        query_metrics = {'query_id': qid, 'query': query_result['query']}
        
        for k in k_values:
            query_metrics[f'P@{k}'] = precision_at_k(labels, k)
            query_metrics[f'HR@{k}'] = hit_rate_at_k(labels, k)
            query_metrics[f'MRR@{k}'] = mrr_at_k(labels, k)
            query_metrics[f'AP@{k}'] = average_precision_at_k(labels, k)
            query_metrics[f'NDCG@{k}'] = ndcg_at_k(labels, k)
            
            sums["precision"][k] += query_metrics[f'P@{k}']
            sums["hit_rate"][k] += query_metrics[f'HR@{k}']
            sums["mrr"][k] += query_metrics[f'MRR@{k}']
            sums["map"][k] += query_metrics[f'AP@{k}']
            sums["ndcg"][k] += query_metrics[f'NDCG@{k}']
        
        per_query_metrics.append(query_metrics)
    
    # Calculate macro averages
    n = len(labeled_results) or 1
    macro_avg = {
        "precision": {k: sums["precision"][k] / n for k in k_values},
        "hit_rate":  {k: sums["hit_rate"][k] / n for k in k_values},
        "mrr":       {k: sums["mrr"][k] / n for k in k_values},
        "map":       {k: sums["map"][k] / n for k in k_values},
        "ndcg":      {k: sums["ndcg"][k] / n for k in k_values},
    }
    
    return {
        'per_query': per_query_metrics,
        'macro_avg': macro_avg,
        'num_queries': n
    }


def print_metrics_table(metrics: Dict, system_name: str):
    """Print metrics in a formatted table"""
    print("\n" + "="*80)
    print(f"{system_name} - EVALUATION METRICS")
    print("="*80)
    
    # Per-query table
    print("\nPer-Query Metrics:")
    print("-" * 80)
    
    k_values = [1, 3, 5]
    header = ["QID"] + [f"P@{k}" for k in k_values] + [f"HR@{k}" for k in k_values] + \
             [f"MRR@{k}" for k in k_values] + [f"AP@{k}" for k in k_values] + \
             [f"NDCG@{k}" for k in k_values]
    print("\t".join(header))
    
    for qm in metrics['per_query']:
        row = [str(qm['query_id'])]
        for k in k_values:
            row.append(f"{qm[f'P@{k}']:.3f}")
        for k in k_values:
            row.append(f"{qm[f'HR@{k}']:.3f}")
        for k in k_values:
            row.append(f"{qm[f'MRR@{k}']:.3f}")
        for k in k_values:
            row.append(f"{qm[f'AP@{k}']:.3f}")
        for k in k_values:
            row.append(f"{qm[f'NDCG@{k}']:.3f}")
        print("\t".join(row))
    
    # Macro averages
    print("\nMacro-Average Metrics:")
    print("-" * 80)
    ma = metrics['macro_avg']
    print("PRECISION: " + ", ".join(f"@{k}={ma['precision'][k]:.3f}" for k in k_values))
    print("HIT_RATE : " + ", ".join(f"@{k}={ma['hit_rate'][k]:.3f}" for k in k_values))
    print("MRR      : " + ", ".join(f"@{k}={ma['mrr'][k]:.3f}" for k in k_values))
    print("MAP      : " + ", ".join(f"@{k}={ma['map'][k]:.3f}" for k in k_values))
    print("NDCG     : " + ", ".join(f"@{k}={ma['ndcg'][k]:.3f}" for k in k_values))
    print("="*80)


def compare_systems(baseline_metrics: Dict, hybrid_metrics: Dict):
    """Compare baseline vs hybrid system"""
    print("\n" + "="*80)
    print("SYSTEM COMPARISON: Baseline vs Hybrid")
    print("="*80)
    
    k_values = [1, 3, 5]
    
    baseline_ma = baseline_metrics['macro_avg']
    hybrid_ma = hybrid_metrics['macro_avg']
    
    print("\nMetric Improvements (Hybrid vs Baseline):")
    print("-" * 80)
    print(f"{'Metric':<15} | {'Baseline':<25} | {'Hybrid':<25} | {'Improvement':<15}")
    print("-" * 80)
    
    for metric_name in ['precision', 'hit_rate', 'mrr', 'map', 'ndcg']:
        baseline_str = ", ".join(f"@{k}={baseline_ma[metric_name][k]:.3f}" for k in k_values)
        hybrid_str = ", ".join(f"@{k}={hybrid_ma[metric_name][k]:.3f}" for k in k_values)
        
        # Calculate average improvement
        avg_baseline = sum(baseline_ma[metric_name][k] for k in k_values) / len(k_values)
        avg_hybrid = sum(hybrid_ma[metric_name][k] for k in k_values) / len(k_values)
        improvement = ((avg_hybrid - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
        
        print(f"{metric_name.upper():<15} | {baseline_str:<25} | {hybrid_str:<25} | {improvement:+.1f}%")
    
    print("="*80)


def measure_speed(retrieval_function, queries: List, num_runs: int = 3) -> Dict:
    """Measure average query latency"""
    latencies = []
    
    for _ in range(num_runs):
        start = time.time()
        for query in queries:
            retrieval_function(query)
        end = time.time()
        
        total_time = end - start
        avg_latency = total_time / len(queries)
        latencies.append(avg_latency)
    
    return {
        'avg_latency_ms': sum(latencies) / len(latencies) * 1000,
        'min_latency_ms': min(latencies) * 1000,
        'max_latency_ms': max(latencies) * 1000
    }


def estimate_cost(num_queries: int, embedding_dim: int = 384, 
                  recipes_count: int = 175000) -> Dict:
    """Estimate computational costs"""
    
    # Embedding generation cost (one-time)
    embedding_compute_flops = recipes_count * embedding_dim * 1000  # Rough estimate
    
    # Storage cost
    embedding_storage_mb = (recipes_count * embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float
    
    # Query cost (per query)
    query_embedding_flops = embedding_dim * 1000
    faiss_search_flops = recipes_count * embedding_dim  # Rough estimate for brute force
    
    total_query_flops = num_queries * (query_embedding_flops + faiss_search_flops)
    
    return {
        'one_time_embedding_gflops': embedding_compute_flops / 1e9,
        'embedding_storage_mb': embedding_storage_mb,
        'per_query_flops': query_embedding_flops + faiss_search_flops,
        'total_query_gflops': total_query_flops / 1e9,
        'estimated_total_gflops': (embedding_compute_flops + total_query_flops) / 1e9
    }


def main():
    """Main evaluation pipeline"""
    
    print("="*80)
    print("PHASE 2A EVALUATION - HYBRID RETRIEVAL SYSTEM")
    print("="*80)
    
    # Paths
    MANUAL_JSON = "manual.json"
    BASELINE_RESULTS = "retrieval_results/faiss_fusion_results.json"
    HYBRID_RESULTS = "retrieval_results/hybrid_results.json"
    
    LABELED_BASELINE = "retrieval_results/labeled_baseline_results.json"
    LABELED_HYBRID = "retrieval_results/labeled_hybrid_results.json"
    
    # Step 1: Load ground truth
    print("\n📋 Loading ground truth labels...")
    ground_truth = load_manual_labels(MANUAL_JSON)
    print(f"   Loaded ground truth for {len(ground_truth)} queries")
    
    # Step 2: Label baseline results
    print("\n🏷️  Labeling baseline results...")
    baseline_labeled = label_retrieval_results(BASELINE_RESULTS, ground_truth, LABELED_BASELINE)
    print(f"   Saved labeled baseline to {LABELED_BASELINE}")
    
    # Step 3: Label hybrid results
    print("\n🏷️  Labeling hybrid results...")
    hybrid_labeled = label_retrieval_results(HYBRID_RESULTS, ground_truth, LABELED_HYBRID)
    print(f"   Saved labeled hybrid to {LABELED_HYBRID}")
    
    # Step 4: Calculate metrics for baseline
    print("\n📊 Calculating baseline metrics...")
    baseline_metrics = calculate_metrics(baseline_labeled)
    print_metrics_table(baseline_metrics, "BASELINE SYSTEM")
    
    # Step 5: Calculate metrics for hybrid
    print("\n📊 Calculating hybrid metrics...")
    hybrid_metrics = calculate_metrics(hybrid_labeled)
    print_metrics_table(hybrid_metrics, "HYBRID SYSTEM")
    
    # Step 6: Compare systems
    compare_systems(baseline_metrics, hybrid_metrics)
    
    # Step 7: Speed analysis
    print("\n⏱️  Speed Analysis:")
    print("-" * 80)
    print("Note: Run hybrid_retrieval.py with timing enabled for accurate measurements")
    print("Estimated per-query latency:")
    print("  - Baseline: ~0.3-0.5 seconds")
    print("  - Hybrid: ~0.4-0.6 seconds (includes metadata filtering)")
    
    # Step 8: Cost estimation
    print("\n💰 Cost Estimation:")
    print("-" * 80)
    num_queries = len(hybrid_labeled)
    costs = estimate_cost(num_queries)
    print(f"  - One-time embedding generation: {costs['one_time_embedding_gflops']:.2f} GFLOPs")
    print(f"  - Embedding storage: {costs['embedding_storage_mb']:.2f} MB")
    print(f"  - Per-query compute: {costs['per_query_flops']/1e6:.2f} MFLOPs")
    print(f"  - Total query compute ({num_queries} queries): {costs['total_query_gflops']:.2f} GFLOPs")
    print(f"  - Estimated total: {costs['estimated_total_gflops']:.2f} GFLOPs")
    
    # Step 9: Save comprehensive report
    report = {
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_queries': num_queries,
        'baseline_metrics': baseline_metrics,
        'hybrid_metrics': hybrid_metrics,
        'cost_estimates': costs
    }
    
    with open('retrieval_results/evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Evaluation complete!")
    print("📄 Full report saved to: retrieval_results/evaluation_report.json")
    print("="*80)


if __name__ == "__main__":
    main()
