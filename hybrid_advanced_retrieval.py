"""Advanced modular retriever for recipe RAG.

Implements the following building blocks in a composable way:
1. Dense retriever (FAISS + sentence-transformers)
2. Sparse retriever (BM25 / keyword-based)
3. Fusion module (Reciprocal Rank Fusion)
4. Filter manager (metadata + quantity + nutrition with hard/soft modes)
5. Query profiler (rule/LLM-pluggable, decides config knobs like top-k, rerank)
6. Semantic cache (approximate query result caching by embedding similarity)

This file does NOT replace the existing `HybridRetriever` directly.
Instead, it offers a more modular API that other scripts can import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_retrieval import MetadataExtractor, MetadataFilter
from quantity_filter import QuantityFilter
from nutrition_filter import NutritionFilter

try:
    # Optional dependency for BM25 sparse retrieval
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional
    BM25Okapi = None


# =====================
# 0. Shared data models
# =====================

@dataclass
class RetrievalConfig:
    """Config knobs decided by QueryProfiler or caller.

    This encapsulates the METIS-style dynamic configuration.
    """

    # Retrieval knobs
    first_stage_top_k: int = 50
    final_top_k: int = 5
    use_dense: bool = True
    use_sparse: bool = True
    use_fusion: bool = True

    # Filtering knobs
    filter_mode: str = "hard"  # "hard" or "soft" (soft may relax filters if empty)
    use_metadata_filter: bool = True
    use_quantity_filter: bool = True

    # Reranking / reasoning knobs
    use_reranker: bool = False

    # Semantic cache
    use_cache: bool = True


@dataclass
class RetrievedItem:
    recipe_index: int
    score: float
    rank: int
    source: str  # "dense" / "sparse" / "fusion" / "cache"


@dataclass
class RetrievalResult:
    query: str
    items: List[RetrievedItem]
    config: RetrievalConfig
    profile: Dict[str, Any] = field(default_factory=dict)


# =====================
# 1. Dense retriever
# =====================

class DenseRetriever:
    """Dense vector retriever on top of FAISS.

    Can work on full corpus or a subset of indices.
    """

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        # Ensure embeddings are L2-normalized for inner product similarity
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
            faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int, candidate_indices: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Search dense index.

        Returns list of (global_recipe_index, score).
        If candidate_indices is provided, will search only within that subset.
        """

        if candidate_indices is None:
            # Full index search
            q = query_embedding.astype(np.float32)
            q = np.expand_dims(q, axis=0)
            faiss.normalize_L2(q)
            k = min(top_k, self.embeddings.shape[0])
            D, I = self.index.search(q, k)
            return [(int(I[0][i]), float(D[0][i])) for i in range(k)]

        # Subset search: build a temporary index over candidate embeddings
        cand_emb = self.embeddings[candidate_indices]
        dim = cand_emb.shape[1]
        tmp_index = faiss.IndexFlatIP(dim)
        tmp_index.add(cand_emb)

        q = query_embedding.astype(np.float32)
        q = np.expand_dims(q, axis=0)
        faiss.normalize_L2(q)
        k = min(top_k, cand_emb.shape[0])
        D, I = tmp_index.search(q, k)

        results: List[Tuple[int, float]] = []
        for i in range(k):
            global_idx = candidate_indices[int(I[0][i])]
            results.append((global_idx, float(D[0][i])))
        return results


# =====================
# 2. Sparse retriever (BM25)
# =====================

class SparseRetriever:
    """Keyword-based sparse retriever using BM25.

    Falls back to a no-op mode if rank_bm25 is not installed.
    """

    def __init__(self, recipes: List[Dict[str, Any]], metadata_list: List[Dict[str, Any]]):
        self.recipes = recipes
        self.metadata_list = metadata_list
        self.enabled = BM25Okapi is not None
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if self.enabled:
            self._build_corpus()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _build_corpus(self) -> None:
        """Build tokenized corpus from recipes for BM25.

        Simple design: use title + ingredients_text as the sparse document.
        """
        docs: List[List[str]] = []
        for meta in self.metadata_list:
            title = meta.get("title", "")
            ingredients_text = meta.get("ingredients_text", "")
            text = f"{title} {ingredients_text}".strip()
            docs.append(self._tokenize(text))
        self.tokenized_docs = docs
        self.bm25 = BM25Okapi(self.tokenized_docs) if BM25Okapi is not None else None

    def search(self, query: str, top_k: int, candidate_indices: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if not self.enabled or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        if candidate_indices is None:
            # Global search
            idx_scores = list(enumerate(scores))
        else:
            idx_scores = [(idx, scores[idx]) for idx in candidate_indices]

        # Sort by score descending and cut top_k
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        idx_scores = idx_scores[: min(top_k, len(idx_scores))]
        return [(int(i), float(s)) for i, s in idx_scores]


# =====================
# 3. Fusion (Reciprocal Rank Fusion)
# =====================

class FusionModule:
    """Implements Reciprocal Rank Fusion (RRF) for combining ranked lists."""

    def __init__(self, k0: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.k0 = k0
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense: List[Tuple[int, float]],
        sparse: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """Fuse two ranked lists with RRF.

        Inputs are lists of (doc_id, score) sorted by descending score.
        Returns fused list of (doc_id, fused_score).
        """
        rank_d = {doc_id: rank for rank, (doc_id, _) in enumerate(dense)}
        rank_s = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse)}

        all_ids = set(rank_d.keys()) | set(rank_s.keys())
        fused_scores: Dict[int, float] = {}

        for doc_id in all_ids:
            score = 0.0
            if doc_id in rank_d:
                score += self.dense_weight / (self.k0 + 1 + rank_d[doc_id])
            if doc_id in rank_s:
                score += self.sparse_weight / (self.k0 + 1 + rank_s[doc_id])
            fused_scores[doc_id] = score

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[: min(top_k, len(sorted_docs))]


# =====================
# 4. Filter manager (metadata + quantity + nutrition)
# =====================

class FilterManager:
    """Central place to apply / relax filters.

    Supports hard and soft modes. Soft mode will relax filters when
    no candidates are found.
    """

    def __init__(self, recipes: List[Dict[str, Any]], metadata_list: List[Dict[str, Any]]):
        self.recipes = recipes
        self.metadata_list = metadata_list

    def _requirements_from_structured(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        general_ingredient_patterns = {
            'fruit': ['fruit', 'fruits'],
            'vegetable': ['vegetable', 'vegetables', 'veggies'],
            'meat': ['meat', 'meats'],
            'spice': ['spice', 'spices', 'seasoning'],
            'seafood': ['seafood', 'fish', 'shellfish'],
            'dairy': ['dairy', 'milk product', 'milk products'],
        }
        general_terms = set(sum(general_ingredient_patterns.values(), []))
        must_have = []
        for ing in structured.get("must_have_ingredients", []):
            if ing not in general_terms:
                if len(ing.split(" ")) == 1:
                    must_have.append(ing)
                else:
                    # check if any word in the phrase is a general term
                    if not any(word in general_terms for word in ing.split(" ")):
                        must_have.extend(ing.split(" "))

        return {
            "dietary_requirements": set(structured.get("dietary_tags", [])),
            "required_ingredients": set(must_have),
            "excluded_ingredients": set(structured.get("avoid_ingredients", [])),
            "dish_type": set(structured.get("meal_types", [])),
            "cooking_method": set(structured.get("cooking_methods", [])),
            "ingredient_logic": "AND",
            "general_ingredient_tags": set(structured.get("general_ingredient_tags", [])),
            "excluded_general_tags": set(structured.get("excluded_general_tags", [])),
            "nutrition": {},
        }

    def _structured_quantity_constraints(self, structured: Optional[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Convert LLM structured quantity_constraints into the schema
        expected by QuantityFilter.satisfies_constraints.
        """
        if not structured:
            return None
        raw = structured.get("quantity_constraints") or {}
        if not isinstance(raw, dict):
            return None

        constraints: Dict[str, Dict[str, Any]] = {}
        for ing, info in raw.items():
            if not isinstance(info, dict):
                continue
            c: Dict[str, Any] = {}
            if "min" in info:
                try:
                    c["min"] = float(info["min"])
                except (TypeError, ValueError):
                    pass
            if "max" in info:
                try:
                    c["max"] = float(info["max"])
                except (TypeError, ValueError):
                    pass
            if "value" in info and "min" not in c and "max" not in c:
                try:
                    c["value"] = float(info["value"])
                except (TypeError, ValueError):
                    pass
            kind = info.get("kind")
            if isinstance(kind, str) and kind:
                c["kind"] = kind
            if c:
                constraints[ing] = c

        return constraints or None

    def _metadata_candidates(self, requirements: Dict[str, Any]) -> List[int]:
        return [
            i
            for i, metadata in enumerate(self.metadata_list)
            if MetadataFilter.matches_requirements(metadata, requirements)
        ]

    def _relax_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristic relaxation: drop dish_type & cooking_method in soft mode."""
        relaxed = dict(requirements)
        relaxed["dish_type"] = set()
        relaxed["cooking_method"] = set()
        relaxed["general_ingredient_tags"] = set()
        return relaxed

    def apply_filters(
        self,
        query: str,
        config: RetrievalConfig,
        structured_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Return (candidate_indices, requirements_used).

        If structured_query is provided (from LLM), use it to build
        requirements directly; otherwise fall back to MetadataFilter's
        regex-based parsing on the raw query string. QuantityFilter
        is also driven by the raw query to keep the old hybrid path
        working for ablations.
        """
        if structured_query is not None:
            requirements = self._requirements_from_structured(structured_query)
            requirements['nutrition'] = NutritionFilter.parse_nutrition_requirements(query)
        else:
            requirements = MetadataFilter.parse_query_requirements(query)

        no_llm_requirements = MetadataFilter.parse_query_requirements(query)
        print(requirements)
        print(no_llm_requirements)

        # 1. metadata filter
        if config.use_metadata_filter:
            candidate_indices = self._metadata_candidates(requirements)
            if not candidate_indices and config.filter_mode == "soft":
                relaxed = self._relax_requirements(requirements)
                candidate_indices = self._metadata_candidates(relaxed)
                if candidate_indices:
                    requirements = relaxed
        else:
            candidate_indices = list(range(len(self.recipes)))

        # if still empty, fallback to all
        if not candidate_indices:
            candidate_indices = list(range(len(self.recipes)))

        # 2. quantity filter: if LLM structured_query provides
        # quantity_constraints, convert and pass them down; otherwise
        # fall back to regex-based extraction from query text.
        if config.use_quantity_filter:
            structured_constraints = self._structured_quantity_constraints(structured_query)
            candidate_indices = QuantityFilter.filter_recipes_by_quantity(
                self.recipes,
                candidate_indices,
                query,
                tolerance=0.0,
                structured_constraints=structured_constraints,
            )

            # if empty and soft mode: relax quantity by allowing tolerance
            if not candidate_indices and config.filter_mode == "soft":
                candidate_indices = QuantityFilter.filter_recipes_by_quantity(
                    self.recipes,
                    list(range(len(self.recipes))),
                    query,
                    tolerance=1.0,
                    structured_constraints=structured_constraints,
                )

        # final fallback
        if not candidate_indices:
            candidate_indices = list(range(len(self.recipes)))

        return candidate_indices, requirements


# =====================
# 5. Query profiler (rule-based skeleton)
# =====================

class QueryProfiler:
    """Lightweight rule-based query profiler.

    LLM can be plugged in later; for now, we use simple heuristics.
    """

    def profile(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()

        # Heuristic: list-like queries
        is_list_query = any(
            kw in q_lower
            for kw in ["list", "kinds of", "options", "different", "several"]
        )

        # Heuristic: explicit count requirement in text ("5 recipes")
        import re as _re

        m = _re.search(r"(\d+)\s+(recipes?|dishes?|options)", q_lower)
        requested_k = int(m.group(1)) if m else None

        # Heuristic: presence of strong quantity words
        is_strict_quantity = any(
            kw in q_lower
            for kw in ["only", "exactly", "at most", "no more than", "no less than"]
        )

        profile: Dict[str, Any] = {
            "is_list_query": is_list_query,
            "requested_k": requested_k,
            "is_strict_quantity": is_strict_quantity,
        }
        return profile

    def config_from_profile(self, profile: Dict[str, Any]) -> RetrievalConfig:
        cfg = RetrievalConfig()

        # Decide top-k
        if profile.get("requested_k"):
            cfg.final_top_k = max(1, min(20, profile["requested_k"]))
            cfg.first_stage_top_k = max(20, cfg.final_top_k * 5)
        elif profile.get("is_list_query"):
            cfg.final_top_k = 5
            cfg.first_stage_top_k = 50
        else:
            cfg.final_top_k = 3
            cfg.first_stage_top_k = 20

        # Quantity filter mode
        if profile.get("is_strict_quantity"):
            cfg.filter_mode = "hard"
        else:
            cfg.filter_mode = "soft"

        # Simple default: enable everything; reranker left to caller
        cfg.use_dense = True
        cfg.use_sparse = True
        cfg.use_fusion = True
        cfg.use_reranker = False
        cfg.use_cache = True

        return cfg


# =====================
# 6. Semantic cache
# =====================

class SemanticCache:
    """Approximate semantic cache based on query embeddings.

    Keeps everything in memory for now; persistence can be added later.
    """

    def __init__(self, model: SentenceTransformer, threshold: float = 0.95):
        self.model = model
        self.threshold = threshold
        self._embeddings: List[np.ndarray] = []
        self._results: List[RetrievalResult] = []

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim > 1:
            a = a.ravel()
        if b.ndim > 1:
            b = b.ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    def lookup(self, query: str) -> Optional[RetrievalResult]:
        if not self._embeddings:
            return None
        q_emb = self.model.encode(query, convert_to_numpy=True)
        best_sim = -1.0
        best_idx = -1
        for i, emb in enumerate(self._embeddings):
            sim = self._cos_sim(q_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim >= self.threshold and best_idx >= 0:
            cached = self._results[best_idx]
            # mark source as cache for transparency
            cached_from_cache = RetrievalResult(
                query=query,
                items=[
                    RetrievedItem(
                        recipe_index=item.recipe_index,
                        score=item.score,
                        rank=item.rank,
                        source="cache",
                    )
                    for item in cached.items
                ],
                config=cached.config,
                profile=cached.profile,
            )
            return cached_from_cache
        return None

    def store(self, query: str, result: RetrievalResult) -> None:
        q_emb = self.model.encode(query, convert_to_numpy=True)
        self._embeddings.append(q_emb)
        self._results.append(result)


# =====================
# 7. AdvancedRetriever orchestrator
# =====================

class AdvancedRetriever:
    """High-level retriever that wires modules 1-6 together.

    Usage:
        model = SentenceTransformer(EMBED_MODEL)
        recipes = ...
        metadata_list = [MetadataExtractor.extract_metadata(r) for r in recipes]
        embeddings = model.encode([...])

        adv = AdvancedRetriever(recipes, model, embeddings, metadata_list)
        result = adv.retrieve("low carb chicken dinner")
    """

    def __init__(
        self,
        recipes: List[Dict[str, Any]],
        model: SentenceTransformer,
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        enable_cache: bool = True,
    ):
        self.recipes = recipes
        self.model = model

        # Metadata
        if metadata_list is not None:
            self.metadata_list = metadata_list
        else:
            self.metadata_list = [MetadataExtractor.extract_metadata(r) for r in recipes]

        # Core modules
        self.dense = DenseRetriever(embeddings)
        self.sparse = SparseRetriever(recipes, self.metadata_list)
        self.fusion = FusionModule()
        self.filter_manager = FilterManager(recipes, self.metadata_list)
        self.profiler = QueryProfiler()
        self.cache = SemanticCache(model) if enable_cache else None

    # ----- public API -----

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        structured_query: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        # 0. profile & config
        profile = self.profiler.profile(query)
        if config is None:
            config = self.profiler.config_from_profile(profile)

        # 1. semantic cache lookup (only keyed by raw text to keep
        # behaviour comparable with non-LLM runs)
        if config.use_cache and self.cache is not None:
            cached = self.cache.lookup(query)
            if cached is not None:
                cached.profile.setdefault("from_cache", True)
                return cached

        # 2. apply filters (metadata + quantity)
        candidate_indices, requirements_used = self.filter_manager.apply_filters(
            query=query,
            config=config,
            structured_query=structured_query,
        )
        print(f"Number of candidates after filtering: {len(candidate_indices)}")

        # 3. build query embedding
        q_emb = self.model.encode(query, convert_to_numpy=True)

        dense_results: List[Tuple[int, float]] = []
        sparse_results: List[Tuple[int, float]] = []

        # 4. dense retrieval over candidates or full corpus
        if config.use_dense:
            dense_results = self.dense.search(q_emb, config.first_stage_top_k, candidate_indices)

        # 5. sparse retrieval
        if config.use_sparse:
            sparse_results = self.sparse.search(query, config.first_stage_top_k, candidate_indices)

        # print(f"Dense results: {dense_results}")
        # print(f"Sparse results: {sparse_results}")
        # 6. fusion
        fused: List[Tuple[int, float]]
        source = "dense"
        if config.use_fusion and dense_results and sparse_results:
            fused = self.fusion.fuse(dense_results, sparse_results, config.first_stage_top_k)
            source = "fusion"
        elif dense_results:
            fused = dense_results
            source = "dense"
        else:
            fused = sparse_results
            source = "sparse"

        # 7. trim to final_top_k and wrap results
        fused = fused[: config.final_top_k]
        items: List[RetrievedItem] = []
        for rank, (idx, score) in enumerate(fused, start=1):
            items.append(
                RetrievedItem(
                    recipe_index=idx,
                    score=score,
                    rank=rank,
                    source=source,
                )
            )

        result = RetrievalResult(
            query=query,
            items=items,
            config=config,
            profile={"profile": profile, "requirements": requirements_used},
        )

        # 8. store in cache
        if config.use_cache and self.cache is not None:
            self.cache.store(query, result)

        return result


__all__ = [
    "RetrievalConfig",
    "RetrievedItem",
    "RetrievalResult",
    "DenseRetriever",
    "SparseRetriever",
    "FusionModule",
    "FilterManager",
    "QueryProfiler",
    "SemanticCache",
    "AdvancedRetriever",
]
